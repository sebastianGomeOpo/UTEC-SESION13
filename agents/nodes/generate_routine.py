import time
from pathlib import Path
from typing import Dict, Any

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableSequence

# Project Imports
from agents.graph_state import GraphState
from rag.models import RutinaActiva, Sesion, Ejercicio, PrincipiosExtraidos
from config.settings import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Constants
MAX_RETRIES = 3

def _validate_generated_routine(rutina: RutinaActiva, principios: PrincipiosExtraidos, perfil: Dict[str, Any]) -> str | None:
    """Performs deep validation of the generated routine against principles and logistics."""
    logger.debug("Validating generated routine...")

    if not rutina.sesiones:
        return "Rutina generada no tiene sesiones."

    for i, sesion in enumerate(rutina.sesiones):
        if not sesion.ejercicios:
            return f"Sesión {i+1} ('{sesion.dia_semana}') no tiene ejercicios."

        for j, ejercicio in enumerate(sesion.ejercicios):
            # Validate against principles (only for 'principal' type for flexibility)
            if ejercicio.tipo == "principal":
                
                # --- INICIO DE VALIDACIÓN DE PRINCIPIOS (Manejo de None) ---
                # Esta validación ahora debe ser robusta ante un None de RAG
                if principios.intensidad_RIR is None:
                    return f"Principios incompletos: 'intensidad_RIR' es None."
                if principios.cadencia_tempo is None:
                    return f"Principios incompletos: 'cadencia_tempo' es None."
                # --- FIN DE VALIDACIÓN DE PRINCIPIOS ---

                if ejercicio.RIR != principios.intensidad_RIR:
                    return (f"Ejercicio '{ejercicio.nombre}' (Sesión {i+1}) tiene RIR='{ejercicio.RIR}', "
                            f"pero los principios requieren RIR='{principios.intensidad_RIR}'.")
                # Add validation for reps range if needed, requires parsing the string range
                # Example: if not _is_rep_range_compatible(ejercicio.reps, principios.rango_repeticiones): return "Reps inconsistentes"

                if ejercicio.tempo != principios.cadencia_tempo:
                    return (f"Ejercicio '{ejercicio.nombre}' (Sesión {i+1}) tiene tempo='{ejercicio.tempo}', "
                            f"pero los principios requieren tempo='{principios.cadencia_tempo}'.")

    # Validate against logistics (example: duration)
    preferencias = perfil.get("preferencias_logistica", {})
    duracion_max_min = preferencias.get("duracion_sesion_min", 60) # Assuming this is max, rename if needed

    # Basic duration estimation (can be refined)
    total_estimated_duration = sum(s.duracion_estimada_min for s in rutina.sesiones)
    average_session_duration = total_estimated_duration / len(rutina.sesiones) if rutina.sesiones else 0

    logger.debug(f"Comparing average session duration ({average_session_duration:.1f} min) with max allowed ({duracion_max_min} min)")
    # Check if *average* exceeds max, or check *each session*
    for sesion in rutina.sesiones:
        if sesion.duracion_estimada_min > duracion_max_min:
            return (f"Sesión '{sesion.dia_semana}' ({sesion.duracion_estimada_min} min) "
                    f"excede la duración máxima permitida ({duracion_max_min} min).")

    # Validate inclusion of ECIs
    required_eci_names = {eci.nombre_ejercicio for eci in principios.ECI_recomendados}
    found_eci_names = {ej.nombre for s in rutina.sesiones for ej in s.ejercicios if ej.tipo == "ECI"}
    missing_ecis = required_eci_names - found_eci_names
    if missing_ecis:
        return f"Rutina generada omite ECIs obligatorios: {', '.join(missing_ecis)}."


    logger.debug("Routine validation passed.")
    return None # No errors

def generate_routine(state: GraphState) -> GraphState:
    """
    Nodo: generate_routine

    Genera una rutina de entrenamiento estructurada (RutinaActiva) utilizando
    un LLM, basándose en los principios extraídos y el perfil del usuario.
    Incluye lógica de reintentos y validación profunda.

    Args:
      state (GraphState): Estado con principios_libro y perfil_usuario llenos.

    Returns:
      GraphState: Estado actualizado con rutina_final o error.

    Raises:
      Ninguno (errores van a state["error"]).
    """
    logger.info("--- Entering Generate Routine Node ---")
    start_time = time.time()

    principios: PrincipiosExtraidos | None = state.get("principios_libro")
    perfil_usuario: Dict[str, Any] | None = state.get("perfil_usuario")

    if not principios:
        logger.error("'principios_libro' missing in state.")
        state["error"] = "Principios del libro no disponibles para generar rutina."
        state["step_completed"] = "generate_routine_error"
        return state
    if not perfil_usuario:
        logger.error("'perfil_usuario' missing in state.")
        state["error"] = "Perfil de usuario no disponible para generar rutina."
        state["step_completed"] = "generate_routine_error"
        return state

    try:
        config = Config()
        # 1. Load Prompt Template
        prompt_file_path = config.PROMPTS_DIR / "routine_assembler.txt"
        logger.info(f"Loading routine assembler prompt from: {prompt_file_path}")
        if not prompt_file_path.exists():
            state["error"] = f"Archivo de prompt no encontrado: {prompt_file_path}"
            state["step_completed"] = "generate_routine_error"
            return state
        raw_prompt_template = prompt_file_path.read_text(encoding="utf-8")

        # 2. Build LCEL Chain
        prompt_template = ChatPromptTemplate.from_template(raw_prompt_template)
        llm = ChatOpenAI(model=config.LLM_MODEL_ASSEMBLE, temperature=0.0)
        output_parser = PydanticOutputParser(pydantic_object=RutinaActiva)

        # Cadena tradicional: Prompt → LLM → Parser (más estable)
        chain: RunnableSequence = prompt_template | llm | output_parser
        logger.info("Routine generation chain built.")

        # 3. Prepare Prompt Variables
        preferencias_logistica = perfil_usuario.get("preferencias_logistica", {})
        prompt_variables = {
            "principios": principios.model_dump(),
            "perfil": perfil_usuario,
            "preferencias": preferencias_logistica,
            "format_instructions": output_parser.get_format_instructions()  # ✅ CRÍTICO
        }
        logger.debug(f"Prompt variables prepared: {list(prompt_variables.keys())}")

        # 4. Invoke Chain with Retry Logic
        generated_routine: RutinaActiva | None = None
        last_exception = None
        for attempt in range(MAX_RETRIES):
            logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES} to generate routine...")
            try:
                generated_routine = chain.invoke(prompt_variables)
                logger.info("LLM invocation successful.")
                # If successful, break the loop
                break
            except OutputParserException as e:
                logger.warning(f"Attempt {attempt + 1} failed: LLM output did not match Pydantic schema. Error: {e}")
                last_exception = e
                time.sleep(1) # Simple backoff
            except Exception as e:
                # Catch potential API errors, timeouts etc.
                logger.exception(f"Attempt {attempt + 1} failed with unexpected error: {e}")
                last_exception = e
                # Depending on error type, might break early or retry
                if "timeout" in str(e).lower(): # Example of breaking early on timeout
                    break
                time.sleep(1)

        if not generated_routine:
            logger.error(f"Failed to generate valid routine after {MAX_RETRIES} attempts.")
            error_detail = str(last_exception) if last_exception else "Unknown error during generation."
            if isinstance(last_exception, OutputParserException):
                state["error"] = f"LLM no retornó JSON válido después de {MAX_RETRIES} intentos: {error_detail}"
            else:
                state["error"] = f"Error generando rutina después de {MAX_RETRIES} intentos: {error_detail}"
            state["step_completed"] = "generate_routine_error"
            return state

        # 5. Validate Generated Routine
        validation_error = _validate_generated_routine(generated_routine, principios, perfil_usuario)
        if validation_error:
            logger.error(f"Generated routine failed validation: {validation_error}")
            state["error"] = f"Validación fallida: {validation_error}"
            state["step_completed"] = "generate_routine_error"
            
            # --- ✅ INICIO DE CORRECCIÓN (Problema 2) ---
            # Asegurarse de que debug_info exista antes de asignarle una clave
            if state.get("debug_info") is None:
                state["debug_info"] = {}
            # --- FIN DE CORRECCIÓN ---
            
            state["debug_info"]["invalid_generated_routine"] = generated_routine.model_dump()
            return state

        # 6. Success
        generation_time = time.time() - start_time
        logger.info(f"Successfully generated and validated routine in {generation_time:.2f} seconds.")
        state["rutina_final"] = generated_routine
        state["step_completed"] = "routine_generated" # Corrected step name
        
        # Add metadata - Asegurar que debug_info existe
        # --- ✅ INICIO DE CORRECCIÓN (Problema 2) ---
        if not isinstance(state.get("debug_info"), dict):
            state["debug_info"] = {}
        # --- FIN DE CORRECCIÓN ---

        metadata = {
            "tiempo_generacion_segundos": round(generation_time, 2),
            "modelo_usado": config.LLM_MODEL_ASSEMBLE,
            "temperatura_llm": 0.0,
            "validation_passed": True
        }
        state["debug_info"]["generation_metadata"] = metadata


    except FileNotFoundError as e:
        logger.error(f"Prompt file error: {e}")
        state["error"] = "Archivo de prompt de generación no encontrado."
        state["step_completed"] = "generate_routine_error"
    except Exception as e:
        logger.exception(f"An unexpected error occurred during routine generation: {e}")
        state["error"] = f"Error inesperado generando rutina: {str(e)}"
        state["step_completed"] = "generate_routine_error"

    logger.info("--- Exiting Generate Routine Node ---")
    return state