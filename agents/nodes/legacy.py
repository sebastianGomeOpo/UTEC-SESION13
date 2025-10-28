# FILENAME: agents/nodes/legacy.py
# -----------------------------------------------------------------------------
# Este archivo contiene nodos para compatibilidad con el agente "legacy"
# (Fase 1), permitiendo al grafo C-R-G invocar las herramientas
# antiguas de registro y consulta.
# -----------------------------------------------------------------------------

from typing import Optional

# LangChain and Project Imports
from agents.graph_state import GraphState
from utils.logger import setup_logger
from config.settings import Config

# --- 💡 MODIFICACIÓN: Nuevas importaciones para parseo con LLM 💡 ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
# ------------------------------------------------------------------


# Importaciones de herramientas Legacy y su contexto
# CRÍTICO: Los nodos legacy deben configurar el UserContext
# tal como lo hacía el TrainingToolsFactory
try:
    from tools._base import UserContext, set_user_context
    from tools.registro import registrar_ejercicio, EjercicioEstructurado
    from tools.historial import consultar_historial
except ImportError as e:
    print(f"Error importando herramientas legacy: {e}. Los nodos legacy fallarán.")
    # Permite que el módulo se importe pero las funciones fallen
    UserContext = None
    set_user_context = None
    registrar_ejercicio = None
    consultar_historial = None
    EjercicioEstructurado = None


logger = setup_logger(__name__)

# -----------------------------------------------------------------------------
# NODO: CALL LEGACY REGISTER
# -----------------------------------------------------------------------------


def _parse_with_llm(user_message: str) -> Optional[EjercicioEstructurado]:
    """
    Usa un LLM para extraer datos de ejercicio del lenguaje natural.
    Retorna un objeto EjercicioEstructurado o None si falla.
    """
    logger.debug(f"Intentando parsear con LLM: '{user_message}'")

    # 1. Definir el parser Pydantic
    parser = PydanticOutputParser(pydantic_object=EjercicioEstructurado)

    # 2. Definir el Prompt
    prompt_template = """
    Eres un asistente que extrae información de ejercicios de un texto en lenguaje natural.
    Analiza el siguiente texto del usuario y extrae SOLAMENTE los 4 campos requeridos.
    
    Texto del usuario:
    "{user_message}"
    
    Campos a extraer:
    - ejercicio: str (ej. "sentadilla", "press de banca")
    - series: int
    - repeticiones: int
    - peso_kg: float
    
    Ejemplo 1:
    Usuario: "acabo de terminar 5 series de 10 reps haciendo sentadilla con 180 kg"
    JSON: {{"ejercicio": "sentadilla", "series": 5, "repeticiones": 10, "peso_kg": 180.0}}
    
    Ejemplo 2:
    Usuario: "hice 3x12 de curl de bíceps con 20"
    JSON: {{"ejercicio": "curl de bíceps", "series": 3, "repeticiones": 12, "peso_kg": 20.0}}
    
    Ejemplo 3:
    Usuario: "registra 50 lagartijas"
    JSON: {{"ejercicio": "lagartijas", "series": 1, "repeticiones": 50, "peso_kg": 0.0}} (Asume 1 serie y 0kg si no se especifica)

    Formato de salida (SOLO JSON):
    {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # 3. Definir el LLM (usando el modelo de extracción de la config)
    try:
        llm = ChatOpenAI(model=Config.LLM_MODEL_EXTRACT, temperature=0.0)
    except Exception as e:
        logger.error(f"No se pudo inicializar ChatOpenAI (¿API key?): {e}")
        return None

    # 4. Crear y ejecutar la cadena
    chain = prompt | llm | parser

    try:
        parsed_data = chain.invoke({"user_message": user_message})
        logger.info(f"LLM parseó exitosamente: {parsed_data}")
        return parsed_data
    except OutputParserException as e:
        logger.warning(f"Error del parser LLM: {e}. Mensaje: '{user_message}'")
        return None
    except Exception as e:
        logger.error(f"Error inesperado en la cadena de parseo LLM: {e}")
        return None


def call_legacy_register(state: GraphState) -> GraphState:
    """
    Nodo: call_legacy_register

    Wrapper para la herramienta legacy 'registrar_ejercicio'.
    Usa un LLM para parsear el user_message.

    Args:
      state (GraphState): Estado con user_id y user_message.

    Returns:
      GraphState: Estado actualizado con respuesta_usuario o error.
    """
    logger.info("--- Entering Legacy Register Node ---")
    user_id = state.get("user_id")
    user_message = state.get("user_message", "")

    if not user_id:
        state["error"] = "User ID faltante para call_legacy_register"
        state["step_completed"] = "call_legacy_register_error"
        return state

    if not UserContext or not set_user_context or not registrar_ejercicio or not EjercicioEstructurado:
        state["error"] = "Componentes legacy (UserContext, tools) no importados o no disponibles."
        state["step_completed"] = "call_legacy_register_error"
        logger.error(state["error"])
        return state

    try:
        # --- 💡 MODIFICACIÓN: Llamar al parser LLM 💡 ---
        logger.info(f"Iniciando parseo con LLM para: '{user_message}'")
        ejercicio_data = _parse_with_llm(user_message)

        if not ejercicio_data:
            logger.warning(f"El LLM no pudo parsear el mensaje: {user_message}")
            state["error"] = "No pude entender los detalles de tu ejercicio. ¿Puedes refrasearlo?"
            state["step_completed"] = "call_legacy_register_error"
            return state
        # ----------------------------------------------------

        # 2. Configurar el contexto legacy
        logger.debug(f"Configurando UserContext legacy para: {user_id}")
        context = UserContext(user_id=user_id)
        set_user_context(context)

        # 3. Preparar datos y llamar a la herramienta usando .invoke()
        # 'ejercicio_data' YA ES un objeto EjercicioEstructurado
        logger.info(f"Invocando herramienta legacy registrar_ejercicio para {user_id}")
        respuesta_tool = registrar_ejercicio.invoke({"datos_ejercicio": ejercicio_data})

        # 4. Éxito
        state["respuesta_usuario"] = respuesta_tool # La herramienta ya devuelve "✅ Registrado: ..."
        state["step_completed"] = "call_legacy_register"
        logger.info(f"Registro legacy exitoso para {user_id}")

    except Exception as e:
        logger.exception(f"Error en call_legacy_register: {e}")
        state["error"] = f"Error interno al registrar ejercicio: {str(e)}"
        state["step_completed"] = "call_legacy_register_error"

    logger.info("--- Exiting Legacy Register Node ---")
    return state


# -----------------------------------------------------------------------------
# NODO: CALL LEGACY QUERY
# -----------------------------------------------------------------------------

def call_legacy_query(state: GraphState) -> GraphState:
    """
    Nodo: call_legacy_query

    Wrapper para la herramienta legacy 'consultar_historial'.
    Configura el UserContext requerido por la herramienta.

    Args:
      state (GraphState): Estado con user_id.

    Returns:
      GraphState: Estado actualizado con respuesta_usuario o error.
    """
    logger.info("--- Entering Legacy Query Node ---")
    user_id = state.get("user_id")

    if not user_id:
        state["error"] = "User ID faltante para call_legacy_query"
        state["step_completed"] = "call_legacy_query_error"
        return state

    if not UserContext or not set_user_context or not consultar_historial:
        state["error"] = "Componentes legacy (UserContext, consultar_historial) no importados o no disponibles."
        state["step_completed"] = "call_legacy_query_error"
        logger.error(state["error"])
        return state

    try:
        # 1. Configurar el contexto legacy
        logger.debug(f"Configurando UserContext legacy para: {user_id}")
        context = UserContext(user_id=user_id)
        set_user_context(context)

        # 2. Llamar a la herramienta usando .invoke()
        logger.info(f"Invocando herramienta legacy consultar_historial para {user_id}")
        # --- CORRECCIÓN TOOL CALL ---
        respuesta_tool = consultar_historial.invoke({"ultimos_n": 7})

        # 3. Éxito
        state["respuesta_usuario"] = respuesta_tool # La herramienta ya devuelve el historial formateado
        state["step_completed"] = "call_legacy_query"
        logger.info(f"Consulta legacy exitosa para {user_id}")

    except Exception as e:
        logger.exception(f"Error en call_legacy_query: {e}")
        state["error"] = f"Error interno al consultar historial: {str(e)}"
        state["step_completed"] = "call_legacy_query_error"

    logger.info("--- Exiting Legacy Query Node ---")
    return state