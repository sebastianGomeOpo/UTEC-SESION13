from typing import Dict, Any

# LangChain and Project Imports
from agents.graph_state import GraphState
from rag.principle_extractor import PrincipleExtractor
from rag.models import PrincipiosExtraidos
from utils.logger import setup_logger

logger = setup_logger(__name__)

def extract_principles(state: GraphState) -> GraphState:
    """
    Nodo: extract_principles

    Llama a RAG (PrincipleExtractor) para obtener principios de entrenamiento
    basados en el perfil del usuario, asegurando que incluyan citas.

    VALIDACIONES CRÍTICAS:
    1. Perfil de usuario presente en estado
    2. RAG retorna resultado válido (no None)
    3. Principios contienen citas de fuente (anti-alucinación)

    Args:
      state (GraphState): Estado con perfil_usuario lleno.

    Returns:
      GraphState: Estado actualizado con principios_libro o error.

    Raises:
      Ninguno (errores van a state["error"]).
    """
    logger.info("--- Entering Extract Principles Node ---")
    perfil_usuario = state.get("perfil_usuario")

    # ════════════════════════════════════════════════════════════════════
    # VALIDACIÓN 1: Verificar que perfil_usuario existe
    # ════════════════════════════════════════════════════════════════════
    if not perfil_usuario:
        logger.error("User profile ('perfil_usuario') is missing in state.")
        state["error"] = "Perfil de usuario no disponible para extraer principios."
        state["step_completed"] = "extract_principles_error"
        return state

    try:
        # ════════════════════════════════════════════════════════════════
        # PASO 1: Inicializar PrincipleExtractor
        # ════════════════════════════════════════════════════════════════
        logger.info("Initializing PrincipleExtractor...")
        extractor = PrincipleExtractor()
        extraction_chain = extractor.get_extraction_chain()
        logger.info("Extraction chain obtained. Invoking...")

        # ════════════════════════════════════════════════════════════════
        # PASO 2: Invocar cadena RAG
        # ════════════════════════════════════════════════════════════════
        principios: PrincipiosExtraidos = extraction_chain.invoke(perfil_usuario)

        # ════════════════════════════════════════════════════════════════
        # VALIDACIÓN 2: Verificar que RAG retornó resultado válido
        # ════════════════════════════════════════════════════════════════
        if not principios:
            logger.error("PrincipleExtractor chain returned a null result.")
            state["error"] = "RAG retornó resultado nulo al extraer principios."
            state["step_completed"] = "extract_principles_error"
            return state

        # ════════════════════════════════════════════════════════════════
        # VALIDACIÓN 3: CRÍTICA - Verificar presencia de citas (anti-alucinación)
        # ════════════════════════════════════════════════════════════════
        # Si no hay citas, el LLM pudo haber alucinado los principios.
        # DEBE fallar aquí para prevenir rutinas basadas en datos inventados.
        if not principios.citas_fuente:
            logger.error("CRITICAL: Extracted principles lack source citations!")
            state["error"] = "Alucinación detectada: principios extraídos sin citas de fuente."
            state["step_completed"] = "extract_principles_error"
            
            # Guardar debug info para análisis posterior
            if state.get("debug_info") is None:
                state["debug_info"] = {}
            state["debug_info"]["principles_without_citations"] = principios.model_dump()
            
            return state

        # ════════════════════════════════════════════════════════════════
        # PASO 3: Validaciones opcionales (confianza, si existe)
        # ════════════════════════════════════════════════════════════════
        # Nota: PrincipiosExtraidos no tiene campo 'confianza' en el modelo actual
        # Si se agrega en el futuro, descomentar:
        #
        # confidence = getattr(principios, 'confianza', 1.0)
        # if confidence < 0.3:
        #     logger.warning(f"Low confidence ({confidence:.2f}) in extracted principles for user {state.get('user_id')}")
        #     if state.get("debug_info") is None:
        #         state["debug_info"] = {}
        #     state["debug_info"]["low_confidence_principles"] = principios.model_dump()

        # ════════════════════════════════════════════════════════════════
        # ÉXITO: Guardar principios en estado
        # ════════════════════════════════════════════════════════════════
        state["principios_libro"] = principios
        state["step_completed"] = "principles_extracted"
        logger.info(f"Successfully extracted principles for user: {state.get('user_id')}")
        
        # Log de auditoría (para debugging)
        logger.debug(
            f"Extracted Principles Summary: "
            f"RIR={principios.intensidad_RIR}, "
            f"Reps={principios.rango_repeticiones}, "
            f"ECIs={len(principios.ECI_recomendados)}, "
            f"Citations={len(principios.citas_fuente)}"
        )

    except ImportError as e:
        # Error de dependencias faltantes
        logger.exception(f"Import error, likely missing RAG components: {e}")
        state["error"] = "Error de importación, componentes RAG podrían faltar."
        state["step_completed"] = "extract_principles_error"
        
    except Exception as e:
        # Errores generales (API timeout, errores de LLM, etc.)
        logger.exception(f"Error invoking principle extraction chain: {e}")
        state["error"] = f"Error de API o RAG al extraer principios: {str(e)}"
        state["step_completed"] = "extract_principles_error"

    logger.info("--- Exiting Extract Principles Node ---")
    return state