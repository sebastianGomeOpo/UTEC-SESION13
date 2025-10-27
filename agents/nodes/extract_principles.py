from typing import Dict, Any

# LangChain and Project Imports
from agents.graph_state import GraphState
from rag.principle_extractor import PrincipleExtractor
from rag.models import PrincipiosExtraidos # Assuming PrincipiosExtraidos is here
from utils.logger import setup_logger
# Assuming Config provides necessary paths/keys, though not directly used here
# from config.settings import Config

logger = setup_logger(__name__)

def extract_principles(state: GraphState) -> GraphState:
    """
    Nodo: extract_principles

    Llama a RAG (PrincipleExtractor) para obtener principios de entrenamiento
    basados en el perfil del usuario, asegurando que incluyan citas.

    Args:
      state (GraphState): Estado con perfil_usuario lleno.

    Returns:
      GraphState: Estado actualizado con principios_libro o error.

    Raises:
      Ninguno (errores van a state["error"]).
    """
    logger.info("--- Entering Extract Principles Node ---")
    perfil_usuario = state.get("perfil_usuario")

    if not perfil_usuario:
        logger.error("User profile ('perfil_usuario') is missing in state.")
        state["error"] = "Perfil de usuario no disponible para extraer principios."
        state["step_completed"] = "extract_principles_error"
        return state

    try:
        logger.info("Initializing PrincipleExtractor...")
        extractor = PrincipleExtractor()
        extraction_chain = extractor.get_extraction_chain()
        logger.info("Extraction chain obtained. Invoking...")

        # Invoke the chain with the user profile
        # The chain expects the input that matches its input schema,
        # which is {"perfil_usuario": ..., "contexto_libro": retriever}
        # RunnablePassthrough() in the chain means perfil_usuario is passed directly.
        principios: PrincipiosExtraidos = extraction_chain.invoke(perfil_usuario)

        if not principios:
            logger.error("PrincipleExtractor chain returned a null result.")
            state["error"] = "RAG retornó resultado nulo al extraer principios."
            state["step_completed"] = "extract_principles_error"
            return state

        # CRITICAL VALIDATION: Check for citations
        if not principios.citas_fuente:
            logger.error("CRITICAL: Extracted principles lack source citations!")
            state["error"] = "Alucinación detectada: principios extraídos sin citas de fuente."
            state["step_completed"] = "extract_principles_error"
            # ✅ FIX: Inicializar debug_info si es None
            if state.get("debug_info") is None:
                state["debug_info"] = {}
            state["debug_info"]["principles_without_citations"] = principios.model_dump()
            return state

        # Optional: Confidence check (if your model provides it, adapt as needed)
        # Assuming confidence isn't part of PrincipiosExtraidos based on Fase 1 spec
        # confidence = getattr(principios, 'confianza', 1.0) # Example placeholder
        # if confidence < 0.3:
        #     logger.warning(f"Low confidence ({confidence:.2f}) in extracted principles for user {state.get('user_id')}")
        #     if state.get("debug_info") is None: state["debug_info"] = {}
        #     state["debug_info"]["low_confidence_principles"] = principios.model_dump()

        # Success
        state["principios_libro"] = principios
        state["step_completed"] = "principles_extracted" # Corrected step name
        logger.info(f"Successfully extracted principles for user: {state.get('user_id')}")
        # Log extracted principles summary for auditability
        logger.debug(f"Extracted Principles Summary: RIR={principios.intensidad_RIR}, Reps={principios.rango_repeticiones}, ECIs={len(principios.ECI_recomendados)}, Citations={len(principios.citas_fuente)}")


    except ImportError as e:
         logger.exception(f"Import error, likely missing RAG components: {e}")
         state["error"] = "Error de importación, componentes RAG podrían faltar."
         state["step_completed"] = "extract_principles_error"
    except Exception as e:
        # Catch potential API errors, timeouts, etc. from the chain invocation
        logger.exception(f"Error invoking principle extraction chain: {e}")
        state["error"] = f"Error de API o RAG al extraer principios: {str(e)}"
        state["step_completed"] = "extract_principles_error"

    logger.info("--- Exiting Extract Principles Node ---")
    return state
