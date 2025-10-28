import json

# Project Imports
from agents.graph_state import GraphState
from config.settings import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_context(state: GraphState) -> GraphState:
    """
    Nodo: load_context

    Valida el request_type y carga el perfil del usuario desde JSON.
    
    CRÍTICO (Fase 5 Fix): Este nodo ahora valida request_type ANTES de cargar
    el perfil, ya que los routers NO pueden setear errores en el estado.

    Args:
      state (GraphState): Estado del grafo con user_id y request_type

    Returns:
      GraphState: Estado actualizado con perfil_usuario lleno o error

    Raises:
      Ninguno (errores van a state["error"])
    """
    logger.info("--- Entering Load Context Node ---")
    
    # ════════════════════════════════════════════════════════════════════
    # VALIDACIÓN 1: REQUEST_TYPE (NUEVO - Movido desde router)
    # ════════════════════════════════════════════════════════════════════
    request_type = state.get("request_type", "unknown")
    VALID_REQUEST_TYPES = ["crear_rutina", "registrar_ejercicio", "consultar_historial"]
    
    if request_type not in VALID_REQUEST_TYPES:
        logger.error(f"Request type inválido: '{request_type}'. Válidos: {VALID_REQUEST_TYPES}")
        state["error"] = f"Tipo de request desconocido: {request_type}"
        state["step_completed"] = "load_context_error"
        logger.info("--- Exiting Load Context Node (request_type inválido) ---")
        return state
    
    logger.debug(f"Request type válido: '{request_type}'")
    
    # ════════════════════════════════════════════════════════════════════
    # VALIDACIÓN 2: USER_ID
    # ════════════════════════════════════════════════════════════════════
    user_id = state.get("user_id")

    if not user_id:
        logger.error("User ID missing in state.")
        state["error"] = "User ID es requerido en el estado inicial."
        state["step_completed"] = "load_context_error"
        logger.info("--- Exiting Load Context Node (user_id ausente) ---")
        return state

    # ════════════════════════════════════════════════════════════════════
    # CARGA DE PERFIL DESDE JSON
    # ════════════════════════════════════════════════════════════════════
    try:
        user_file_path = Config.USERS_DIR / f"{user_id}.json"
        logger.info(f"Attempting to load user profile from: {user_file_path}")

        # Validar existencia del archivo
        if not user_file_path.exists():
            logger.error(f"User file not found for user_id: {user_id} at {user_file_path}")
            state["error"] = f"Usuario '{user_id}' no encontrado en {Config.USERS_DIR}"
            state["step_completed"] = "load_context_error"
            logger.info("--- Exiting Load Context Node (archivo no encontrado) ---")
            return state

        # Cargar JSON
        with open(user_file_path, "r", encoding="utf-8") as f:
            user_profile = json.load(f)

        # ════════════════════════════════════════════════════════════════
        # VALIDACIÓN 3: CAMPOS REQUERIDOS EN PERFIL
        # ════════════════════════════════════════════════════════════════
        required_fields = ["level", "objetivo"]
        missing_fields = [field for field in required_fields if field not in user_profile]

        if missing_fields:
            logger.error(f"User profile for {user_id} is incomplete. Missing fields: {missing_fields}")
            state["error"] = f"Perfil incompleto: falta {', '.join(missing_fields)}"
            state["step_completed"] = "load_context_error"
            logger.info("--- Exiting Load Context Node (perfil incompleto) ---")
            return state

        # ════════════════════════════════════════════════════════════════
        # ÉXITO: POBLAR ESTADO
        # ════════════════════════════════════════════════════════════════
        state["perfil_usuario"] = user_profile
        state["step_completed"] = "context_loaded"
        logger.info(f"Successfully loaded profile for user: {user_id}")
        logger.debug(f"Profile summary: level={user_profile.get('level')}, objetivo={user_profile.get('objetivo')}")

    except json.JSONDecodeError as e:
        logger.exception(f"Failed to decode JSON for user {user_id}: {e}")
        state["error"] = f"Archivo corrupto para usuario '{user_id}': {str(e)}"
        state["step_completed"] = "load_context_error"
        logger.info("--- Exiting Load Context Node (JSON corrupto) ---")
        return state
        
    except Exception as e:
        logger.exception(f"An unexpected error occurred loading context for user {user_id}: {e}")
        state["error"] = f"Error inesperado cargando perfil de '{user_id}': {str(e)}"
        state["step_completed"] = "load_context_error"
        logger.info("--- Exiting Load Context Node (error inesperado) ---")
        return state

    logger.info("--- Exiting Load Context Node ---")
    return state