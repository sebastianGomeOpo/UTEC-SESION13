import json

# Project Imports
from agents.graph_state import GraphState
from config.settings import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_context(state: GraphState) -> GraphState:
    """
    Nodo: load_context

    Carga el perfil del usuario desde JSON y lo añade al state.

    Args:
      state (GraphState): Estado del grafo con user_id

    Returns:
      GraphState: Estado actualizado con perfil_usuario lleno o error

    Raises:
      Ninguno (errores van a state["error"])
    """
    logger.info("--- Entering Load Context Node ---")
    user_id = state.get("user_id")

    if not user_id:
        logger.error("User ID missing in state.")
        state["error"] = "User ID es requerido en el estado inicial."
        state["step_completed"] = "load_context_error"
        return state

    try:
        user_file_path = Config.USERS_DIR / f"{user_id}.json"
        logger.info(f"Attempting to load user profile from: {user_file_path}")

        if not user_file_path.exists():
            logger.error(f"User file not found for user_id: {user_id} at {user_file_path}")
            state["error"] = f"Usuario '{user_id}' no encontrado en {Config.USERS_DIR}"
            state["step_completed"] = "load_context_error"
            return state

        with open(user_file_path, "r", encoding="utf-8") as f:
            user_profile = json.load(f)

        # Validate required fields
        required_fields = ["level", "objetivo"]
        missing_fields = [field for field in required_fields if field not in user_profile]

        if missing_fields:
            logger.error(f"User profile for {user_id} is incomplete. Missing fields: {missing_fields}")
            state["error"] = f"Perfil incompleto: falta {', '.join(missing_fields)}"
            state["step_completed"] = "load_context_error"
            return state

        # Successfully loaded and validated
        state["perfil_usuario"] = user_profile
        state["step_completed"] = "context_loaded"
        logger.info(f"Successfully loaded profile for user: {user_id}")

    except json.JSONDecodeError as e:
        logger.exception(f"Failed to decode JSON for user {user_id}: {e}")
        state["error"] = f"Archivo corrupto para usuario '{user_id}': {str(e)}"
        state["step_completed"] = "load_context_error"
    except Exception as e:
        logger.exception(f"An unexpected error occurred loading context for user {user_id}: {e}")
        state["error"] = f"Error inesperado cargando perfil de '{user_id}': {str(e)}"
        state["step_completed"] = "load_context_error"

    logger.info("--- Exiting Load Context Node ---")
    return state