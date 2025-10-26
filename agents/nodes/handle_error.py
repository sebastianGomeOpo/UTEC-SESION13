from typing import Dict, Any

# Project Imports
from agents.graph_state import GraphState
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Mapping from technical errors (substrings) to user-friendly messages
ERROR_MESSAGE_MAP = {
    "Usuario no encontrado": "Tu perfil de usuario no se encontró en el sistema.",
    "Archivo corrupto": "Hubo un problema al leer los datos de tu perfil. Podría estar corrupto.",
    "Perfil incompleto": "A tu perfil le faltan datos esenciales (como nivel u objetivo).",
    "RAG retornó resultado nulo": "No pude obtener los principios de entrenamiento del libro en este momento.",
    "Alucinación detectada": "No pude verificar la información extraída del libro (faltan citas). No puedo proceder de forma segura.",
    "Error de API o RAG": "Hubo un problema comunicándome con el sistema de extracción de información.",
    "Archivo de prompt no encontrado": "Error interno: falta una plantilla necesaria para generar la rutina.",
    "LLM no retornó JSON válido": "La IA generó una respuesta en un formato inesperado y no se pudo procesar.",
    "Validación fallida": "La rutina generada no parece cumplir con los principios o tus preferencias.",
    "RIR inconsistente": "La rutina generada no respeta el RIR recomendado por el libro.",
    "Reps inconsistentes": "La rutina generada no respeta las repeticiones recomendadas por el libro.",
    "Rutina excede duración": "La rutina generada es demasiado larga para el tiempo que especificaste.",
    "Rutina generada omite ECIs": "La rutina generada no incluyó ejercicios compensatorios necesarios.",
    "Rutina final vacía": "No se generó una rutina válida para guardar.",
    "Error de archivo guardando": "Hubo un problema al intentar guardar la rutina en tu perfil.",
    "Permisos insuficientes": "Error del sistema al intentar guardar (problema de permisos).",
    "Espacio en disco insuficiente": "Error del sistema al intentar guardar (espacio en disco lleno).",
    "Verificación de persistencia falló": "Error interno al verificar que la rutina se guardó correctamente.",
}

DEFAULT_ERROR_MESSAGE = "Ocurrió un error inesperado procesando tu solicitud."

def handle_error(state: GraphState) -> GraphState:
    """
    Nodo: handle_error

    Captura el error técnico almacenado en state["error"], lo loguea
    detalladamente, selecciona un mensaje user-friendly y lo pone en
    state["respuesta_usuario"].

    Args:
      state (GraphState): Estado que contiene un valor no vacío en state["error"].

    Returns:
      GraphState: Estado actualizado con respuesta_usuario y step_completed='error'.

    Raises:
      Ninguno (este nodo es el último recurso y no debe fallar).
    """
    logger.info("--- Entering Handle Error Node ---")
    technical_error_msg = state.get("error", "Error desconocido")
    failed_step = state.get("step_completed", "paso desconocido") # Step where error occurred

    # 1. Log técnico completo
    logger.error(f"Error captured in step '{failed_step}': {technical_error_msg}")
    # Log more state context for debugging (be mindful of sensitive data if applicable)
    log_state = {k: v for k, v in state.items() if k not in ['principios_libro', 'rutina_final', 'perfil_usuario']} # Avoid logging large objects
    logger.debug(f"Full state context at error (partial): {log_state}")
    # logger.error(f"Full state at error: {state}") # Uncomment for deeper debugging if needed


    # 2. Mapeo a mensaje público
    user_friendly_message = DEFAULT_ERROR_MESSAGE
    for key, message in ERROR_MESSAGE_MAP.items():
        if key.lower() in technical_error_msg.lower():
            user_friendly_message = message
            break # Use the first match found

    # 3. Formatear respuesta final para el usuario
    state["respuesta_usuario"] = f"❌ Lo siento, hubo un problema: {user_friendly_message}\n(Referencia: paso '{failed_step}')"
    state["step_completed"] = "error" # Mark the final state as error
    # Clear sensitive/large fields that are no longer needed after error
    state["principios_libro"] = None
    state["rutina_final"] = None


    logger.info(f"Generated user-friendly error message: {state['respuesta_usuario']}")
    logger.info("--- Exiting Handle Error Node ---")
    return state