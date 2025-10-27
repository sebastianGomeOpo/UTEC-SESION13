from typing import Dict, Any

# Project Imports
from agents.graph_state import GraphState
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ════════════════════════════════════════════════════════════════════════
# MAPEO DE ERRORES TÉCNICOS A MENSAJES USER-FRIENDLY
# ════════════════════════════════════════════════════════════════════════
# CRÍTICO: El orden importa - keys específicas PRIMERO, genéricas DESPUÉS
# Python evalúa este dict en orden de inserción (Python 3.7+)
# ════════════════════════════════════════════════════════════════════════

ERROR_MESSAGE_MAP = {
    # ─────────────────────────────────────────────────────────────────────
    # GRUPO 1: ERRORES DE VALIDACIÓN (MÁS ESPECÍFICOS PRIMERO)
    # ─────────────────────────────────────────────────────────────────────
    "RIR inconsistente": "La rutina generada no respeta el RIR recomendado por el libro.",
    "Reps inconsistentes": "La rutina generada no respeta las repeticiones recomendadas por el libro.",
    "Rutina excede duración": "La rutina generada es demasiado larga para el tiempo que especificaste.",
    "Rutina generada omite ECIs": "La rutina generada no incluyó ejercicios compensatorios necesarios.",
    
    # ─────────────────────────────────────────────────────────────────────
    # GRUPO 2: ERRORES DE GUARDADO (ESPECÍFICOS ANTES DE GENÉRICOS)
    # ─────────────────────────────────────────────────────────────────────
    "Permission denied": "Error del sistema al intentar guardar (problema de permisos).",
    "Permisos insuficientes": "Error del sistema al intentar guardar (problema de permisos).",
    "Espacio en disco insuficiente": "Error del sistema al intentar guardar (espacio en disco lleno).",
    "Error de archivo guardando": "Error del sistema al intentar guardar la rutina en tu perfil.",
    "Rutina final vacía": "No se generó una rutina válida para guardar.",
    "Verificación de persistencia falló": "Error interno al verificar que la rutina se guardó correctamente.",
    
    # ─────────────────────────────────────────────────────────────────────
    # GRUPO 3: ERRORES DE PERFIL/CONTEXTO
    # ─────────────────────────────────────────────────────────────────────
    "no encontrado": "Tu perfil de usuario no se encontró en el sistema.",  # ✅ Genérico para capturar variantes
    "Archivo corrupto": "Hubo un problema al leer los datos de tu perfil. Podría estar corrupto.",
    "Perfil incompleto": "A tu perfil le faltan datos esenciales (como nivel u objetivo).",
    
    # ─────────────────────────────────────────────────────────────────────
    # GRUPO 4: ERRORES RAG (EXTRACCIÓN DE PRINCIPIOS)
    # ─────────────────────────────────────────────────────────────────────
    "Alucinación detectada": "No pude verificar la información extraída del libro (faltan citas). No puedo proceder de forma segura.",
    "RAG retornó resultado nulo": "No pude obtener los principios de entrenamiento del libro en este momento.",
    "Error de API o RAG": "Hubo un problema comunicándome con el sistema de extracción de información.",
    
    # ─────────────────────────────────────────────────────────────────────
    # GRUPO 5: ERRORES DE GENERACIÓN DE RUTINA
    # ─────────────────────────────────────────────────────────────────────
    "Archivo de prompt no encontrado": "Error interno: falta una plantilla necesaria para generar la rutina.",
    "LLM no retornó JSON válido": "La IA generó una respuesta en un formato inesperado y no se pudo procesar.",
    "Validación fallida": "La rutina generada no parece cumplir con los principios o tus preferencias.",  # ✅ Genérico al final
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
    failed_step = state.get("step_completed", "paso desconocido")  # Step where error occurred

    # ════════════════════════════════════════════════════════════════════
    # 1. LOGGING TÉCNICO COMPLETO
    # ════════════════════════════════════════════════════════════════════
    logger.error(f"Error captured in step '{failed_step}': {technical_error_msg}")
    
    # Log state context (evitando objetos grandes)
    log_state = {
        k: v for k, v in state.items() 
        if k not in ['principios_libro', 'rutina_final', 'perfil_usuario']
    }
    logger.debug(f"Full state context at error (partial): {log_state}")

    # ════════════════════════════════════════════════════════════════════
    # 2. MAPEO A MENSAJE USER-FRIENDLY
    # ════════════════════════════════════════════════════════════════════
    user_friendly_message = DEFAULT_ERROR_MESSAGE
    
    for key, message in ERROR_MESSAGE_MAP.items():
        if key.lower() in technical_error_msg.lower():
            user_friendly_message = message
            break  # ✅ Usar el primer match (por eso el orden importa)

    # ════════════════════════════════════════════════════════════════════
    # 3. FORMATEAR RESPUESTA FINAL PARA EL USUARIO
    # ════════════════════════════════════════════════════════════════════
    state["respuesta_usuario"] = (
        f"❌ Lo siento, hubo un problema: {user_friendly_message}\n"
        f"(Referencia: paso '{failed_step}')"
    )
    state["step_completed"] = "error"  # Mark the final state as error
    
    # Limpiar campos sensibles/grandes que ya no se necesitan
    state["principios_libro"] = None
    state["rutina_final"] = None

    logger.info(f"Generated user-friendly error message: {state['respuesta_usuario']}")
    logger.info("--- Exiting Handle Error Node ---")
    
    return state