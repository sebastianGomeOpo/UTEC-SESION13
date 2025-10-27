from typing import Dict, Any, Optional # Import Optional

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
    "Rutina generada está vacía": "La rutina que se intentó generar resultó vacía.", # Añadido para cubrir Falla 6
    
    # ─────────────────────────────────────────────────────────────────────
    # GRUPO 3: ERRORES DE PERFIL/CONTEXTO
    # ─────────────────────────────────────────────────────────────────────
    "no encontrado": "Tu perfil de usuario no se encontró en el sistema.",  # ✅ Genérico para capturar variantes
    "Archivo corrupto": "Hubo un problema al leer los datos de tu perfil. Podría estar corrupto.",
    "Perfil incompleto": "A tu perfil le faltan datos esenciales (como nivel u objetivo).",
    "Perfil de usuario no disponible": "No se pudo cargar tu perfil para esta acción.", # Añadido
    
    # ─────────────────────────────────────────────────────────────────────
    # GRUPO 4: ERRORES RAG (EXTRACCIÓN DE PRINCIPIOS)
    # ─────────────────────────────────────────────────────────────────────
    "Alucinación detectada": "No pude verificar la información extraída del libro (faltan citas). No puedo proceder de forma segura.",
    "RAG retornó resultado nulo": "No pude obtener los principios de entrenamiento del libro en este momento.",
    "Error de API o RAG": "Hubo un problema comunicándome con el sistema de extracción de información.",
    "Principios del libro retornaron vacíos": "No se encontraron principios aplicables en el libro para tu perfil.", # Añadido
    
    # ─────────────────────────────────────────────────────────────────────
    # GRUPO 5: ERRORES DE GENERACIÓN DE RUTINA
    # ─────────────────────────────────────────────────────────────────────
    "Archivo de prompt no encontrado": "Error interno: falta una plantilla necesaria para generar la rutina.",
    "LLM no retornó JSON válido": "La IA generó una respuesta en un formato inesperado y no se pudo procesar.",
    "Validación fallida": "La rutina generada no parece cumplir con los principios o tus preferencias.",  # ✅ Genérico al final
    "Error generando rutina": "Hubo un problema al intentar generar tu rutina.", # Añadido genérico

    # --- CORRECCIÓN --- Añadido para cubrir el caso de 'unknown request'
    "Tipo de request desconocido": "No entendí qué acción deseas realizar.",
    "structuredtool' object is not callable": "Error interno al intentar usar una herramienta legacy.", # Para Falla 4
    "Formato no reconocido": "El formato del comando no es el esperado.", # Para Falla 3
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
    # --- CORRECCIÓN --- Asegurar que technical_error_msg sea siempre string
    technical_error_msg: Optional[str] = state.get("error")
    if technical_error_msg is None:
        logger.warning("handle_error llamado sin mensaje de error en el estado. Usando default.")
        technical_error_msg = "Error desconocido reportado sin mensaje."
        # Opcional: Podrías querer setear state['error'] aquí
        # state['error'] = technical_error_msg

    failed_step = state.get("step_completed", "paso desconocido")  # Step where error occurred

    # ════════════════════════════════════════════════════════════════════
    # 1. LOGGING TÉCNICO COMPLETO
    # ════════════════════════════════════════════════════════════════════
    # Ahora technical_error_msg es garantizado string (o el default)
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

    technical_error_lower = technical_error_msg.lower() # Ahora es seguro llamar a .lower()
    for key, message in ERROR_MESSAGE_MAP.items():
        if key.lower() in technical_error_lower:
            user_friendly_message = message
            logger.debug(f"Mapped error key '{key}' to message: '{message}'")
            break  # ✅ Usar el primer match (por eso el orden importa)
    else: # Si el bucle termina sin break
         logger.debug(f"No specific map found for '{technical_error_msg}'. Using default.")


    # ════════════════════════════════════════════════════════════════════
    # 3. FORMATEAR RESPUESTA FINAL PARA EL USUARIO
    # ════════════════════════════════════════════════════════════════════
    # --- CORRECCIÓN --- Asegurar que respuesta_usuario siempre se setea
    state["respuesta_usuario"] = (
        f"❌ Lo siento, hubo un problema: {user_friendly_message}\n"
        f"(Referencia: paso '{failed_step}')"
    )
    state["step_completed"] = "error"  # Mark the final state as error

    # Limpiar campos sensibles/grandes que ya no se necesitan
    # Es seguro llamar a .pop() con un default None si la clave no existe
    state.pop("principios_libro", None)
    state.pop("rutina_final", None)

    logger.info(f"Generated user-friendly error message: {state['respuesta_usuario']}")
    logger.info("--- Exiting Handle Error Node ---")

    return state