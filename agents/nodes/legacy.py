# FILENAME: agents/nodes/legacy.py
# -----------------------------------------------------------------------------
# Este archivo contiene nodos para compatibilidad con el agente "legacy"
# (Fase 1), permitiendo al grafo C-R-G invocar las herramientas
# antiguas de registro y consulta.
# -----------------------------------------------------------------------------

import re
import json
from typing import Dict, Any

# LangChain and Project Imports
from agents.graph_state import GraphState
from utils.logger import setup_logger

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

def _parse_legacy_exercise(user_message: str) -> Dict[str, Any] | None:
    """
    Parsea un string de formato libre a datos estructurados de ejercicio.
    Ej: "registra 5x5 de sentadilla con 100kg"
    """
    # --- CORRECCIÓN REGEX ---
    # Patrón más robusto para el nombre del ejercicio (acepta espacios y es greedy)
    pattern = (
        r"(registra|anota)\s+"
        r"(\d+)\s*x\s*(\d+)\s+"      # 5x5
        r"de\s+([\w ]+)\s+"          # de sentadilla (acepta espacios, greedy)
        r"con\s+([\d\.]+)\s*kg"      # con 100kg (acepta decimales)
    )

    match = re.search(pattern, user_message.lower().strip()) # Añadido strip() por si acaso

    if match:
        try:
            return {
                "ejercicio": match.group(4).strip(),
                "series": int(match.group(2)),
                "repeticiones": int(match.group(3)),
                "peso_kg": float(match.group(5)),
            }
        except (ValueError, IndexError):
            logger.warning(f"Error de parsing en regex match: {match.groups()} para mensaje '{user_message}'")
            return None

    # --- CORRECCIÓN LOG ---
    # Incluir el user_message en el log de advertencia
    logger.warning(f"Mensaje legacy no coincide con patrón regex: '{user_message}'")
    return None

def call_legacy_register(state: GraphState) -> GraphState:
    """
    Nodo: call_legacy_register

    Wrapper para la herramienta legacy 'registrar_ejercicio'.
    Configura el UserContext requerido por la herramienta.

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
        # 1. Parsear el mensaje
        parsed_data = _parse_legacy_exercise(user_message)

        if not parsed_data:
            # --- CORRECCIÓN LOG --- (Ya corregido en _parse_legacy_exercise)
            # logger.warning(f"Formato no reconocido para registro: {user_message}") # Redundante
            state["error"] = "Formato no reconocido. Usa: 'Registra 5x5 de sentadilla con 100kg'"
            state["step_completed"] = "call_legacy_register_error"
            return state

        # 2. Configurar el contexto legacy
        logger.debug(f"Configurando UserContext legacy para: {user_id}")
        context = UserContext(user_id=user_id)
        set_user_context(context)

        # 3. Preparar datos y llamar a la herramienta usando .invoke()
        ejercicio_data = EjercicioEstructurado(**parsed_data)

        logger.info(f"Invocando herramienta legacy registrar_ejercicio para {user_id}")
        # --- CORRECCIÓN TOOL CALL ---
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