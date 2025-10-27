# FILENAME: agents/entrenador.py
# -----------------------------------------------------------------------------
# Fase 5: Compilador del Grafo
# Este archivo define el StateGraph que orquesta todo el flujo (C-R-G + Legacy).
# -----------------------------------------------------------------------------

from langgraph.graph import StateGraph, END
from langgraph.graph import CompiledGraph

# Imports del proyecto
from agents.graph_state import GraphState, PrincipiosExtraidos
from utils.logger import setup_logger

# Importar TODOS los nodos que usará el grafo
from agents.nodes import (
    load_context,
    extract_principles,
    generate_routine,
    save_routine,
    handle_error,
)
from agents.nodes.legacy import (
    call_legacy_register,
    call_legacy_query
)

logger = setup_logger(__name__)


# -----------------------------------------------------------------------------
# FUNCIONES DE ROUTING (DECISIÓN)
# -----------------------------------------------------------------------------

def route_after_load(state: GraphState) -> str:
    """
    Router 1: Decide qué flujo tomar después de cargar el contexto.
    (C-R-G, Legacy Register, Legacy Query, o Error).
    """
    if state.get("error"):
        logger.warning(f"Error detectado en load_context: {state['error']}")
        return "handle_error"

    request_type = state.get("request_type", "unknown")
    logger.info(f"Routing after load, request_type: {request_type}")
    
    if request_type == "crear_rutina":
        return "extract_principles"
    elif request_type == "registrar_ejercicio":
        return "call_legacy_register"
    elif request_type == "consultar_historial":
        return "call_legacy_query"
    else:
        logger.error(f"Tipo de request desconocido: {request_type}")
        state["error"] = f"Tipo de request desconocido: {request_type}"
        return "handle_error"

def route_after_extract(state: GraphState) -> str:
    """
    Router 2: Valida los principios extraídos antes de generar.
    (Previene alucinaciones y errores de RAG).
    """
    if state.get("error"):
        logger.warning(f"Error detectado en extract_principles: {state['error']}")
        return "handle_error"
        
    principios: PrincipiosExtraidos | None = state.get("principios_libro")

    if not principios:
        logger.error("RAG retornó principios vacíos o nulos.")
        state["error"] = "Principios del libro retornaron vacíos."
        return "handle_error"

    # CRÍTICO: Validación de alucinación (sin citas)
    if not principios.citas_fuente:
        logger.error("¡ALUCINACIÓN DETECTADA! Principios extraídos sin citas.")
        state["error"] = "Alucinación detectada: principios extraídos sin citas de fuente."
        # Opcional: guardar debug info
        state["debug_info"] = state.get("debug_info", {})
        state["debug_info"]["principles_without_citations"] = principios.model_dump()
        return "handle_error"
    
    logger.info("Principios validados (con citas). Procediendo a generar rutina.")
    return "generate_routine"

def route_after_generate(state: GraphState) -> str:
    """
    Router 3: Valida la rutina generada antes de guardarla.
    """
    if state.get("error"):
        logger.warning(f"Error detectado en generate_routine: {state['error']}")
        return "handle_error"

    if not state.get("rutina_final"):
        logger.error("El nodo de generación no produjo una rutina_final.")
        state["error"] = "Rutina generada está vacía."
        return "handle_error"
        
    logger.info("Rutina generada y validada. Procediendo a guardar.")
    return "save_routine"

# -----------------------------------------------------------------------------
# CONSTRUCTOR DEL GRAFO
# -----------------------------------------------------------------------------

def build_graph() -> CompiledGraph:
    """
    Construye y compila el StateGraph C-R-G + Legacy.

    Returns:
        CompiledGraph: El grafo compilado y listo para invocar.
        
    Raises:
        Exception: Si el grafo no puede compilarse.
    """
    logger.info("Construyendo el grafo de LangGraph...")
    
    # 1. Crear instancia del StateGraph
    workflow = StateGraph(GraphState)

    # 2. Agregar TODOS los nodos (7 en total)
    logger.debug("Agregando nodos al grafo...")
    workflow.add_node("load_context", load_context)
    workflow.add_node("extract_principles", extract_principles)
    workflow.add_node("generate_routine", generate_routine)
    workflow.add_node("save_routine", save_routine)
    workflow.add_node("handle_error", handle_error)
    workflow.add_node("call_legacy_register", call_legacy_register)
    workflow.add_node("call_legacy_query", call_legacy_query)

    # 3. Definir entry point
    workflow.set_entry_point("load_context")
    logger.debug("Entry point seteado a 'load_context'")

    # 4. Aristas condicionales (Routing)
    
    # Router 1: Después de cargar contexto
    workflow.add_conditional_edges(
        "load_context",
        route_after_load,
        {
            "extract_principles": "extract_principles",
            "call_legacy_register": "call_legacy_register",
            "call_legacy_query": "call_legacy_query",
            "handle_error": "handle_error",
        }
    )
    logger.debug("Aristas condicionales agregadas para 'load_context'")

    # Router 2: Después de extraer principios (Validación RAG)
    workflow.add_conditional_edges(
        "extract_principles",
        route_after_extract,
        {
            "generate_routine": "generate_routine",
            "handle_error": "handle_error",
        }
    )
    logger.debug("Aristas condicionales agregadas para 'extract_principles'")

    # Router 3: Después de generar rutina (Validación Generación)
    workflow.add_conditional_edges(
        "generate_routine",
        route_after_generate,
        {
            "save_routine": "save_routine",
            "handle_error": "handle_error",
        }
    )
    logger.debug("Aristas condicionales agregadas para 'generate_routine'")

    # 5. Aristas normales (Terminales)
    # Todos los flujos exitosos o de error deben terminar.
    workflow.add_edge("save_routine", END)
    workflow.add_edge("handle_error", END)
    workflow.add_edge("call_legacy_register", END)
    workflow.add_edge("call_legacy_query", END)
    logger.debug("Aristas terminales (END) agregadas.")

    # 6. Compilar el grafo
    try:
        graph = workflow.compile()
        logger.info("✅ Grafo compilado exitosamente.")
        return graph
    except Exception as e:
        logger.exception(f"❌ Error CRÍTICO compilando el grafo: {e}")
        raise

# -----------------------------------------------------------------------------
# EXPORTACIÓN
# -----------------------------------------------------------------------------

# Instancia global, compilada y lista para ser importada por main.py
graph = build_graph()