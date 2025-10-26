"""
Estado del grafo LangGraph para el flujo C-R-G.

Este TypedDict define la estructura de datos que se pasa entre nodos.
Cada nodo recibe un GraphState, lo modifica, y retorna un GraphState actualizado.

IMPORTANTE: Usar TypedDict en lugar de Pydantic para compatibilidad con LangGraph.
"""
from typing import TypedDict, Optional, Dict, Any, List


class GraphState(TypedDict, total=False):
    """
    Estado del grafo LangGraph.
    
    Flujo de datos:
    1. Inputs: user_id, request_type
    2. Paso C (Contexto): perfil_usuario, preferencias_logistica
    3. Paso R (Recuperar): principios_libro
    4. Paso G (Generar): rutina_final
    5. Control: step_completed, error, timestamp
    
    IMPORTANTE: 
    - total=False permite que campos sean opcionales
    - Cada nodo actualiza solo los campos que le corresponden
    - El estado se propaga automáticamente entre nodos
    """
    
    # ========================================
    # INPUTS INICIALES (SIEMPRE REQUERIDOS)
    # ========================================
    user_id: str
    """ID del usuario (ej: 'user_001', 'default')"""
    
    request_type: str
    """Tipo de petición: 'crear_rutina' | 'consultar_historial' | 'actualizar_perfil'"""
    
    # ========================================
    # PASO C: CONTEXTO (llenado por load_context)
    # ========================================
    perfil_usuario: Optional[Dict[str, Any]]
    """
    Perfil físico del usuario.
    Estructura: {level, objetivo, restricciones}
    Ejemplo: {'level': 'intermedio', 'objetivo': 'hipertrofia', 'restricciones': ['rodilla izquierda']}
    """
    
    preferencias_logistica: Optional[Dict[str, Any]]
    """
    Preferencias logísticas del usuario.
    Estructura: {equipamiento_disponible, dias_preferidos, duracion_sesion_min}
    Ejemplo: {'equipamiento_disponible': 'gimnasio completo', 'dias_preferidos': ['lunes', 'martes']}
    """
    
    # ========================================
    # PASO R: RECUPERAR PRINCIPIOS (llenado por extract_principles)
    # ========================================
    principios_libro: Optional[Dict[str, Any]]
    """
    Principios extraídos del libro mediante RAG.
    Estructura: PrincipiosExtraidos serializado como dict
    CRÍTICO: DEBE incluir 'citas_fuente' para verificabilidad
    """
    
    # ========================================
    # PASO G: GENERAR RUTINA (llenado por generate_routine)
    # ========================================
    rutina_final: Optional[Dict[str, Any]]
    """
    Rutina generada aplicando los principios.
    Estructura: RutinaActiva serializado como dict
    Incluye 'principios_aplicados' con citas para auditoría
    """
    
    # ========================================
    # CONTROL DE FLUJO
    # ========================================
    step_completed: str
    """
    Último paso completado exitosamente.
    Valores posibles:
    - '' (inicial)
    - 'context_loaded'
    - 'principles_extracted'
    - 'routine_generated'
    - 'saved'
    - 'error'
    """
    
    error: Optional[str]
    """
    Mensaje de error si algo falló.
    Si no es None, el flujo debe terminar o ir a nodo de error.
    """
    
    timestamp: str
    """Timestamp ISO de cuándo se inició el flujo (para logging)"""
    
    # ========================================
    # METADATA (OPCIONAL)
    # ========================================
    debug_info: Optional[Dict[str, Any]]
    """
    Información de debugging (solo en modo verbose).
    Ejemplo: {'rag_chunks': 5, 'retrieval_time_ms': 123}
    """


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def create_initial_state(user_id: str, request_type: str = "crear_rutina") -> GraphState:
    """
    Crea el estado inicial del grafo.
    
    Args:
        user_id: ID del usuario
        request_type: Tipo de petición (default: crear_rutina)
    
    Returns:
        GraphState inicial con solo inputs poblados
    """
    from datetime import datetime
    
    return GraphState(
        user_id=user_id,
        request_type=request_type,
        perfil_usuario=None,
        preferencias_logistica=None,
        principios_libro=None,
        rutina_final=None,
        step_completed="",
        error=None,
        timestamp=datetime.now().isoformat(),
        debug_info=None
    )


def is_state_valid(state: GraphState) -> tuple[bool, Optional[str]]:
    """
    Valida que el estado tenga campos requeridos.
    
    Args:
        state: Estado a validar
    
    Returns:
        (is_valid, error_message)
    """
    # Validar inputs
    if not state.get("user_id"):
        return False, "user_id es requerido"
    
    if not state.get("request_type"):
        return False, "request_type es requerido"
    
    # Si hay error, verificar que step_completed sea 'error'
    if state.get("error") and state.get("step_completed") != "error":
        return False, "Estado inconsistente: hay error pero step_completed no es 'error'"
    
    return True, None


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "GraphState",
    "create_initial_state",
    "is_state_valid"
]