"""
Herramienta: Consulta de Historial
Ahora personalizada por usuario mediante _base.UserContext
"""
from langchain_core.tools import tool
from ._base import get_user_context

@tool
def consultar_historial(ultimos_n: int = 7) -> str:
    """
    Consulta los últimos N entrenamientos del usuario.

    Esta herramienta está contextualizada por usuario.
    Solo devuelve entrenamientos del usuario actual.
    """
    try:
        user_context = get_user_context()
        historial = user_context.get_historial()

        if not historial:
            return f"📭 No hay entrenamientos registrados aún."

        # Obtener últimos N
        recientes = historial[-ultimos_n:]

        # Formatear respuesta
        resultado = f"📋 Últimos {len(recientes)} entrenamientos:\n\n"
        for i, entrada in enumerate(recientes, 1):
            resultado += f"{i}. {entrada.get('ejercicio', 'N/A')}\n"
            resultado += f"   └─ {entrada.get('series', 'N/A')}x{entrada.get('repeticiones', 'N/A')} @ {entrada.get('peso_kg', 'N/A')}kg\n"
            resultado += f"   📅 {entrada.get('timestamp', 'N/A')[:10]}\n"

        return resultado
    except Exception as e:
        return f"❌ Error al consultar historial: {e}"

@tool
def estadisticas_usuario() -> str:
    """
    Obtiene estadísticas del entrenamiento del usuario.
    """
    try:
        user_context = get_user_context()
        historial = user_context.get_historial()

        if not historial:
            return "📊 No hay datos de entrenamientos aún."

        # Calcular estadísticas
        total = len(historial)
        ejercicios_unicos = len(set(e.get('ejercicio', '') for e in historial))
        peso_maximo = max(float(e.get('peso_kg', 0)) for e in historial)

        resultado = f"📊 Estadísticas:\n"
        resultado += f"• Entrenamientos totales: {total}\n"
        resultado += f"• Ejercicios diferentes: {ejercicios_unicos}\n"
        resultado += f"• Peso máximo: {peso_maximo}kg\n"

        return resultado
    except Exception as e:
        return f"❌ Error al obtener estadísticas: {e}"