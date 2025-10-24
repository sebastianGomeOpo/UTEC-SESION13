"""
Herramienta: Registro de Ejercicios
Ahora personalizada por usuario mediante _base.UserContext
"""
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from ._base import get_user_context


class EjercicioEstructurado(BaseModel):
    ejercicio: str = Field(description="Nombre del ejercicio")
    series: int = Field(description="Número de series")
    repeticiones: int = Field(description="Repeticiones por serie")
    peso_kg: float = Field(description="Peso en kg")


@tool
def registrar_ejercicio(datos_ejercicio: EjercicioEstructurado) -> str:
    """
    Registra un ejercicio en el historial personal del usuario.
    
    Esta herramienta está contextualizada por usuario.
    El historial se guarda en: data/historial/{user_id}.json
    """
    try:
        user_context = get_user_context()
        
        # Crear entrada
        nueva_entrada = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_context.user_id,
            "ejercicio": datos_ejercicio.ejercicio,
            "series": datos_ejercicio.series,
            "repeticiones": datos_ejercicio.repeticiones,
            "peso_kg": datos_ejercicio.peso_kg
        }
        
        # Guardar
        user_context.add_to_historial(nueva_entrada)
        
        return f"✅ Registrado: {datos_ejercicio.ejercicio} - {datos_ejercicio.series}x{datos_ejercicio.repeticiones} @ {datos_ejercicio.peso_kg}kg"
    
    except Exception as e:
        return f"❌ Error al registrar: {e}"