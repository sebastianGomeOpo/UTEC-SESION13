"""
Herramienta: Cálculo de 1RM
"""
from langchain_core.tools import tool

@tool
def calcular_1rm(peso: float, repeticiones: int) -> str:
    """
    Calcula el One-Rep Max (1RM) estimado usando la fórmula de Brzycki.
    """
    if repeticiones < 1 or repeticiones > 12:
        return "⚠️ El cálculo es fiable solo para 1-12 repeticiones."
    
    # Fórmula de Brzycki
    rm_estimado = peso / (1.0278 - (0.0278 * repeticiones))
    
    return f"💪 Tu 1RM estimado es: {rm_estimado:.2f} kg"