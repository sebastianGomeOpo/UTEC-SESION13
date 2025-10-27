"""
Módulo de agentes.
"""

from .base import Agent
# Se eliminó la importación incorrecta de EntrenadorAgent desde entrenador.py
# Si tienes una clase EntrenadorAgent en otro archivo, impórtala desde allí.
# Si no, esta línea debe permanecer comentada o eliminada.
# from .entrenador import EntrenadorAgent

__all__ = ["Agent"] # Ajusta si tienes otras clases de agentes para exportar
