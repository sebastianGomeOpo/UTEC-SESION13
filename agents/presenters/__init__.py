"""
Módulo de presenters para agentes.
Exporta los presenters disponibles.
"""

from .base import Presenter
from .console import ConsolePresenter
from .verbose import VerbosePresenter

__all__ = [
    "Presenter",
    "ConsolePresenter", 
    "VerbosePresenter"
]