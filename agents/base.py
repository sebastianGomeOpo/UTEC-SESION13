"""
Clase base abstracta para todos los agentes.
Define la interfaz común para los agentes.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class Agent(ABC):
    """
    Clase base abstracta para todos los agentes.
    Asegura que todos los agentes implementen los métodos principales.
    """
    
    @abstractmethod
    def run(self) -> None:
        """Inicia el loop principal del agente"""
        pass
    
    @abstractmethod
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Retorna el historial de conversación"""
        pass
    
    def set_verbose(self, verbose: bool) -> None:
        """Establece el modo verbose"""
        pass