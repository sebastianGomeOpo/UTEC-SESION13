"""
Clase base abstracta para presenters.
Un presenter es responsable de mostrar información del agente.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class Presenter(ABC):
    """Interfaz base para presenters"""
    
    @abstractmethod
    def print_user_context(self, context: Dict[str, str]) -> None:
        """Imprime el contexto del usuario"""
        pass
    
    @abstractmethod
    def print_thinking(self, step: int, content: str) -> None:
        """Imprime el pensamiento del agente"""
        pass
    
    @abstractmethod
    def print_tool_call(self, tool_name: str, tool_input: Dict[str, Any]) -> None:
        """Imprime una llamada a herramienta"""
        pass
    
    @abstractmethod
    def print_tool_result(self, tool_name: str, result: str) -> None:
        """Imprime el resultado de una herramienta"""
        pass
    
    @abstractmethod
    def print_user_message(self, message: str) -> None:
        """Imprime un mensaje del usuario"""
        pass
    
    @abstractmethod
    def print_final_response(self, response: str) -> None:
        """Imprime la respuesta final del agente"""
        pass
    
    @abstractmethod
    def print_error(self, error: str) -> None:
        """Imprime un error"""
        pass