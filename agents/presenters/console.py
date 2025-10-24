"""
Presenter simple para consola.
Imprime información básica sin modo verbose.
"""
from .base import Presenter
from typing import Dict, Any


class ConsolePresenter(Presenter):
    """Presenter para modo normal (sin verbose)"""
    
    def print_user_context(self, context: Dict[str, str]) -> None:
        """Imprime el contexto del usuario"""
        print("\n" + "="*50)
        print(f"👤 Usuario: {context.get('USER_NAME', 'N/A')}")
        print(f"🎯 Objetivo: {context.get('OBJETIVO', 'N/A')}")
        print(f"📊 Nivel: {context.get('USER_LEVEL', 'N/A')}")
        print("="*50 + "\n")
    
    def print_thinking(self, step: int, content: str) -> None:
        """En modo normal no se muestra pensamiento intermedio"""
        pass
    
    def print_tool_call(self, tool_name: str, tool_input: Dict[str, Any]) -> None:
        """En modo normal no se muestra llamada a herramientas"""
        pass
    
    def print_tool_result(self, tool_name: str, result: str) -> None:
        """En modo normal no se muestra resultado de herramientas"""
        pass
    
    def print_user_message(self, message: str) -> None:
        """En modo normal no se muestra mensaje del usuario"""
        pass
    
    def print_final_response(self, response: str) -> None:
        """Imprime solo la respuesta final"""
        print(response)
    
    def print_error(self, error: str) -> None:
        """Imprime un error"""
        print(f"❌ Error: {error}")