"""
Presenter inteligente para modo verbose.
Solo imprime eventos NUEVOS en cada step (no repite).
"""
from .base import Presenter
from typing import Dict, Any, Set, Tuple


class VerbosePresenter(Presenter):
    """
    Presenter inteligente que solo imprime cambios/eventos nuevos.
    """
    
    def __init__(self):
        self.step_counter = 0
        self.printed_events: Set[Tuple[str, str]] = set()  # Evita duplicados
    
    def _create_event_id(self, event_type: str, content: str) -> Tuple[str, str]:
        """Crea un ID único para un evento"""
        return (event_type, content[:50])  # Primeros 50 chars como identificador
    
    def print_user_context(self, context: Dict[str, str]) -> None:
        """Imprime el contexto del usuario"""
        print("\n" + "="*60)
        print(f"👤 Usuario: {context.get('USER_NAME', 'N/A')}")
        print(f"🎯 Objetivo: {context.get('OBJETIVO', 'N/A')}")
        print(f"📊 Nivel: {context.get('USER_LEVEL', 'N/A')}")
        print(f"🔍 Modo: VERBOSE (solo eventos nuevos)")
        print("="*60 + "\n")
    
    def print_thinking(self, step: int, content: str) -> None:
        """
        Imprime pensamiento del agente.
        
        INTELIGENTE: Solo imprime si es pensamiento nuevo
        """
        event_id = self._create_event_id("thinking", content)
        
        if event_id not in self.printed_events:
            self.printed_events.add(event_id)
            print(f"\n📍 [Step {step}]")
            print(f"  💭 Pensamiento: {content}")
    
    def print_tool_call(self, tool_name: str, tool_input: Dict[str, Any]) -> None:
        """
        Imprime llamada a herramienta.
        
        INTELIGENTE: Solo imprime si es una llamada nueva
        """
        input_str = str(tool_input)
        event_id = self._create_event_id("tool_call", tool_name + input_str)
        
        if event_id not in self.printed_events:
            self.printed_events.add(event_id)
            print(f"  🔧 Llamando herramienta: {tool_name}")
            print(f"     Parámetros: {tool_input}")
    
    def print_tool_result(self, tool_name: str, result: str) -> None:
        """
        Imprime resultado de herramienta.
        
        INTELIGENTE: Solo imprime si es resultado nuevo
        """
        event_id = self._create_event_id("tool_result", tool_name + result[:30])
        
        if event_id not in self.printed_events:
            self.printed_events.add(event_id)
            print(f"  ✅ Resultado [{tool_name}]:")
            # Limitar líneas largas
            lines = result.split('\n')
            for line in lines[:5]:  # Máximo 5 líneas
                print(f"     {line}")
            if len(lines) > 5:
                print(f"     ... ({len(lines) - 5} líneas más)")
    
    def print_user_message(self, message: str) -> None:
        """Imprime mensaje del usuario (solo una vez)"""
        event_id = self._create_event_id("user_msg", message)
        
        if event_id not in self.printed_events:
            self.printed_events.add(event_id)
            print(f"  👤 User: {message}")
    
    def print_final_response(self, response: str) -> None:
        """Imprime respuesta final con separador visual"""
        print(f"\n\n✨ Respuesta Final:")
        print(f"{response}")
    
    def print_error(self, error: str) -> None:
        """Imprime error con destacado"""
        print(f"\n  ⚠️  Error: {error}")
    
    def reset(self) -> None:
        """Reseta el contador de eventos (para nueva conversación)"""
        self.printed_events.clear()
        self.step_counter = 0