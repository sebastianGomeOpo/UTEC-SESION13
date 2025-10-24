"""
Estrategias de memoria
"""

class MemoryStrategy:
    """Clase base para estrategias de memoria"""
    
    @staticmethod
    def apply_window(messages, window_size):
        """Buffer Window: mantiene últimos N mensajes"""
        if len(messages) <= window_size:
            return messages
        return messages[-window_size:]
    
    @staticmethod
    def create_summary(messages, llm):
        """Crea resumen de conversación"""
        conversation_text = ""
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                role = "Usuario" if msg.type == "human" else "Entrenador"
                conversation_text += f"{role}: {msg.content}\n"
        
        summary_prompt = f"""Resume esta conversación fitness en 3 puntos:
{conversation_text}

Formato:
- Ejercicios registrados: ...
- Progreso: ...
- Objetivos: ...
"""
        summary = llm.invoke(summary_prompt)
        return summary.content