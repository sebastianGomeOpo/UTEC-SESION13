"""
Gestor de memoria del agente
"""
from .strategies import MemoryStrategy

class MemoryManager:
    """Gestiona la memoria del agente"""
    
    def __init__(self, window_size=10, summary_threshold=20, llm=None):
        self.window_size = window_size
        self.summary_threshold = summary_threshold
        self.llm = llm
        
        self.messages = []
        self.summary = None
    
    def update(self, new_messages):
        """Actualiza la memoria con nuevos mensajes"""
        self.messages = new_messages
        
        # Aplicar summary si es necesario
        if len(self.messages) > self.summary_threshold and self.llm:
            self._apply_summary()
    
    def get_windowed_history(self):
        """Retorna ventana de memoria"""
        return MemoryStrategy.apply_window(self.messages, self.window_size)
    
    def _apply_summary(self):
        """Aplica resumen a la memoria"""
        print("\n🧠 [Resumiendo conversación...]")
        
        self.summary = MemoryStrategy.create_summary(self.messages, self.llm)
        
        # Mantener resumen + últimos mensajes
        summary_msg = ("system", f"[RESUMEN]\n{self.summary}")
        recent = self.messages[-self.window_size:]
        self.messages = [summary_msg] + recent
        
        print("📝 [Resumen guardado]\n")
    
    def get_stats(self):
        """Estadísticas de memoria"""
        return {
            "total": len(self.messages),
            "window": len(self.get_windowed_history()),
            "has_summary": self.summary is not None
        }