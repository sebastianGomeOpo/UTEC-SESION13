"""
[LEGACY] Cargador de prompts del sistema antiguo.

Este módulo se mantiene para backward compatibility con el agente EntrenadorAgent legacy.
El nuevo sistema RAG (Fase 2+) carga prompts directamente desde archivos .txt:
- prompts/rag_principle_extractor.txt
- prompts/routine_assembler.txt

NO modificar este archivo a menos que sea necesario para el agente legacy.
"""
from pathlib import Path
from datetime import datetime

class PromptLoader:
    """Carga y formatea prompts desde archivos .txt"""
    
    def __init__(self):
        self.prompts_dir = Path(__file__).parent.parent / "prompts"
    
    def _load_prompt(self, filename: str) -> str:
        """Carga un archivo de prompt"""
        file_path = self.prompts_dir / filename
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def get_adaptive_prompt(self, config) -> str:
        """Genera prompt adaptado desde config (que tiene datos del JSON)"""
        template = self._load_prompt("system_adaptative.txt")
        
        # Calcular motivación según workout_count
        if config.WORKOUT_COUNT == 0:
            motivacion = "¡Bienvenido a tu viaje fitness!"
        elif config.WORKOUT_COUNT < 10:
            motivacion = "¡Vas muy bien! Sigue así."
        else:
            motivacion = "¡Eres un veterano! Excelente consistencia."
        
        fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Formatear con datos del JSON
        return template.format(
            user_name=config.USER_NAME,
            nivel=config.USER_LEVEL,
            workout_count=config.WORKOUT_COUNT,
            fecha=fecha,
            motivacion=motivacion,
            objetivo=config.OBJETIVO,
            frecuencia=config.FRECUENCIA,
            ejercicios_fav=", ".join(config.EJERCICIOS_FAV),
            restricciones=", ".join(config.RESTRICCIONES) if config.RESTRICCIONES else "Ninguna"
        )
