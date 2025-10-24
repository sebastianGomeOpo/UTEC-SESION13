"""
Base compartida para todas las herramientas.
Define la estructura de usuario y el gestor de historial.
"""
import json
from pathlib import Path
from typing import Any, Dict, List
from config.settings import Config


class UserContext:
    """Contexto del usuario actual - inyectable en herramientas"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.historial_dir = Config.DATA_DIR / "historial"
        self.historial_dir.mkdir(parents=True, exist_ok=True)
        self.user_historial_file = self.historial_dir / f"{user_id}.json"
    
    def get_historial(self) -> List[Dict[str, Any]]:
        """Obtiene el historial del usuario"""
        try:
            if self.user_historial_file.exists():
                with open(self.user_historial_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return []
    
    def save_historial(self, historial: List[Dict[str, Any]]) -> None:
        """Guarda el historial del usuario"""
        with open(self.user_historial_file, "w", encoding="utf-8") as f:
            json.dump(historial, f, indent=2, ensure_ascii=False)
    
    def add_to_historial(self, entrada: Dict[str, Any]) -> None:
        """Agrega una entrada al historial"""
        historial = self.get_historial()
        historial.append(entrada)
        self.save_historial(historial)


# Almacenamiento global del contexto actual (se actualiza en factory)
_current_user_context: UserContext = None


def get_user_context() -> UserContext:
    """Obtiene el contexto del usuario actual"""
    if _current_user_context is None:
        raise RuntimeError("No hay contexto de usuario. Usa TrainingToolsFactory.")
    return _current_user_context


def set_user_context(user_context: UserContext) -> None:
    """Establece el contexto del usuario"""
    global _current_user_context
    _current_user_context = user_context