"""
Funciones auxiliares
"""
import json
import os

def ensure_data_dir():
    """Asegura que existe el directorio data/"""
    os.makedirs("data", exist_ok=True)

def init_historial(path="data/historial.json"):
    """Inicializa archivo de historial si no existe"""
    ensure_data_dir()
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f)