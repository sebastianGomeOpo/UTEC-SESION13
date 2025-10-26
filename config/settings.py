"""
Configuración con carga desde JSONs
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuración de la aplicación"""
    
    # ========================================
    # ATRIBUTOS DE CLASE (ESTÁTICOS)
    # ========================================
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    HISTORIAL_PATH = DATA_DIR / "historial.json"
    USERS_DIR = DATA_DIR / "users"
    PROMPTS_DIR = BASE_DIR / "prompts"
    
    def __init__(self, user_id: str = None):
        # Cargar configuración de la app
        self._load_app_settings()
        
        # Cargar configuración del usuario
        self.user_id = user_id or self.app_settings["default_user_id"]
        self._load_user_settings()
        
        # API Keys
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    def _load_app_settings(self):
        """Carga settings.json"""
        settings_path = Config.DATA_DIR / "config" / "app_settings.json"
        
        if not settings_path.exists():
            self._create_default_app_settings(settings_path)
        
        with open(settings_path, "r", encoding="utf-8") as f:
            self.app_settings = json.load(f)
        
        # LLM config
        llm = self.app_settings["llm"]
        self.MODEL_NAME = llm["model_name"]
        self.TEMPERATURE = llm["temperature"]
        self.MAX_TOKENS = llm.get("max_tokens", 1000)
        
        # Memory config
        memory = self.app_settings["memory"]
        self.MEMORY_WINDOW_SIZE = memory["window_size"]
        self.SUMMARY_THRESHOLD = memory["summary_threshold"]
        self.RECURSION_LIMIT = memory["recursion_limit"]
        
        # ========================================
        # ACTUALIZAR ATRIBUTOS DE CLASE DESDE JSON
        # ========================================
        paths = self.app_settings.get("paths", {})
        if "historial" in paths:
            Config.HISTORIAL_PATH = Config.BASE_DIR / paths["historial"]
        if "users_dir" in paths:
            Config.USERS_DIR = Config.BASE_DIR / paths["users_dir"]
        if "prompts_dir" in paths:
            Config.PROMPTS_DIR = Config.BASE_DIR / paths["prompts_dir"]
        
        # Actualizar instancia también (para compatibilidad)
        self.HISTORIAL_PATH = Config.HISTORIAL_PATH
        self.USERS_DIR = Config.USERS_DIR
        self.PROMPTS_DIR = Config.PROMPTS_DIR
    
    def _load_user_settings(self):
        """Carga configuración del usuario"""
        user_file = Config.USERS_DIR / f"{self.user_id}.json"
        
        if not user_file.exists():
            print(f"⚠️  Usuario '{self.user_id}' no encontrado, usando default")
            user_file = Config.USERS_DIR / "default.json"
        
        with open(user_file, "r", encoding="utf-8") as f:
            self.user_data = json.load(f)
        
        # Extraer datos del usuario
        self.USER_ID = self.user_data["user_id"]
        self.USER_NAME = self.user_data["name"]
        self.USER_LEVEL = self.user_data["level"]
        self.WORKOUT_COUNT = self.user_data["workout_count"]
        self.OBJETIVO = self.user_data["objetivo"]
        self.FRECUENCIA = self.user_data["frecuencia_semanal"]
        self.EJERCICIOS_FAV = self.user_data["ejercicios_favoritos"]
        self.RESTRICCIONES = self.user_data["restricciones"]
    
    def _create_default_app_settings(self, path: Path):
        """Crea app_settings.json por defecto"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        default_settings = {
            "llm": {
                "model_name": "gpt-4o-mini",
                "temperature": 0.4,
                "max_tokens": 1000
            },
            "memory": {
                "window_size": 10,
                "summary_threshold": 20,
                "recursion_limit": 10
            },
            "paths": {
                "historial": "data/historial.json",
                "users_dir": "data/users",
                "prompts_dir": "prompts"
            },
            "default_user_id": "default"
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_settings, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Creado: {path}")
    
    def save_user_data(self):
        """Guarda cambios en el perfil del usuario"""
        user_file = Config.USERS_DIR / f"{self.user_id}.json"
        
        self.user_data.update({
            "name": self.USER_NAME,
            "level": self.USER_LEVEL,
            "workout_count": self.WORKOUT_COUNT,
            "objetivo": self.OBJETIVO,
            "frecuencia_semanal": self.FRECUENCIA,
            "ejercicios_favoritos": self.EJERCICIOS_FAV,
            "restricciones": self.RESTRICCIONES
        })
        
        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(self.user_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Usuario guardado: {user_file}")
    
    def validate(self):
        """Valida que las variables críticas existan"""
        if not self.OPENAI_API_KEY:
            raise ValueError("❌ OPENAI_API_KEY no está configurada en .env")
        
        print(f"✅ Config cargada: Usuario '{self.USER_NAME}' ({self.user_id})")

# Nuevo contenido a insertar (como atributos de clase estáticos)

    # RAG Configuration (Phase 0 Migration)
    BOOK_PATH = BASE_DIR / "data" / "books" / "entrenamiento.pdf"
    CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

    # LLM Models (Phase 0 Migration)
    LLM_MODEL_EXTRACT = "gpt-4o-mini"
    LLM_MODEL_ASSEMBLE = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.0