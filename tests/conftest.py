# Standard Library Imports
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Generator

# Third-party Imports
import pytest
from unittest.mock import MagicMock

# Project Imports (Ensure these paths are correct relative to your project root)
# It assumes tests are run from the project root directory
try:
    from agents.graph_state import GraphState
    from rag.models import PrincipiosExtraidos, RutinaActiva, Sesion, Ejercicio, ECI
    from rag.chunking_strategy import get_semantic_text_splitter
    from rag.vectorstore_manager import VectorStoreManager
    from rag.principle_extractor import PrincipleExtractor
    from config.settings import Config
except ImportError as e:
    # Add project root to sys.path if running tests from `tests` directory directly
    import sys
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    try:
        from agents.graph_state import GraphState
        from rag.models import PrincipiosExtraidos, RutinaActiva, Sesion, Ejercicio, ECI
        from rag.chunking_strategy import get_semantic_text_splitter
        from rag.vectorstore_manager import VectorStoreManager
        from rag.principle_extractor import PrincipleExtractor
        from config.settings import Config
    except ImportError:
        print(f"ERROR: Could not import project modules. Ensure tests are run from project root or paths are correct. {e}")
        pytest.exit(f"Failed to import project modules: {e}", 1)


# --- Helper Functions ---

def create_valid_principles(
    rir: str = "1-2",
    reps: str = "6-8",
    citas: list[str] | None = None,
    ecis: list[ECI] | None = None
) -> PrincipiosExtraidos:
    """Factory for creating valid PrincipiosExtraidos objects for tests."""
    if citas is None:
        citas = ["Página 138, Tabla 20"]
    if ecis is None:
        ecis = []
    return PrincipiosExtraidos(
        intensidad_RIR=rir,
        rango_repeticiones=reps,
        descanso_series_s=90,
        cadencia_tempo="3:0:1:1",
        frecuencia_semanal="3-5 días",
        ECI_recomendados=ecis,
        citas_fuente=citas,
    )

def create_valid_routine(principles: PrincipiosExtraidos | Dict[str, Any], user_id="test_user") -> RutinaActiva:
    """Factory for creating a valid RutinaActiva object matching principles."""
    if isinstance(principles, PrincipiosExtraidos):
        principles_dict = principles.model_dump()
        rir = principles.intensidad_RIR
        tempo = principles.cadencia_tempo
        reps = principles.rango_repeticiones
        descanso = principles.descanso_series_s
    else: # Assume dict
        principles_dict = principles
        rir = principles_dict.get("intensidad_RIR", "1-2")
        tempo = principles_dict.get("cadencia_tempo", "3:0:1:1")
        reps = principles_dict.get("rango_repeticiones", "6-8")
        descanso = principles_dict.get("descanso_series_s", 90)

    # Add ECIs from principles if they exist
    ecis_from_principles = principles_dict.get("ECI_recomendados", [])
    eci_ejercicios = []
    for eci_data in ecis_from_principles:
        eci_ejercicios.append(
            Ejercicio(
                nombre=eci_data.get("nombre_ejercicio", "ECI Desconocido"),
                tipo="ECI",
                sets=eci_data.get("sets", 3),
                reps=eci_data.get("reps", "15"),
                RIR="3",
                tempo="2:1:2:1",
                descanso_s=60,
                notas=f"Compensatorio: {eci_data.get('motivo', 'N/A')}"
            )
        )

    return RutinaActiva(
        nombre="Rutina Test Válida",
        fecha_creacion=datetime.now().isoformat(),
        validez_semanas=4,
        sesiones=[
            Sesion(
                dia_semana="lunes",
                enfoque_muscular="pierna",
                duracion_estimada_min=55,
                ejercicios=[
                    Ejercicio(
                        nombre="Sentadilla",
                        tipo="principal",
                        sets=4,
                        reps=reps,
                        RIR=rir,
                        tempo=tempo,
                        descanso_s=descanso,
                        notas="Técnica estricta",
                    ),
                    *eci_ejercicios,
                    Ejercicio(
                        nombre="Curl femoral",
                        tipo="accesorio",
                        sets=3,
                        reps="12-15",
                        RIR="2-3",
                        tempo="2:0:2:0",
                        descanso_s=60,
                        notas=None,
                    )
                ],
            )
        ],
        principios_aplicados=principles_dict if isinstance(principles, dict) else principles
    )

# --- Fixtures ---

@pytest.fixture
def empty_graph_state() -> GraphState:
    """Provides an empty/initial GraphState dictionary."""
    return GraphState(
        user_id="test_user",
        request_type="crear_rutina",
        perfil_usuario=None,
        preferencias_logistica=None,
        principios_libro=None,
        rutina_final=None,
        step_completed="",
        error=None,
        timestamp=datetime.now().isoformat(),
        debug_info=None,
        respuesta_usuario=""
    )

@pytest.fixture
def user_profile_sample() -> Dict[str, Any]:
    """Provides a realistic user profile dictionary."""
    return {
        "user_id": "test_user",
        "name": "Test User",
        "level": "intermedio",
        "workout_count": 15,
        "objetivo": "hipertrofia",
        "frecuencia_semanal": 4,
        "ejercicios_favoritos": ["Sentadilla", "Press banca"],
        "restricciones": ["rodilla izquierda"],
        "preferencias_logistica": {
            "equipamiento_disponible": "gimnasio completo",
            "dias_preferidos": ["lunes", "miércoles", "viernes"],
            "duracion_sesion_min": 60
        },
        "rutina_activa": None,
        "created_at": "2024-01-01",
        "last_login": "2025-10-25"
    }


@pytest.fixture
def populated_graph_state(empty_graph_state: GraphState, user_profile_sample: Dict[str, Any]) -> GraphState:
    """Provides a GraphState with the perfil_usuario field populated."""
    state = empty_graph_state.copy()
    state["perfil_usuario"] = user_profile_sample
    state["step_completed"] = "context_loaded"
    return state


@pytest.fixture
def temp_users_dir(tmp_path: Path, user_profile_sample: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Creates a temporary directory mimicking data/users/ with a test user file.
    Also mocks the Config.USERS_DIR setting to point to this temp directory.
    """
    users_dir = tmp_path / "users"
    users_dir.mkdir()

    # Create user_profile_sample.json
    user_id = user_profile_sample["user_id"]
    user_json_path = users_dir / f"{user_id}.json"
    user_json_path.write_text(json.dumps(user_profile_sample, indent=2, ensure_ascii=False), encoding='utf-8')

    # Create a corrupted JSON file
    corrupt_json_path = users_dir / "corrupt_user.json"
    corrupt_json_path.write_text('{"user_id": "corrupt_user", "name": "Bad JSON",', encoding='utf-8')

    # Create an incomplete profile file
    incomplete_json_path = users_dir / "incomplete_user.json"
    incomplete_json_path.write_text(json.dumps({
        "user_id": "incomplete_user",
        "name": "Incomplete",
        "restricciones": []
        }, indent=2, ensure_ascii=False), encoding='utf-8')

    # Mock Config.USERS_DIR to use this temporary directory
    monkeypatch.setattr(Config, 'USERS_DIR', users_dir)

    return users_dir


@pytest.fixture
def sample_text_with_table() -> str:
    """Provides sample text including a table, mimicking PDF content."""
    return """
Párrafo inicial sobre calentamiento. Es importante preparar los músculos.

Tabla 16. Test de Rodilla y Compensatorios
===========================================
| Prueba         | Indicador          | Compensatorio Recomendado        | Pág |
|----------------|--------------------|---------------------------------|-----|
| Dolor Anterior | Debilidad Glúteos  | Puente de Glúteo (3x15)         | 107 |
| Valgo Dinámico | Abductores Débiles | Clamshell con Banda (3x12 c/l)  | 108 |
| ...            | ...                | ...                             | ... |
===========================================

Párrafo después de la tabla, hablando sobre la técnica del Puente de Glúteo.
Asegúrate de apretar los glúteos en la parte superior del movimiento.
"""

@pytest.fixture
def sample_long_paragraphs() -> str:
    """Provides multiple long paragraphs for chunking tests."""
    para1 = "Párrafo uno. " + ("Este es texto de relleno largo para probar el chunking. " * 30) + "Final del párrafo uno."
    para2 = "Párrafo dos, diferente. " + ("Continuamos con más texto de ejemplo para asegurar la división. " * 30) + "Final del párrafo dos."
    para3 = "Párrafo tres, concluyendo. " + ("El último bloque de texto para verificar el overlap y tamaño. " * 30) + "Final del párrafo tres."
    return f"{para1}\n\n{para2}\n\n{para3}"


@pytest.fixture
def chunk_splitter():
    """Provides an instance of the configured text splitter."""
    return get_semantic_text_splitter()

@pytest.fixture(scope="session")
def vectorstore(tmp_path_factory) -> Generator[Any, None, None]:
    """
    Provides a real Chroma vector store instance, building it if necessary.
    Uses a session-scoped temporary directory if CHROMA_DIR doesn't exist.
    """
    config = Config()
    chroma_persist_dir = config.CHROMA_DIR
    if not os.path.exists(chroma_persist_dir):
        temp_dir = tmp_path_factory.mktemp("chroma_db_session")
        chroma_persist_dir = temp_dir
        print(f"\n[Fixture] Vectorstore doesn't exist, building in session temp dir: {temp_dir}")
        original_chroma_dir = Config.CHROMA_DIR
        Config.CHROMA_DIR = temp_dir
        manager = VectorStoreManager()
        try:
            vs = manager.get_or_create_vectorstore()
        finally:
            Config.CHROMA_DIR = original_chroma_dir
        yield vs
    else:
        print(f"\n[Fixture] Loading existing vectorstore from: {chroma_persist_dir}")
        manager = VectorStoreManager()
        vs = manager.get_or_create_vectorstore()
        yield vs

@pytest.fixture
def principle_extractor() -> PrincipleExtractor:
    """Provides an instance of PrincipleExtractor."""
    return PrincipleExtractor()

@pytest.fixture
def mock_openai_chat_completions(monkeypatch: pytest.MonkeyPatch):
    """
    Mocks the ChatOpenAI().invoke call.
    
    ✅ CORRECCIÓN FINAL (2025-10-26):
    - Retorna AIMessage(content=json_string)
    - Errores persisten durante reintentos (persist_error=True por defecto)
    - Se limpian automáticamente entre tests vía yield
    """
    from langchain_core.messages import AIMessage
    
    # ✅ Estado del mock
    mock_state = {
        "force_error": None,
        "persist_error": True,  # ← NUEVO: Controla si el error persiste
        "custom_responses": {},
        "call_count": 0
    }

    def add_mock_response(identifier: str, response: Any, persist: bool = True):
        """
        Configura el comportamiento del mock.
        
        Args:
            identifier: Identificador (para debugging)
            response: Exception (se lanzará) o RutinaActiva
            persist: Si True, error persiste en reintentos. Si False, solo una vez.
        """
        if isinstance(response, Exception):
            print(f"[Mock Setup] Configured to raise: {response} (persist={persist})")
            mock_state["force_error"] = response
            mock_state["persist_error"] = persist  # ← NUEVO
        else:
            mock_state["custom_responses"][identifier] = response

    def mock_invoke(self, *args, **kwargs):
        """Mock implementation of ChatOpenAI.invoke()"""
        mock_state["call_count"] += 1
        call_num = mock_state["call_count"]
        
        # ================================================================
        # PRIORIDAD 1: Error forzado
        # ================================================================
        if mock_state["force_error"] is not None:
            error = mock_state["force_error"]
            
            # ✅ CORRECCIÓN: Solo limpiar si persist_error=False
            if not mock_state["persist_error"]:
                mock_state["force_error"] = None
                print(f"[Mock OpenAI #{call_num}] Raising error (one-time): {error}")
            else:
                print(f"[Mock OpenAI #{call_num}] Raising error (persistent): {error}")
                        
            raise error
        
        # ================================================================
        # PRIORIDAD 2: Procesar input normal
        # ================================================================
        # Extraer el input
        if args:
            prompt_input = args[0]
        else:
            prompt_input = kwargs.get('input', kwargs)
        
        # Convertir a string para análisis
        if hasattr(prompt_input, 'to_messages'):
            messages = prompt_input.to_messages()
            content_str = str(messages)
        elif hasattr(prompt_input, 'to_string'):
            content_str = prompt_input.to_string()
        else:
            content_str = str(prompt_input)
        
        # ================================================================
        # PRIORIDAD 3: Detectar tipo de solicitud y responder
        # ================================================================
        
        # Caso A: Solicitud de generación de rutina
        if 'principios' in content_str.lower() or 'perfil' in content_str.lower():
            print(f"[Mock OpenAI #{call_num}] Detected routine generation request.")
            
            # Crear rutina válida
            principios = create_valid_principles()
            rutina = create_valid_routine(principios)
            
            # ✅ Serializar y envolver en AIMessage
            json_output = rutina.model_dump_json(indent=2)
            return AIMessage(content=json_output)
        
        # Caso B: Solicitud con "parse_error" (para test de parsing)
        if 'parse_error' in content_str.lower():
            print(f"[Mock OpenAI #{call_num}] Returning malformed JSON.")
            # JSON intencionalmente roto
            return AIMessage(content='{"nombre": "incomplete...')
        
        # Caso C: Response personalizada por identificador
        for identifier, response in mock_state["custom_responses"].items():
            if identifier in content_str:
                print(f"[Mock OpenAI #{call_num}] Found custom response: {identifier}")
                if isinstance(response, RutinaActiva):
                    json_output = response.model_dump_json(indent=2)
                    return AIMessage(content=json_output)
                return response
        
        # ================================================================
        # PRIORIDAD 4: Default - Rutina válida genérica
        # ================================================================
        print(f"[Mock OpenAI #{call_num}] No specific match. Returning default routine.")
        principios = create_valid_principles()
        rutina = create_valid_routine(principios)
        json_output = rutina.model_dump_json(indent=2)
        return AIMessage(content=json_output)

    # ✅ Patchear el método de clase
    monkeypatch.setattr(
        "langchain_openai.chat_models.base.ChatOpenAI.invoke",
        mock_invoke
    )

    # ✅ YIELD: Retornar el helper y limpiar al final del test
    yield add_mock_response
    
    # ✅ AUTO-LIMPIEZA: Resetear estado después de cada test
    mock_state["force_error"] = None
    mock_state["persist_error"] = True
    mock_state["call_count"] = 0
    print("[Mock Cleanup] State reset for next test")


@pytest.fixture
def mock_principle_extractor_chain(monkeypatch: pytest.MonkeyPatch):
    """
    Mocks the entire PrincipleExtractor chain's invoke method
    to bypass RAG and LLM calls for node testing.
    
    ✅ CORRECCIÓN: Eliminar 'self' del mock_invoke para que funcione como función simple
    """
    mock_responses = {"default": create_valid_principles()}

    def add_mock_response(profile_identifier: str, response: PrincipiosExtraidos | Exception):
        """Set a specific response for a profile characteristic."""
        mock_responses[profile_identifier] = response

    # ✅ CORRECCIÓN: Quitar 'self' - esta es una función, no un método
    def mock_invoke(perfil_usuario: Dict[str, Any]):
        """Mocked invoke method for the extraction chain."""
        level = perfil_usuario.get("level", "unknown")
        user_id = perfil_usuario.get("user_id", "unknown")
        print(f"[Mock RAG Chain] Invoked for user '{user_id}', level '{level}'.")

        if level in mock_responses:
            response = mock_responses[level]
        elif "no_citations" in mock_responses and level == "no_citations_test":
            response = mock_responses["no_citations"]
        elif "error" in mock_responses and level == "error_test":
            response = mock_responses["error"]
        else:
            response = mock_responses["default"]

        if isinstance(response, Exception):
            print(f"[Mock RAG Chain] Raising predefined exception: {response}")
            raise response
        else:
            print(f"[Mock RAG Chain] Returning predefined principles: RIR={response.intensidad_RIR}, Citations={len(response.citas_fuente)}")
            return response

    mock_chain = MagicMock()
    mock_chain.invoke = mock_invoke  # ✅ Ahora funciona correctamente

    monkeypatch.setattr(
        "rag.principle_extractor.PrincipleExtractor.get_extraction_chain",
        lambda self: mock_chain
    )

    return add_mock_response