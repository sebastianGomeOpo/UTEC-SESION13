# Standard Library Imports
import json
import os
import shutil
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
    # Note: 'confianza' was in spec but not in model, removed. Add if needed.
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
                RIR="3", # ECIs can have different RIR
                tempo="2:1:2:1", # ECIs can have different tempo
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
                        reps=reps, # Match principles
                        RIR=rir, # Match principles
                        tempo=tempo, # Match principles
                        descanso_s=descanso, # Match principles
                        notas="Técnica estricta",
                    ),
                    *eci_ejercicios, # Include generated ECIs
                    Ejercicio(
                        nombre="Curl femoral",
                        tipo="accesorio",
                        sets=3,
                        reps="12-15",
                        RIR="2-3", # Different RIR is ok for accessory
                        tempo="2:0:2:0",
                        descanso_s=60,
                        notas=None,
                    )
                ],
            )
        ],
        principios_aplicados=principles_dict if isinstance(principles, dict) else principles # Store the used principles
    )

# --- Fixtures ---

@pytest.fixture
def empty_graph_state() -> GraphState:
    """Provides an empty/initial GraphState dictionary."""
    # Ensure all keys from TypedDict are present, even if None/empty
    return GraphState(
        user_id="test_user", # Default test user
        request_type="crear_rutina",
        perfil_usuario=None,
        preferencias_logistica=None, # Added based on spec
        principios_libro=None,
        rutina_final=None,
        step_completed="",
        error=None,
        timestamp=datetime.now().isoformat(),
        debug_info=None,
        # Add any other keys defined in GraphState
        respuesta_usuario="" # Added based on spec
    )

@pytest.fixture
def user_profile_sample() -> Dict[str, Any]:
    """Provides a realistic user profile dictionary."""
    return {
        "user_id": "test_user",
        "name": "Test User",
        "level": "intermedio", # Correct key based on JSON
        "workout_count": 15,
        "objetivo": "hipertrofia",
        "frecuencia_semanal": 4,
        "ejercicios_favoritos": ["Sentadilla", "Press banca"],
        "restricciones": ["rodilla izquierda"], # Example restriction
        "preferencias_logistica": {
            "equipamiento_disponible": "gimnasio completo", # Correct key based on JSON
            "dias_preferidos": ["lunes", "miércoles", "viernes"], # Correct key based on JSON
            "duracion_sesion_min": 60 # Correct key based on JSON
        },
        "rutina_activa": None, # Added based on JSON update
        "created_at": "2024-01-01",
        "last_login": "2025-10-25"
    }


@pytest.fixture
def populated_graph_state(empty_graph_state: GraphState, user_profile_sample: Dict[str, Any]) -> GraphState:
    """Provides a GraphState with the perfil_usuario field populated."""
    state = empty_graph_state.copy()
    state["perfil_usuario"] = user_profile_sample
    state["step_completed"] = "context_loaded" # Simulate context loaded
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
    corrupt_json_path.write_text('{"user_id": "corrupt_user", "name": "Bad JSON",', encoding='utf-8') # Missing closing brace

    # Create an incomplete profile file
    incomplete_json_path = users_dir / "incomplete_user.json"
    incomplete_json_path.write_text(json.dumps({
        "user_id": "incomplete_user",
        "name": "Incomplete",
        # Missing "level" and "objetivo"
        "restricciones": []
        }, indent=2, ensure_ascii=False), encoding='utf-8')


    # Mock Config.USERS_DIR to use this temporary directory
    # We mock it on the Config class itself before any instance is created in tests
    monkeypatch.setattr(Config, 'USERS_DIR', users_dir)

    # Ensure the original USERS_DIR is restored after the test if needed elsewhere,
    # although monkeypatch usually handles this cleanup automatically.

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
    # Renamed from chunking_strategy based on user's file name
    return get_semantic_text_splitter()

# Scope="session" is crucial: build/load the vector store only ONCE per test session
@pytest.fixture(scope="session")
def vectorstore(tmp_path_factory) -> Generator[Any, None, None]:
    """
    Provides a real Chroma vector store instance, building it if necessary.
    Uses a session-scoped temporary directory if CHROMA_DIR doesn't exist.
    """
    config = Config()
    # Use a session-specific temp dir only if the main one doesn't exist
    # to avoid rebuilding repeatedly during local dev testing if DB exists
    chroma_persist_dir = config.CHROMA_DIR
    if not os.path.exists(chroma_persist_dir):
        # Create a temp dir that persists for the whole session
        temp_dir = tmp_path_factory.mktemp("chroma_db_session")
        chroma_persist_dir = temp_dir
        print(f"\n[Fixture] Vectorstore doesn't exist, building in session temp dir: {temp_dir}")
        # Monkeypatch CHROMA_DIR for the manager during build ONLY if using temp dir
        # Be careful if other tests rely on the original Config path
        original_chroma_dir = Config.CHROMA_DIR
        Config.CHROMA_DIR = temp_dir # Temporarily override
        manager = VectorStoreManager() # Uses the overridden path
        try:
            vs = manager.get_or_create_vectorstore() # Builds in temp dir
        finally:
            Config.CHROMA_DIR = original_chroma_dir # Restore original path
        yield vs # Yield the store built in temp dir
    else:
        print(f"\n[Fixture] Loading existing vectorstore from: {chroma_persist_dir}")
        manager = VectorStoreManager() # Uses the existing path
        vs = manager.get_or_create_vectorstore() # Loads from existing path
        yield vs # Yield the loaded store

@pytest.fixture
def principle_extractor() -> PrincipleExtractor:
    """Provides an instance of PrincipleExtractor."""
    # Ensure environment variables (like OPENAI_API_KEY) are set for real tests
    # or mock the LLM/embeddings if needed for isolated testing
    return PrincipleExtractor()

@pytest.fixture
def mock_openai_chat_completions(monkeypatch: pytest.MonkeyPatch):
    """
    Mocks the ChatOpenAI().invoke call to return predefined responses
    for testing generate_routine node without hitting the actual API.
    """
    mock_responses = {} # Store mock responses keyed by input characteristics if needed

    def add_mock_response(identifier: str, response: Any):
        """Helper to set up specific mock responses."""
        mock_responses[identifier] = response

    def mock_invoke(*args, **kwargs):
        # args[0] is typically the input dictionary/prompt value
        prompt_input = args[0] if args else kwargs

        # Simple example: return a valid routine structure if prompt contains "generate"
        # More complex logic can inspect prompt_input keys/values
        if isinstance(prompt_input, dict) and "principios" in prompt_input:
            print("[Mock OpenAI] Returning predefined valid routine.")
            # Use the factory to create a consistent valid routine
            # We need principles here, try to get from input or use default
            principles_dict = prompt_input.get("principios", {})
            principles_obj = PrincipiosExtraidos(**principles_dict) if principles_dict else create_valid_principles()
            # Return the Pydantic object itself, as the parser is part of the chain
            return create_valid_routine(principles_obj)

        # Example: Simulate parsing error
        elif isinstance(prompt_input, str) and "parse_error" in prompt_input:
            print("[Mock OpenAI] Returning malformed JSON string to cause parse error.")
            # The parser will try to parse this, causing OutputParserException
            return MagicMock(content='{"rutina": "bad json", ...') # Mock AIMessage with bad content

        # Default fallback or raise error
        print(f"[Mock OpenAI] No specific mock found for input: {prompt_input}. Returning default or error.")
        # Default valid routine if no other match
        return create_valid_routine(create_valid_principles())
        # raise NotImplementedError("Mock invoke called with unexpected input")

    # Mock the 'invoke' method of the ChatOpenAI class instance used in the node
    # Adjust the target path if necessary based on where ChatOpenAI is instantiated
    monkeypatch.setattr("langchain_openai.ChatOpenAI.invoke", mock_invoke)

    # Return the helper function to allow tests to set specific responses
    return add_mock_response

@pytest.fixture
def mock_principle_extractor_chain(monkeypatch: pytest.MonkeyPatch):
    """
    Mocks the entire PrincipleExtractor chain's invoke method
    to bypass RAG and LLM calls for node testing.
    """
    mock_responses = {"default": create_valid_principles()} # Default valid response

    def add_mock_response(profile_identifier: str, response: PrincipiosExtraidos | Exception):
        """Set a specific response for a profile characteristic."""
        mock_responses[profile_identifier] = response

    def mock_invoke(self, perfil_usuario: Dict[str, Any]):
        """Mocked invoke method for the extraction chain."""
        # Example: Return specific response based on user level
        level = perfil_usuario.get("level", "unknown")
        user_id = perfil_usuario.get("user_id", "unknown")
        print(f"[Mock RAG Chain] Invoked for user '{user_id}', level '{level}'.")

        if level in mock_responses:
            response = mock_responses[level]
        elif "no_citations" in mock_responses and level == "no_citations_test": # Specific trigger
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

    # The target is the invoke method of the RunnableSequence *returned by* get_extraction_chain
    # This is tricky. It might be easier to mock PrincipleExtractor.get_extraction_chain
    # to return a MagicMock object whose 'invoke' method is the one defined above.

    mock_chain = MagicMock()
    mock_chain.invoke = mock_invoke

    monkeypatch.setattr(
        "rag.principle_extractor.PrincipleExtractor.get_extraction_chain",
        lambda self: mock_chain # Return the mock chain instead of the real one
    )

    return add_mock_response # Return helper to configure mocks in tests
