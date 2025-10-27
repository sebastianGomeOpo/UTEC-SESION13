# -----------------------------------------------------------------------------
# Fase 5: Tests End-to-End
# Valida los flujos completos del grafo compilado.
# -----------------------------------------------------------------------------

import json
import shutil
from pathlib import Path
from typing import Generator, Dict  # <--- CORRECCIÓN 1: 'Dict' importado aquí

import pytest
from unittest.mock import MagicMock

# Imports del proyecto
from config.settings import Config
from agents.graph_state import GraphState, create_initial_state
from rag.models import ECI ,PrincipiosExtraidos, RutinaActiva# Necesario para mockear
from utils.logger import setup_logger  # <--- CORRECCIÓN 2: 'setup_logger' importado aquí

# Importar la función de main (para testearla)
from main import determinar_request_type

# Importar el grafo compilado
# Esta importación compila el grafo (si build_graph() se ejecuta al importar)
try:
    from agents.entrenador import graph, extract_principles, generate_routine
except Exception as e:
    pytest.fail(f"Error CRÍTICO: No se pudo importar o compilar el grafo. {e}")


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graph_compiled():
    """Provee el grafo compilado importado."""
    assert graph is not None, "El grafo no se compiló/importó correctamente"
    return graph


@pytest.fixture
def temp_project_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Dict[str, Path], None, None]:
    """
    Crea una estructura de directorios temporal (data/users, data/historial)
    y monkeypatchea Config para que apunte a ella.
    """
    base_dir = tmp_path / "temp_project"
    base_dir.mkdir()
    
    users_dir = base_dir / "users"
    users_dir.mkdir()
    
    historial_dir = base_dir / "historial"
    historial_dir.mkdir()
    
    # Monkeypatch las constantes de Config
    monkeypatch.setattr(Config, 'USERS_DIR', users_dir)
    # Patch DATA_DIR para que 'historial' se resuelva correctamente
    monkeypatch.setattr(Config, 'DATA_DIR', base_dir) 
    
    yield {"users": users_dir, "historial": historial_dir}
    
    # Cleanup (aunque tmp_path lo hace, esto es explícito)
    shutil.rmtree(base_dir)


@pytest.fixture
def user_e2e_profile(temp_project_dirs: Dict[str, Path]) -> str:
    """
    Crea un archivo de perfil de usuario de prueba en el directorio temporal.
    Retorna el user_id.
    """
    user_id = "test_user_e2e"
    user_file_path = temp_project_dirs["users"] / f"{user_id}.json"
    
    profile_data = {
      "user_id": user_id,
      "name": "Test User E2E",
      "level": "intermedio",
      "objetivo": "hipertrofia",
      "restricciones": ["rodilla izquierda"],
      "preferencias_logistica": {
        "equipamiento_disponible": "gimnasio completo",
        "dias_preferidos": ["lunes", "miércoles", "viernes"],
        "duracion_sesion_min": 60
      },
      "rutina_activa": None # CRÍTICO para testear save_routine
    }
    
    with open(user_file_path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)
        
    return user_id


@pytest.fixture
def mock_crg_nodes(monkeypatch: pytest.MonkeyPatch):
    """
    Mockea los nodos C-R-G (extract_principles y generate_routine)
    para aislar el testeo del grafo (routing) de las APIs externas (LLM/RAG).
    """
    
    # 1. Mock para extract_principles
    def mock_extract(state: GraphState) -> GraphState:
        """Mock que retorna principios válidos con citas."""
        logger = setup_logger("mock_extract")
        logger.info("--- MOCK Extract Principles ---")
        
        # Simular éxito
        principios = PrincipiosExtraidos(
            intensidad_RIR="1-2",
            rango_repeticiones="8-12",
            descanso_series_s=90,
            cadencia_tempo="3:0:1:1",
            frecuencia_semanal="3-4 días",
            ECI_recomendados=[
                ECI(
                    nombre_ejercicio="Puente de Glúteo",
                    motivo="Compensación rodilla izquierda",
                    fuente_cita="Página 107",
                    sets=3,
                    reps="15"
                )
            ],
            citas_fuente=["Página 138 (Tabla 20)"] # CRÍTICO: Debe tener citas
        )
        state["principios_libro"] = principios
        state["step_completed"] = "principles_extracted"
        return state

    # 2. Mock para generate_routine
    def mock_generate(state: GraphState) -> GraphState:
        """Mock que retorna una rutina válida."""
        logger = setup_logger("mock_generate")
        logger.info("--- MOCK Generate Routine ---")
        
        principios = state.get("principios_libro")
        if not principios:
            state["error"] = "Mock generate no recibió principios"
            state["step_completed"] = "generate_routine_error"
            return state

        # Simular éxito
        rutina = RutinaActiva(
            nombre="Rutina Mock E2E",
            sesiones=[
                # ... (una sesión simple sería suficiente) ...
            ],
            principios_aplicados=principios,
            validez_semanas=4
        )
        state["rutina_final"] = rutina
        state["step_completed"] = "routine_generated"
        return state

    # 3. Aplicar los Mocks
    # Asumimos que 'agents.entrenador' importa los nodos
    monkeypatch.setattr("agents.entrenador.extract_principles", mock_extract)
    monkeypatch.setattr("agents.entrenador.generate_routine", mock_generate)
    
    # También mockear los imports directos si existen en el módulo
    if hasattr(extract_principles, '__module__'):
        monkeypatch.setattr(f"{extract_principles.__module__}.extract_principles", mock_extract)
    if hasattr(generate_routine, '__module__'):
        monkeypatch.setattr(f"{generate_routine.__module__}.generate_routine", mock_generate)


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("user_input, expected_type", [
    ("Crea una rutina nueva para hipertrofia", "crear_rutina"),
    ("Quiero un plan de entrenamiento", "crear_rutina"),
    ("registra 5x5 de sentadilla con 100kg", "registrar_ejercicio"),
    ("anota 10x3 de press banca con 80 kg", "registrar_ejercicio"),
    ("Muéstrame mi historial", "consultar_historial"),
    ("qué ejercicios hice la semana pasada", "consultar_historial"),
    ("Hola, ¿cómo estás?", "unknown"),
    ("", "unknown"),
])
def test_determinar_request_type(user_input, expected_type):
    """Valida el clasificador de intención de main.py."""
    assert determinar_request_type(user_input) == expected_type


def test_e2e_crear_rutina_flow(graph_compiled, user_e2e_profile, mock_crg_nodes, temp_project_dirs):
    """
    Test E2E (Happy Path): Flujo C-R-G completo.
    load -> extract -> generate -> save -> END
    """
    user_id = user_e2e_profile
    initial_state = create_initial_state(user_id, "crear_rutina")
    initial_state["user_message"] = "Crea una rutina por favor"
    
    # Invocar el grafo
    final_state = graph_compiled.invoke(initial_state)
    
    # Validar estado final
    assert final_state["error"] is None, f"Hubo un error inesperado: {final_state['error']}"
    assert final_state["step_completed"] == "saved"
    assert "Rutina guardada" in final_state["respuesta_usuario"]
    
    # Validar persistencia (que el archivo JSON fue modificado)
    user_file_path = temp_project_dirs["users"] / f"{user_id}.json"
    with open(user_file_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        
    assert saved_data["rutina_activa"] is not None
    assert saved_data["rutina_activa"]["nombre"] == "Rutina Mock E2E"
    assert "updated_at" in saved_data


def test_e2e_registrar_ejercicio_flow(graph_compiled, user_e2e_profile, temp_project_dirs):
    """
    Test E2E (Happy Path): Flujo Legacy Register.
    load -> call_legacy_register -> END
    """
    user_id = user_e2e_profile
    user_message = "anota 5x5 de sentadilla con 100kg"
    
    initial_state = create_initial_state(user_id, "registrar_ejercicio")
    initial_state["user_message"] = user_message
    
    # Invocar el grafo
    final_state = graph_compiled.invoke(initial_state)
    
    # Validar estado final
    assert final_state["error"] is None, f"Hubo un error inesperado: {final_state['error']}"
    assert final_state["step_completed"] == "call_legacy_register"
    assert "Registrado: sentadilla" in final_state["respuesta_usuario"]
    
    # Validar persistencia (que el historial fue creado)
    historial_file_path = temp_project_dirs["historial"] / f"{user_id}.json"
    assert historial_file_path.exists()
    with open(historial_file_path, "r", encoding="utf-8") as f:
        historial_data = json.load(f)
    assert len(historial_data) == 1
    assert historial_data[0]["ejercicio"] == "sentadilla"
    assert historial_data[0]["peso_kg"] == 100.0


def test_e2e_consultar_historial_flow(graph_compiled, user_e2e_profile):
    """
    Test E2E (Happy Path): Flujo Legacy Query.
    load -> call_legacy_query -> END
    """
    user_id = user_e2e_profile
    initial_state = create_initial_state(user_id, "consultar_historial")
    
    # Invocar el grafo
    final_state = graph_compiled.invoke(initial_state)
    
    # Validar estado final
    assert final_state["error"] is None, f"Hubo un error inesperado: {final_state['error']}"
    assert final_state["step_completed"] == "call_legacy_query"
    # El historial está vacío (porque es un test separado)
    assert "No hay entrenamientos registrados" in final_state["respuesta_usuario"]


def test_e2e_error_handling_user_not_found(graph_compiled, temp_project_dirs):
    """
    Test E2E (Sad Path): Error en load_context.
    load -> handle_error -> END
    """
    user_id = "usuario_inexistente"
    initial_state = create_initial_state(user_id, "crear_rutina")
    
    # Invocar el grafo
    final_state = graph_compiled.invoke(initial_state)
    
    # Validar estado final
    assert final_state["error"] is not None, "Se esperaba un error"
    assert final_state["step_completed"] == "error" # El nodo handle_error setea esto
    assert "no encontrado" in final_state["respuesta_usuario"]


def test_e2e_error_handling_unknown_request(graph_compiled, user_e2e_profile):
    """
    Test E2E (Sad Path): Error en routing (request_type unknown).
    load -> route_after_load -> handle_error -> END
    """
    user_id = user_e2e_profile
    initial_state = create_initial_state(user_id, "unknown")
    initial_state["user_message"] = "Hola mundo"
    
    # Invocar el grafo
    final_state = graph_compiled.invoke(initial_state)
    
    # Validar estado final
    assert final_state["error"] is not None, "Se esperaba un error"
    assert final_state["step_completed"] == "error"
    assert "Tipo de request desconocido: unknown" in final_state["error"]
    assert "request desconocido" in final_state["respuesta_usuario"]


@pytest.fixture
def mock_extract_no_citations(monkeypatch: pytest.MonkeyPatch):
    """Mockea extract_principles para que retorne principios SIN citas."""
    def mock_extract_alucinacion(state: GraphState) -> GraphState:
        logger = setup_logger("mock_extract_alucinacion")
        logger.info("--- MOCK Extract (Sin Citas) ---")
        principios = PrincipiosExtraidos(
            intensidad_RIR="1-2",
            rango_repeticiones="8-12",
            descanso_series_s=90,
            cadencia_tempo="3:0:1:1",
            frecuencia_semanal="3-4 días",
            ECI_recomendados=[],
            citas_fuente=[] # <-- ¡Sin citas!
        )
        state["principios_libro"] = principios
        state["step_completed"] = "principles_extracted"
        return state

    monkeypatch.setattr("agents.entrenador.extract_principles", mock_extract_alucinacion)
    if hasattr(extract_principles, '__module__'):
        monkeypatch.setattr(f"{extract_principles.__module__}.extract_principles", mock_extract_alucinacion)


def test_e2e_error_handling_no_citations(graph_compiled, user_e2e_profile, mock_extract_no_citations):
    """
    Test E2E (Sad Path): Error en routing (Alucinación detectada).
    load -> extract -> route_after_extract -> handle_error -> END
    """
    user_id = user_e2e_profile
    initial_state = create_initial_state(user_id, "crear_rutina")
    
    # Invocar el grafo
    final_state = graph_compiled.invoke(initial_state)
    
    # Validar estado final
    assert final_state["error"] is not None, "Se esperaba un error"
    assert final_state["step_completed"] == "error"
    assert "Alucinación detectada" in final_state["error"]
    assert "Alucinación detectada" in final_state["respuesta_usuario"]