# Standard Library Imports
import json
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

# Third-party Imports
import pytest
from langchain_core.output_parsers import OutputParserException

# Project Imports (ensure paths are correct)
try:
    from agents.graph_state import GraphState
    from agents.nodes.load_context import load_context
    from agents.nodes.extract_principles import extract_principles
    from agents.nodes.generate_routine import generate_routine
    from agents.nodes.save_routine import save_routine
    from agents.nodes.handle_error import handle_error, ERROR_MESSAGE_MAP, DEFAULT_ERROR_MESSAGE
    from rag.models import PrincipiosExtraidos, RutinaActiva, Sesion, Ejercicio, ECI
    from config.settings import Config # Needed for mocking paths if not done globally
except ImportError as e:
    pytest.exit(f"Failed to import node functions or dependencies: {e}", 1)

# Import helper factories from conftest
from .conftest import create_valid_principles, create_valid_routine


# --- Test Load Context Node ---

def test_load_context_fills_profile(empty_graph_state: GraphState, temp_users_dir: Path):
    """
    Given: GraphState with a valid user_id pointing to an existing user JSON.
    When: load_context node is executed.
    Then: state["perfil_usuario"] should be populated with data from the JSON.
          state["step_completed"] should be "context_loaded".
          state["error"] should be None or empty.
    """
    # Arrange
    state = empty_graph_state.copy()
    state["user_id"] = "test_user" # This user exists in temp_users_dir fixture

    # Act
    result_state = load_context(state)

    # Assert
    assert result_state["error"] is None or result_state["error"] == "", "Node should not report an error on success."
    assert result_state["step_completed"] == "context_loaded", "Step completed status is incorrect."
    assert result_state["perfil_usuario"] is not None, "User profile should be loaded."
    assert isinstance(result_state["perfil_usuario"], dict), "User profile should be a dictionary."
    assert result_state["perfil_usuario"].get("user_id") == "test_user"
    assert "level" in result_state["perfil_usuario"], "Required field 'level' missing in loaded profile."
    assert "objetivo" in result_state["perfil_usuario"], "Required field 'objetivo' missing in loaded profile."
    assert result_state["perfil_usuario"].get("name") == "Test User"


def test_load_context_handles_missing_user(empty_graph_state: GraphState, temp_users_dir: Path):
    """
    Given: GraphState with a user_id that does not correspond to a JSON file.
    When: load_context node is executed.
    Then: state["error"] should contain a 'not found' message.
          state["step_completed"] should indicate an error.
          state["perfil_usuario"] should remain None or empty.
    """
    # Arrange
    state = empty_graph_state.copy()
    state["user_id"] = "nonexistent_user"

    # Act
    result_state = load_context(state)

    # Assert
    assert result_state["error"] is not None and result_state["error"] != "", "Node should report an error for missing user."
    assert "no encontrado" in result_state["error"].lower(), "Error message should indicate user not found."
    assert "error" in result_state["step_completed"].lower(), "Step completed should indicate an error state."
    assert result_state["perfil_usuario"] is None, "User profile should not be populated on error."


def test_load_context_handles_corrupt_json(empty_graph_state: GraphState, temp_users_dir: Path):
    """
    Given: GraphState with a user_id pointing to a corrupt JSON file.
    When: load_context node is executed.
    Then: state["error"] should contain a 'corrupt' or 'decode' message.
          state["step_completed"] should indicate an error.
    """
    # Arrange
    state = empty_graph_state.copy()
    state["user_id"] = "corrupt_user" # This file exists but is invalid JSON

    # Act
    result_state = load_context(state)

    # Assert
    assert result_state["error"] is not None and result_state["error"] != "", "Node should report an error for corrupt JSON."
    assert "corrupto" in result_state["error"].lower() or "decode" in result_state["error"].lower(), \
        "Error message should indicate JSON corruption."
    assert "error" in result_state["step_completed"].lower()

def test_load_context_handles_incomplete_profile(empty_graph_state: GraphState, temp_users_dir: Path):
    """
    Given: GraphState with a user_id pointing to JSON missing required fields.
    When: load_context node is executed.
    Then: state["error"] should indicate missing fields.
          state["step_completed"] should indicate an error.
    """
    # Arrange
    state = empty_graph_state.copy()
    state["user_id"] = "incomplete_user" # This file is valid JSON but missing level/objetivo

    # Act
    result_state = load_context(state)

    # Assert
    assert result_state["error"] is not None and result_state["error"] != "", "Node should report an error for incomplete profile."
    assert "incompleto" in result_state["error"].lower(), "Error message should indicate profile incompleteness."
    assert "level" in result_state["error"].lower() or "objetivo" in result_state["error"].lower(), "Error message should mention missing fields."
    assert "error" in result_state["step_completed"].lower()


# --- Test Extract Principles Node ---

def test_extract_principles_success(populated_graph_state: GraphState, mock_principle_extractor_chain):
    """
    Given: GraphState with a valid user profile.
           Mocked RAG chain returning valid principles with citations.
    When: extract_principles node is executed.
    Then: state["principios_libro"] should be populated with a PrincipiosExtraidos object.
          The object must have non-empty citas_fuente.
          state["step_completed"] should be "principles_extracted".
          state["error"] should be None or empty.
    """
    # Arrange
    state = populated_graph_state.copy()
    # Mock is applied via fixture

    # Act
    result_state = extract_principles(state)

    # Assert
    assert result_state["error"] is None or result_state["error"] == "", "Node should not report an error on success."
    assert result_state["step_completed"] == "principles_extracted", "Step completed status is incorrect."
    assert result_state["principios_libro"] is not None, "Principles should be extracted."
    assert isinstance(result_state["principios_libro"], PrincipiosExtraidos), "Extracted principles should be a PrincipiosExtraidos object."
    assert hasattr(result_state["principios_libro"], 'citas_fuente'), "Principles object missing citations."
    assert len(result_state["principios_libro"].citas_fuente) > 0, "CRITICAL: Citations list is empty."


def test_extract_principles_detects_hallucination(populated_graph_state: GraphState, mock_principle_extractor_chain):
    """
    Given: GraphState with a valid user profile.
           Mocked RAG chain configured to return principles WITHOUT citations.
    When: extract_principles node is executed.
    Then: state["error"] should contain 'citas' or 'alucinación'.
          state["step_completed"] should indicate an error.
          state["principios_libro"] should be None or not set.
    """
    # Arrange
    state = populated_graph_state.copy()
    # Configure the mock to return principles without citations for this specific test
    principles_no_citations = create_valid_principles(citas=[])
    mock_principle_extractor_chain("no_citations_test", principles_no_citations) # Special trigger for mock
    state["perfil_usuario"]["level"] = "no_citations_test" # Match the trigger

    # Act
    result_state = extract_principles(state)

    # Assert
    assert result_state["error"] is not None and result_state["error"] != "", "Node should report an error for missing citations."
    assert "citas" in result_state["error"].lower() or "alucinación" in result_state["error"].lower(), \
        "Error message should mention citations or hallucination."
    assert "error" in result_state["step_completed"].lower(), "Step completed should indicate an error state."
    # Ensure principles are not added to state if validation fails
    assert result_state.get("principios_libro") is None or isinstance(result_state.get("principios_libro"), dict), \
        "Principles should not be set on citation validation failure."


def test_extract_principles_handles_rag_failure(populated_graph_state: GraphState, mock_principle_extractor_chain):
    """
    Given: GraphState with a valid user profile.
           Mocked RAG chain configured to raise an Exception.
    When: extract_principles node is executed.
    Then: state["error"] should capture the RAG error message.
          state["step_completed"] should indicate an error.
    """
    # Arrange
    state = populated_graph_state.copy()
    error_message = "Simulated RAG API Error"
    mock_principle_extractor_chain("error_test", Exception(error_message)) # Configure mock to raise error
    state["perfil_usuario"]["level"] = "error_test" # Match the trigger

    # Act
    result_state = extract_principles(state)

    # Assert
    assert result_state["error"] is not None and result_state["error"] != "", "Node should report an error from RAG."
    assert error_message in result_state["error"], "Error message from RAG should be captured in state."
    assert "error" in result_state["step_completed"].lower()


def test_extract_principles_handles_missing_profile(empty_graph_state: GraphState):
    """
    Given: GraphState where perfil_usuario is missing or None.
    When: extract_principles node is executed.
    Then: state["error"] should indicate the profile is missing.
          state["step_completed"] should indicate an error.
    """
    # Arrange
    state = empty_graph_state.copy()
    state["perfil_usuario"] = None # Explicitly set to None

    # Act
    result_state = extract_principles(state)

    # Assert
    assert result_state["error"] is not None and result_state["error"] != "", "Node should report an error for missing profile."
    assert "perfil" in result_state["error"].lower(), "Error message should mention missing profile."
    assert "error" in result_state["step_completed"].lower()

# --- Test Generate Routine Node ---

# Note: This test relies heavily on the mock_openai_chat_completions fixture
# to simulate LLM responses without actual API calls.
def test_generate_routine_success_and_validation(
    populated_graph_state: GraphState,
    mock_openai_chat_completions # Applies mock to ChatOpenAI.invoke
):
    """
    Given: GraphState with valid profile and principles.
           Mocked LLM returning a valid RutinaActiva object matching principles.
    When: generate_routine node is executed.
    Then: state["rutina_final"] should be populated with the RutinaActiva object.
          Validation within the node should pass.
          state["step_completed"] should be "routine_generated".
          state["error"] should be None or empty.
    """
    # Arrange
    state = populated_graph_state.copy()
    valid_principles = create_valid_principles(rir="1-2", reps="6-8")
    state["principios_libro"] = valid_principles
    # Mock LLM is configured via fixture to return a valid routine based on these principles

    # Act
    result_state = generate_routine(state)

    # Assert
    assert result_state["error"] is None or result_state["error"] == "", f"Node reported unexpected error: {result_state['error']}"
    assert result_state["step_completed"] == "routine_generated", "Step completed status is incorrect."
    assert result_state["rutina_final"] is not None, "Routine should be generated."
    assert isinstance(result_state["rutina_final"], RutinaActiva), "Generated routine is not a RutinaActiva object."
    assert len(result_state["rutina_final"].sesiones) > 0, "Generated routine has no sessions."
    assert len(result_state["rutina_final"].sesiones[0].ejercicios) > 0, "First session has no exercises."

    # Check validation passed (implicitly, as no error was set)
    # Explicit check based on validation logic:
    first_exercise = result_state["rutina_final"].sesiones[0].ejercicios[0]
    assert first_exercise.RIR == valid_principles.intensidad_RIR, "RIR in generated routine doesn't match principles."
    # Add more validation checks if needed

def test_generate_routine_handles_llm_parsing_error(
    populated_graph_state: GraphState,
    mock_openai_chat_completions, # Fixture used to trigger error
    monkeypatch # To modify MAX_RETRIES for faster testing
):
    """
    Given: GraphState with valid profile and principles.
           Mocked LLM configured to return malformed JSON multiple times.
    When: generate_routine node is executed.
    Then: Node should retry MAX_RETRIES times.
          state["error"] should indicate a parsing or invalid format error.
          state["step_completed"] should indicate failure.
    """
    # Arrange
    state = populated_graph_state.copy()
    state["principios_libro"] = create_valid_principles()
    # Configure the mock to return bad JSON via a specific input trigger if needed,
    # or just configure it globally if this test runs in isolation.
    # We'll rely on the mock fixture returning bad JSON based on input content.
    # Let's modify the input slightly to trigger the mock's error condition (if designed).
    state["perfil_usuario"]["objetivo"] = "parse_error_trigger" # Example trigger
    mock_openai_chat_completions("parse_error_trigger", MagicMock(content='{"bad": json}')) # Tell mock what to return

    # Reduce retries for faster test execution
    monkeypatch.setattr("agents.nodes.generate_routine.MAX_RETRIES", 1)

    # Act
    result_state = generate_routine(state)

    # Assert
    assert result_state["error"] is not None and result_state["error"] != "", "Node should report a parsing error."
    assert "json válido" in result_state["error"].lower() or "parser" in result_state["error"].lower(), \
        "Error message should mention invalid JSON or parser issue."
    assert "failed" in result_state["step_completed"].lower(), "Step completed should indicate failure."
    assert result_state.get("rutina_final") is None, "Routine should not be set on parsing failure."

def test_generate_routine_handles_validation_failure(
     populated_graph_state: GraphState,
     mock_openai_chat_completions # Fixture to return mismatched routine
):
    """
    Given: GraphState with valid profile and principles.
           Mocked LLM returning a RutinaActiva object that VIOLATES principles (e.g., wrong RIR).
    When: generate_routine node is executed.
    Then: Node's internal validation should detect the mismatch.
          state["error"] should indicate a validation failure (e.g., "RIR inconsistente").
          state["step_completed"] should indicate failure.
    """
    # Arrange
    state = populated_graph_state.copy()
    principles = create_valid_principles(rir="1-2") # Expect RIR 1-2
    state["principios_libro"] = principles

    # Create a routine that violates the principles
    mismatched_routine = create_valid_routine(principles)
    mismatched_routine.sesiones[0].ejercicios[0].RIR = "3-4" # VIOLATION!

    # Configure the mock LLM to return this mismatched routine
    # Assuming the mock fixture can be configured:
    mock_openai_chat_completions("validation_fail_trigger", mismatched_routine)
    state["perfil_usuario"]["objetivo"] = "validation_fail_trigger" # Trigger the mock

    # Act
    result_state = generate_routine(state)

    # Assert
    assert result_state["error"] is not None and result_state["error"] != "", "Node should report a validation error."
    assert "validación fallida" in result_state["error"].lower() or "rir inconsistente" in result_state["error"].lower(), \
        "Error message should mention validation failure or specific RIR issue."
    assert "failed" in result_state["step_completed"].lower(), "Step completed should indicate failure."
    assert result_state.get("rutina_final") is None, "Routine should not be set on validation failure."
    assert "invalid_generated_routine" in result_state.get("debug_info", {}), "Debug info should contain invalid routine."

# --- Test Save Routine Node ---

def test_save_routine_success_and_backup(
    populated_graph_state: GraphState,
    temp_users_dir: Path # Uses the temp dir with test_user.json
):
    """
    Given: GraphState with a valid user_id and a valid RutinaActiva object.
           An existing user JSON file in the (temporary) USERS_DIR.
    When: save_routine node is executed.
    Then: The user's JSON file should be updated with the new routine under 'rutina_activa'.
          A backup file (e.g., test_user.json.<timestamp>.backup) should be created.
          state["step_completed"] should be "saved".
          state["respuesta_usuario"] should indicate success.
          state["error"] should be None or empty.
    """
    # Arrange
    state = populated_graph_state.copy()
    state["user_id"] = "test_user"
    valid_principles = create_valid_principles()
    state["rutina_final"] = create_valid_routine(valid_principles, user_id="test_user")
    state["step_completed"] = "routine_generated" # Prerequisite step

    user_json_path = temp_users_dir / f"{state['user_id']}.json"
    initial_mtime = user_json_path.stat().st_mtime

    # Act
    result_state = save_routine(state)

    # Assert
    assert result_state["error"] is None or result_state["error"] == "", f"Node reported unexpected error: {result_state['error']}"
    assert result_state["step_completed"] == "saved", "Step completed status is incorrect."
    assert "guardada exitosamente" in result_state.get("respuesta_usuario", "").lower()

    # Verify file update
    assert user_json_path.exists(), "User JSON file should still exist."
    current_mtime = user_json_path.stat().st_mtime
    assert current_mtime > initial_mtime, "User JSON file modification time did not change."

    # Verify content update
    with open(user_json_path, "r", encoding="utf-8") as f:
        updated_data = json.load(f)
    assert "rutina_activa" in updated_data, "'rutina_activa' key missing in updated JSON."
    assert isinstance(updated_data["rutina_activa"], dict), "'rutina_activa' should be a dict."
    # Check a specific field from the saved routine
    assert updated_data["rutina_activa"].get("nombre") == "Rutina Test Válida"
    assert "updated_at" in updated_data, "'updated_at' timestamp missing."

    # Verify backup creation
    backup_files = list(temp_users_dir.glob(f"{state['user_id']}.json.*.backup"))
    assert len(backup_files) >= 1, "Backup file was not created."
    assert backup_files[0].exists(), "Backup file path exists."


def test_save_routine_handles_missing_routine(populated_graph_state: GraphState):
    """
    Given: GraphState where rutina_final is None.
    When: save_routine node is executed.
    Then: state["error"] should indicate the routine is missing.
          state["step_completed"] should indicate failure.
    """
    # Arrange
    state = populated_graph_state.copy()
    state["user_id"] = "test_user"
    state["rutina_final"] = None # Explicitly set to None
    state["step_completed"] = "generate_routine_failed" # Example previous state

    # Act
    result_state = save_routine(state)

    # Assert
    assert result_state["error"] is not None and result_state["error"] != "", "Node should report an error for missing routine."
    assert "vacía" in result_state["error"].lower() or "missing" in result_state["error"].lower(), \
        "Error message should indicate routine is empty or missing."
    assert "failed" in result_state["step_completed"].lower()


@patch("builtins.open", side_effect=IOError("Simulated Permission Denied"))
@patch("shutil.copy") # Also mock copy to prevent it interfering
def test_save_routine_handles_io_error(mock_shutil_copy, mock_open, populated_graph_state: GraphState, temp_users_dir: Path):
    """
    Given: GraphState with a valid routine.
           Mocked 'open' or 'shutil.copy' to raise an IOError.
    When: save_routine node is executed.
    Then: state["error"] should capture the IOError message.
          state["step_completed"] should indicate failure.
          Attempt to restore should be logged (though mock prevents actual restore check).
    """
    # Arrange
    state = populated_graph_state.copy()
    state["user_id"] = "test_user"
    state["rutina_final"] = create_valid_routine(create_valid_principles(), user_id="test_user")
    state["step_completed"] = "routine_generated"
    # Mocks are applied via @patch decorators

    # Act
    result_state = save_routine(state)

    # Assert
    assert result_state["error"] is not None and result_state["error"] != "", "Node should report an I/O error."
    assert "error de archivo" in result_state["error"].lower() or "permission" in result_state["error"].lower(), \
        "Error message should indicate a file or permission error."
    assert "Simulated Permission Denied" in result_state["error"], "Specific mock error message not found."
    assert "failed" in result_state["step_completed"].lower()

# --- Test Handle Error Node ---

@pytest.mark.parametrize("technical_error, expected_substring", [
    ("Usuario 'xyz' no encontrado", "perfil de usuario no se encontró"),
    ("Archivo corrupto para usuario 'abc': Unterminated string", "podría estar corrupto"),
    ("Alucinación detectada: principios extraídos sin citas", "verificar la información"),
    ("Validación fallida: RIR inconsistente", "respeta el RIR recomendado"),
    ("Error de archivo guardando rutina: [Errno 13] Permission denied", "Error del sistema al intentar guardar"),
    ("Some unknown cryptic error occurred", DEFAULT_ERROR_MESSAGE), # Test default fallback
])
def test_handle_error_maps_messages(empty_graph_state: GraphState, technical_error: str, expected_substring: str):
    """
    Given: GraphState with a specific technical error message set in state["error"].
    When: handle_error node is executed.
    Then: state["respuesta_usuario"] should contain the corresponding user-friendly message.
          It should NOT contain the raw technical error details directly visible to the user.
          state["step_completed"] should be "error".
    """
    # Arrange
    state = empty_graph_state.copy()
    state["error"] = technical_error
    state["step_completed"] = "previous_step_failed" # Example previous step

    # Act
    result_state = handle_error(state)

    # Assert
    assert result_state["step_completed"] == "error", "Step completed should be set to 'error'."
    assert result_state["respuesta_usuario"] is not None and result_state["respuesta_usuario"] != "", \
        "User response should be generated."
    assert "❌" in result_state["respuesta_usuario"], "User response should start with error indicator."
    assert expected_substring.lower() in result_state["respuesta_usuario"].lower(), \
        f"Expected substring '{expected_substring}' not found in user message: '{result_state['respuesta_usuario']}'"
    # Check that technical details aren't directly exposed (unless part of the friendly message)
    if technical_error != DEFAULT_ERROR_MESSAGE and expected_substring != DEFAULT_ERROR_MESSAGE:
         assert technical_error not in result_state["respuesta_usuario"] or technical_error.lower() in expected_substring.lower(), \
             "Raw technical error seems exposed in user message."
    assert f"Referencia: paso '{state['step_completed']}'" in result_state["respuesta_usuario"], \
        "Error message should include reference to the failed step."


# --- Test All Nodes Return State ---

# Dynamically get node functions from the nodes module
try:
    from agents import nodes
    all_node_functions = [
        nodes.load_context,
        nodes.extract_principles,
        nodes.generate_routine,
        nodes.save_routine,
        nodes.handle_error,
    ]
    node_names = [f.__name__ for f in all_node_functions]
except ImportError:
    all_node_functions = []
    node_names = []
    print("Warning: Could not dynamically import node functions for parametrization.")

@pytest.mark.parametrize("node_func", all_node_functions, ids=node_names)
def test_all_nodes_return_state_struct(node_func, empty_graph_state: GraphState):
    """
    Given: Any node function from the agents.nodes module.
           An initial (potentially empty) GraphState.
    When: The node function is called with the state.
    Then: The function MUST return a dictionary (GraphState).
          The returned dictionary MUST contain essential keys like 'step_completed' and 'error'.
          The function should NOT return None or raise an unhandled exception (errors go into state).
    """
    # Arrange
    state = empty_graph_state.copy()
    # Add minimal required input for the specific node if necessary (e.g., error for handle_error)
    if node_func.__name__ == "handle_error":
        state["error"] = "Simulated error for handle_error test"

    # Act
    # We expect nodes to handle their own errors internally and modify state
    # We wrap in try-except here mainly to catch unexpected *test* failures or node bugs
    # where the node itself crashes instead of setting state['error']
    try:
         # Need to handle nodes requiring mocks or specific state setups
         if node_func.__name__ in ["extract_principles", "generate_routine", "save_routine"]:
             # These require more setup or mocks, skip for this basic structure test
             # or add specific minimal state + mocks here if feasible
             pytest.skip(f"Skipping structure test for {node_func.__name__}, requires more setup/mocks.")

         result_state = node_func(state)

    except Exception as e:
        pytest.fail(f"Node function {node_func.__name__} raised an unhandled exception: {e}")


    # Assert
    assert result_state is not None, f"{node_func.__name__} returned None instead of a GraphState dictionary."
    assert isinstance(result_state, dict), f"{node_func.__name__} did not return a dictionary."

    # Check for essential GraphState keys
    assert "step_completed" in result_state, f"{node_func.__name__} missing 'step_completed' in returned state."
    assert "error" in result_state, f"{node_func.__name__} missing 'error' in returned state."
    # Add other mandatory keys if necessary
    assert "user_id" in result_state
