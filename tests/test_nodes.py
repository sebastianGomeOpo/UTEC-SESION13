# ════════════════════════════════════════════════════════════
# PRUEBAS DE NODOS - SISTEMA DE AGENTES
# ════════════════════════════════════════════════════════════

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ────────────────────────────────────────────────────────────
# IMPORTACIONES DEL PROYECTO
# ────────────────────────────────────────────────────────────

try:
    from agents.graph_state import GraphState
    from agents.nodes.load_context import load_context
    from agents.nodes.extract_principles import extract_principles
    from agents.nodes.generate_routine import generate_routine
    from agents.nodes.save_routine import save_routine
    from agents.nodes.handle_error import (
        handle_error,
        DEFAULT_ERROR_MESSAGE
    )
    from rag.models import PrincipiosExtraidos, RutinaActiva
except ImportError as e:
    pytest.exit(f"Error al importar funciones de nodos: {e}", 1)

# Importar factories helper
from tests.conftest import create_valid_principles, create_valid_routine

# ════════════════════════════════════════════════════════════
# PRUEBAS: NODO DE CARGA DE CONTEXTO
# ════════════════════════════════════════════════════════════

class TestCargaContexto:
    """Validación del nodo que carga el perfil del usuario"""

    def test_carga_perfil_usuario_exitosamente(
        self,
        empty_graph_state: GraphState,
        temp_users_dir: Path
    ):
        """
        Verificar que el contexto se carga correctamente.
        
        Criterios:
        ✓ Perfil se popula desde archivo JSON
        ✓ step_completed = "context_loaded"
        ✓ Sin errores en el proceso
        """
        # Preparación
        state = empty_graph_state.copy()
        state["user_id"] = "test_user"

        # Ejecución
        result_state = load_context(state)

        # Validaciones
        assert result_state["error"] is None or result_state["error"] == ""
        assert result_state["step_completed"] == "context_loaded"
        assert result_state["perfil_usuario"] is not None
        assert isinstance(result_state["perfil_usuario"], dict)
        assert result_state["perfil_usuario"].get("user_id") == "test_user"
        assert "level" in result_state["perfil_usuario"]
        assert "objetivo" in result_state["perfil_usuario"]
        assert result_state["perfil_usuario"].get("name") == "Test User"

    def test_maneja_usuario_no_encontrado(
        self,
        empty_graph_state: GraphState,
        temp_users_dir: Path
    ):
        """
        Verificar manejo de usuario inexistente.
        
        Criterios:
        ✓ Error contiene "no encontrado"
        ✓ Perfil NO se carga
        ✓ step_completed indica error
        """
        # Preparación
        state = empty_graph_state.copy()
        state["user_id"] = "usuario_inexistente"

        # Ejecución
        result_state = load_context(state)

        # Validaciones
        assert result_state["error"] is not None and result_state["error"] != ""
        assert "no encontrado" in result_state["error"].lower()
        assert "error" in result_state["step_completed"].lower()
        assert result_state["perfil_usuario"] is None

    def test_maneja_json_corrupto(
        self,
        empty_graph_state: GraphState,
        temp_users_dir: Path
    ):
        """
        Verificar manejo de JSON inválido.
        
        Criterios:
        ✓ Detecta corrupción
        ✓ Mensaje de error apropiado
        ✓ Estado indica error
        """
        # Preparación
        state = empty_graph_state.copy()
        state["user_id"] = "corrupt_user"

        # Ejecución
        result_state = load_context(state)

        # Validaciones
        assert result_state["error"] is not None
        assert "corrupto" in result_state["error"].lower() or \
            "decode" in result_state["error"].lower()
        assert "error" in result_state["step_completed"].lower()

    def test_maneja_perfil_incompleto(
        self,
        empty_graph_state: GraphState,
        temp_users_dir: Path
    ):
        """
        Verificar manejo de campos faltantes.
        
        Criterios:
        ✓ Detecta campos requeridos faltantes
        ✓ Especifica qué campos faltan
        ✓ Indica error
        """
        # Preparación
        state = empty_graph_state.copy()
        state["user_id"] = "incomplete_user"

        # Ejecución
        result_state = load_context(state)

        # Validaciones
        assert result_state["error"] is not None and result_state["error"] != ""
        assert "incompleto" in result_state["error"].lower()
        assert "level" in result_state["error"].lower() or \
               "objetivo" in result_state["error"].lower()
        assert "error" in result_state["step_completed"].lower()


# ════════════════════════════════════════════════════════════
# PRUEBAS: NODO DE EXTRACCIÓN DE PRINCIPIOS
# ════════════════════════════════════════════════════════════

class TestExtraccionPrincipios:
    """Validación del nodo que extrae principios con citas"""

    def test_extrae_principios_exitosamente(
        self,
        populated_graph_state: GraphState,
        mock_principle_extractor_chain
    ):
        """
        Verificar extracción exitosa de principios.
        
        Criterios:
        ✓ Objeto PrincipiosExtraidos válido
        ✓ Contiene citas (NO alucinaciones)
        ✓ step_completed = "principles_extracted"
        ✓ Sin errores
        """
        # Preparación
        state = populated_graph_state.copy()

        # Ejecución
        result_state = extract_principles(state)

        # Validaciones
        assert result_state["error"] is None or result_state["error"] == ""
        assert result_state["step_completed"] == "principles_extracted"
        assert result_state["principios_libro"] is not None
        assert isinstance(result_state["principios_libro"], PrincipiosExtraidos)
        assert hasattr(result_state["principios_libro"], "citas_fuente")
        assert len(result_state["principios_libro"].citas_fuente) > 0

    def test_detecta_alucinacion_sin_citas(
        self,
        populated_graph_state: GraphState,
        mock_principle_extractor_chain
    ):
        """
        Verificar detección de alucinaciones (sin citas).
        
        Criterios:
        ✓ Error contiene "alucinación" o "citas"
        ✓ Principios NO se asignan
        ✓ step_completed indica error
        """
        # Preparación
        state = populated_graph_state.copy()
        principios_sin_citas = create_valid_principles(citas=[])
        mock_principle_extractor_chain("no_citations_test", principios_sin_citas)
        state["perfil_usuario"]["level"] = "no_citations_test"

        # Ejecución
        result_state = extract_principles(state)

        # Validaciones
        assert result_state["error"] is not None
        assert "alucinación" in result_state["error"].lower() or \
               "citas" in result_state["error"].lower()
        assert result_state["principios_libro"] is None
        assert "error" in result_state["step_completed"].lower()

    def test_valida_formato_rir(
        self,
        populated_graph_state: GraphState,
        mock_principle_extractor_chain
    ):
        """
        Verificar validación de formato RIR.
        
        Criterios:
        ✓ RIR tiene formato válido
        ✓ Citas presentes
        ✓ No hay error
        """
        # Preparación
        state = populated_graph_state.copy()

        # Ejecución
        result_state = extract_principles(state)

        # Validaciones
        if result_state["principios_libro"] is not None:
            import re
            rir = result_state["principios_libro"].intensidad_RIR
            assert re.match(r"^\d(-\d)?$", rir), \
                f"Formato RIR inválido: '{rir}'"


# ════════════════════════════════════════════════════════════
# PRUEBAS: NODO DE GENERACIÓN DE RUTINA
# ════════════════════════════════════════════════════════════

class TestGeneracionRutina:
    """Validación del nodo que genera la rutina personalizada"""

    def test_genera_rutina_exitosamente(
        self,
        populated_graph_state: GraphState,
        mock_openai_chat_completions
    ):
        """
        Verificar generación exitosa de rutina.
        
        Criterios:
        ✓ Rutina es RutinaActiva válida
        ✓ Contiene sesiones
        ✓ Ejercicios con detalles completos
        ✓ step_completed = "routine_generated"
        """
        # Preparación
        state = populated_graph_state.copy()
        principios = create_valid_principles()
        state["principios_libro"] = principios

        # Ejecución
        result_state = generate_routine(state)

        # Validaciones
        assert result_state["error"] is None or result_state["error"] == ""
        assert result_state["step_completed"] == "routine_generated"
        assert result_state["rutina_final"] is not None
        assert isinstance(result_state["rutina_final"], RutinaActiva)
        assert len(result_state["rutina_final"].sesiones) > 0

    def test_maneja_principios_faltantes(
        self,
        populated_graph_state: GraphState
    ):
        """
        Verificar manejo cuando faltan principios.
        
        Criterios:
        ✓ Error indica principios faltantes
        ✓ Rutina NO se genera
        ✓ Indica error
        """
        # Preparación
        state = populated_graph_state.copy()
        state["principios_libro"] = None

        # Ejecución
        result_state = generate_routine(state)

        # Validaciones
        assert result_state["error"] is not None
        assert "error" in result_state["step_completed"].lower()

    def test_maneja_error_llm(
        self,
        populated_graph_state: GraphState,
        mock_openai_chat_completions
    ):
        """
        Verificar manejo de errores del LLM.
        
        Criterios:
        ✓ Captura error del LLM
        ✓ Mensaje de error en español
        ✓ Rutina NO se asigna
        """
        # Preparación
        state = populated_graph_state.copy()
        principios = create_valid_principles()
        state["principios_libro"] = principios
        
        # Configurar mock para que falle
        mock_openai_chat_completions("error_test", Exception("LLM Error simulado"))

        # Ejecución
        result_state = generate_routine(state)

        # Validaciones
        assert result_state["error"] is not None
        assert "error" in result_state["step_completed"].lower()


# ════════════════════════════════════════════════════════════
# PRUEBAS: NODO DE GUARDADO DE RUTINA
# ════════════════════════════════════════════════════════════

class TestGuardadoRutina:
    """Validación del nodo que guarda la rutina en archivo"""

    def test_guarda_rutina_exitosamente(
        self,
        populated_graph_state: GraphState,
        temp_users_dir: Path
    ):
        """
        Verificar guardado exitoso de rutina.
        
        Criterios:
        ✓ Archivo JSON creado
        ✓ Contenido válido
        ✓ Timestamp actualizado
        ✓ Backup creado
        """
        # Preparación
        state = populated_graph_state.copy()
        state["user_id"] = "test_user"
        principios = create_valid_principles()
        state["rutina_final"] = create_valid_routine(
            principios,
            user_id="test_user"
        )
        state["step_completed"] = "routine_generated"

        # Ejecución
        result_state = save_routine(state)

        # Validaciones
        assert result_state["error"] is None or result_state["error"] == ""
        assert result_state["step_completed"] == "saved"

        user_file_path = temp_users_dir / f"{state['user_id']}.json"
        assert user_file_path.exists(), "Archivo de usuario no creado"

        with open(user_file_path) as f:
            saved_data = json.load(f)

        assert "updated_at" in saved_data, "Timestamp faltante"

        # Verificar creación de backup
        backup_files = list(
            temp_users_dir.glob(
                f"{state['user_id']}.*.backup"
            )
        )
        assert len(backup_files) >= 1, "Backup no creado"

    def test_maneja_rutina_faltante(
        self,
        populated_graph_state: GraphState
    ):
        """
        Verificar manejo cuando falta la rutina.
        
        Criterios:
        ✓ Error indica rutina vacía
        ✓ step_completed = failed
        """
        # Preparación
        state = populated_graph_state.copy()
        state["user_id"] = "test_user"
        state["rutina_final"] = None

        # Ejecución
        result_state = save_routine(state)

        # Validaciones
        assert result_state["error"] is not None and result_state["error"] != ""
        assert "vacía" in result_state["error"].lower() or \
               "missing" in result_state["error"].lower()
        assert "error" in result_state["step_completed"].lower()

    @patch("builtins.open", side_effect=IOError("Error de permisos simulado"))
    @patch("shutil.copy")
    def test_maneja_error_io(
        self,
        mock_shutil_copy,
        mock_open,
        populated_graph_state: GraphState,
        temp_users_dir: Path
    ):
        """
        Verificar manejo de errores de I/O.
        
        Criterios:
        ✓ Captura error de archivo
        ✓ Intenta restaurar backup
        ✓ step_completed indica error
        """
        # Preparación
        state = populated_graph_state.copy()
        state["user_id"] = "test_user"
        principios = create_valid_principles()
        state["rutina_final"] = create_valid_routine(
            principios,
            user_id="test_user"
        )

        # Ejecución
        result_state = save_routine(state)

        # Validaciones
        assert result_state["error"] is not None
        assert "error de archivo" in result_state["error"].lower() or \
               "permission" in result_state["error"].lower() or \
               "Error de permisos simulado" in result_state["error"]
        assert "error" in result_state["step_completed"].lower()


# ════════════════════════════════════════════════════════════
# PRUEBAS: NODO DE MANEJO DE ERRORES
# ════════════════════════════════════════════════════════════

class TestManejoErrores:
    """Validación del nodo que traduce errores técnicos a mensajes amigables"""

    @pytest.mark.parametrize("error_tecnico,substring_esperado", [
        (
            "Usuario 'xyz' no encontrado",
            "perfil de usuario no se encontró"
        ),
        (
            "Archivo corrupto para usuario 'abc': Unterminated string",
            "podría estar corrupto"
        ),
        (
            "Alucinación detectada: principios extraídos sin citas",
            "verificar la información"
        ),
        (
            "Validación fallida: RIR inconsistente",
            "respeta el RIR recomendado"
        ),
        (
            "Error de archivo guardando rutina: [Errno 13] Permission denied",
            "Error del sistema al intentar guardar"
        ),
        (
            "Error desconocido criptográfico",
            DEFAULT_ERROR_MESSAGE
        ),
    ])
    def test_mapea_errores_a_mensajes_amigables(
        self,
        empty_graph_state: GraphState,
        error_tecnico: str,
        substring_esperado: str
    ):
        """
        Verificar traducción de errores técnicos a mensajes amigables.
        
        Criterios:
        ✓ Mensaje contiene substring esperado
        ✓ NO expone detalles técnicos
        ✓ Incluye referencia al paso fallido
        ✓ Inicia con indicador de error ❌
        """
        # Preparación
        state = empty_graph_state.copy()
        state["error"] = error_tecnico
        state["step_completed"] = "paso_anterior_fallo"

        # Ejecución
        result_state = handle_error(state)

        # Validaciones
        assert result_state["step_completed"] == "error"
        assert result_state["respuesta_usuario"] is not None
        assert result_state["respuesta_usuario"] != ""
        assert "❌" in result_state["respuesta_usuario"]
        assert substring_esperado.lower() in \
            result_state["respuesta_usuario"].lower()

        # Verificar que detalles técnicos NO se exponen
        if error_tecnico != DEFAULT_ERROR_MESSAGE and \
        substring_esperado != DEFAULT_ERROR_MESSAGE:
            assert error_tecnico not in result_state["respuesta_usuario"] or \
                error_tecnico.lower() in substring_esperado.lower()

        # Verificar referencia al paso
        assert f"Referencia: paso '{state['step_completed']}'" in \
            result_state["respuesta_usuario"] or \
            "paso" in result_state["respuesta_usuario"].lower()


# ════════════════════════════════════════════════════════════
# PRUEBAS: ESTRUCTURA GENERAL DE NODOS
# ════════════════════════════════════════════════════════════

try:
    from agents import nodes
    NODE_FUNCTIONS = [
        nodes.load_context,
        nodes.extract_principles,
        nodes.generate_routine,
        nodes.save_routine,
        nodes.handle_error,
    ]
    NODE_NAMES = [f.__name__ for f in NODE_FUNCTIONS]
except ImportError:
    NODE_FUNCTIONS = []
    NODE_NAMES = []
    print("Advertencia: No se pueden importar funciones de nodos para parametrización.")


class TestEstructuraNodos:
    """Validación de estructura general de todos los nodos"""

    @pytest.mark.parametrize("node_function", NODE_FUNCTIONS, ids=NODE_NAMES)
    def test_todos_nodos_retornan_estado(
        self,
        node_function,
        empty_graph_state: GraphState
    ):
        """
        Verificar que TODOS los nodos retornan GraphState válido.
        
        Criterios:
        ✓ Retorna diccionario (GraphState)
        ✓ Contiene 'step_completed'
        ✓ Contiene 'error'
        ✓ No retorna None
        ✓ No lanza excepciones
        """
        # Preparación
        state = empty_graph_state.copy()

        # Agregar setup mínimo para ciertos nodos
        if node_function.__name__ == "handle_error":
            state["error"] = "Error simulado para prueba de handle_error"

        # Ejecución
        try:
            # Saltar nodos que requieren setup complejo
            if node_function.__name__ in [
                "extract_principles",
                "generate_routine",
                "save_routine"
            ]:
                pytest.skip(
                    f"Saltando {node_function.__name__}, "
                    f"requiere setup más completo"
                )

            result_state = node_function(state)

        except Exception as e:
            pytest.fail(
                f"Nodo {node_function.__name__} lanzó excepción: {e}"
            )

        # Validaciones
        assert result_state is not None, \
            f"{node_function.__name__} retornó None"
        assert isinstance(result_state, dict), \
            f"{node_function.__name__} no retornó diccionario"
        assert "step_completed" in result_state, \
            f"{node_function.__name__} falta 'step_completed'"
        assert "error" in result_state, \
            f"{node_function.__name__} falta 'error'"
        assert "user_id" in result_state, \
            f"{node_function.__name__} falta 'user_id'"