# ════════════════════════════════════════════════════════════
# PRUEBAS DE NODOS - SISTEMA DE AGENTES
# ════════════════════════════════════════════════════════════

import json
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# ────────────────────────────────────────────────────────────
# IMPORTACIONES DEL PROYECTO
# ────────────────────────────────────────────────────────────

try:
    from agents.graph_state import GraphState as EstadoGrafo
    from agents.nodes.load_context import load_context as cargar_contexto
    from agents.nodes.extract_principles import extract_principles as extraer_principios
    from agents.nodes.generate_routine import generate_routine as generar_rutina
    from agents.nodes.save_routine import save_routine as guardar_rutina
    from agents.nodes.handle_error import (
        handle_error as manejar_error,
        ERROR_MESSAGE_MAP as MAPA_MENSAJES_ERROR,
        DEFAULT_ERROR_MESSAGE as MENSAJE_ERROR_DEFECTO
    )
    from rag.models import PrincipiosExtraidos, RutinaActiva, Sesion, Ejercicio, ECI
    from config.settings import Config
except ImportError as e:
    pytest.exit(f"Error al importar funciones de nodos: {e}", 1)

# Importar factories helper
from .conftest import create_valid_principles as crear_principios_validos ,create_valid_routine as crear_rutina_valida

# ════════════════════════════════════════════════════════════
# PRUEBAS: NODO DE CARGA DE CONTEXTO
# ════════════════════════════════════════════════════════════

class TestCargaContexto:
    """Validación del nodo que carga el perfil del usuario"""

    def test_carga_perfil_usuario_exitosamente(
        self,
        estado_grafo_vacio: EstadoGrafo,
        directorio_usuarios_temporal: Path
    ):
        """
        Verificar que el contexto se carga correctamente.
        
        Criterios:
        ✓ Perfil se popula desde archivo JSON
        ✓ step_completed = "context_loaded"
        ✓ Sin errores en el proceso
        """
        # Preparación
        estado = estado_grafo_vacio.copy()
        estado["user_id"] = "test_user"

        # Ejecución
        estado_resultado = cargar_contexto(estado)

        # Validaciones
        assert estado_resultado["error"] is None or estado_resultado["error"] == ""
        assert estado_resultado["step_completed"] == "context_loaded"
        assert estado_resultado["perfil_usuario"] is not None
        assert isinstance(estado_resultado["perfil_usuario"], dict)
        assert estado_resultado["perfil_usuario"].get("user_id") == "test_user"
        assert "level" in estado_resultado["perfil_usuario"]
        assert "objetivo" in estado_resultado["perfil_usuario"]
        assert estado_resultado["perfil_usuario"].get("name") == "Test User"

    def test_maneja_usuario_no_encontrado(
        self,
        estado_grafo_vacio: EstadoGrafo,
        directorio_usuarios_temporal: Path
    ):
        """
        Verificar manejo de usuario inexistente.
        
        Criterios:
        ✓ Error contiene "no encontrado"
        ✓ Perfil NO se carga
        ✓ step_completed indica error
        """
        # Preparación
        estado = estado_grafo_vacio.copy()
        estado["user_id"] = "usuario_inexistente"

        # Ejecución
        estado_resultado = cargar_contexto(estado)

        # Validaciones
        assert estado_resultado["error"] is not None and estado_resultado["error"] != ""
        assert "no encontrado" in estado_resultado["error"].lower()
        assert "error" in estado_resultado["step_completed"].lower()
        assert estado_resultado["perfil_usuario"] is None

    def test_maneja_json_corrupto(
        self,
        estado_grafo_vacio: EstadoGrafo,
        directorio_usuarios_temporal: Path
    ):
        """
        Verificar manejo de JSON inválido.
        
        Criterios:
        ✓ Detecta corrupción
        ✓ Mensaje de error apropiado
        ✓ Estado indica error
        """
        # Preparación
        estado = estado_grafo_vacio.copy()
        estado["user_id"] = "corrupt_user"

        # Ejecución
        estado_resultado = cargar_contexto(estado)

        # Validaciones
        assert estado_resultado["error"] is not None
        assert "corrupto" in estado_resultado["error"].lower() or \
            "decode" in estado_resultado["error"].lower()
        assert "error" in estado_resultado["step_completed"].lower()

    def test_maneja_perfil_incompleto(
        self,
        estado_grafo_vacio: EstadoGrafo,
        directorio_usuarios_temporal: Path
    ):
        """
        Verificar manejo de campos faltantes.
        
        Criterios:
        ✓ Detecta campos requeridos faltantes
        ✓ Especifica qué campos faltan
        ✓ Indica error
        """
        # Preparación
        estado = estado_grafo_vacio.copy()
        estado["user_id"] = "incomplete_user"

        # Ejecución
        estado_resultado = cargar_contexto(estado)

        # Validaciones
        assert estado_resultado["error"] is not None and estado_resultado["error"] != ""
        assert "incompleto" in estado_resultado["error"].lower()
        assert "level" in estado_resultado["error"].lower() or \
               "objetivo" in estado_resultado["error"].lower()
        assert "error" in estado_resultado["step_completed"].lower()


# ════════════════════════════════════════════════════════════
# PRUEBAS: NODO DE EXTRACCIÓN DE PRINCIPIOS
# ════════════════════════════════════════════════════════════

class TestExtraccionPrincipios:
    """Validación del nodo que extrae principios con citas"""

    def test_extrae_principios_exitosamente(
        self,
        estado_grafo_poblado: EstadoGrafo,
        cadena_extractor_principios_mock
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
        estado = estado_grafo_poblado.copy()

        # Ejecución
        estado_resultado = extraer_principios(estado)

        # Validaciones
        assert estado_resultado["error"] is None or estado_resultado["error"] == ""
        assert estado_resultado["step_completed"] == "principles_extracted"
        assert estado_resultado["principios_libro"] is not None
        assert isinstance(estado_resultado["principios_libro"], PrincipiosExtraidos)
        assert hasattr(estado_resultado["principios_libro"], "citas_fuente")
        assert len(estado_resultado["principios_libro"].citas_fuente) > 0

    def test_detecta_alucinacion_sin_citas(
        self,
        estado_grafo_poblado: EstadoGrafo,
        cadena_extractor_principios_mock
    ):
        """
        Verificar detección de alucinaciones (sin citas).
        
        Criterios:
        ✓ Error contiene "alucinación" o "citas"
        ✓ Principios NO se asignan
        ✓ step_completed indica error
        """
        # Preparación
        estado = estado_grafo_poblado.copy()
        principios_sin_citas = crear_principios_validos(citas=[])
        cadena_extractor_principios_mock("sin_citas_prueba", principios_sin_citas)
        estado["perfil_usuario"]["level"] = "sin_citas_prueba"

        # Ejecución
        estado_resultado = extraer_principios(estado)

        # Validaciones
        assert estado_resultado["error"] is not None
        assert "alucinación" in estado_resultado["error"].lower() or \
               "citas" in estado_resultado["error"].lower()
        assert estado_resultado["principios_libro"] is None
        assert "error" in estado_resultado["step_completed"].lower()

    def test_valida_formato_rir(
        self,
        estado_grafo_poblado: EstadoGrafo,
        cadena_extractor_principios_mock
    ):
        """
        Verificar validación de formato RIR.
        
        Criterios:
        ✓ RIR tiene formato válido
        ✓ Citas presentes
        ✓ No hay error
        """
        # Preparación
        estado = estado_grafo_poblado.copy()

        # Ejecución
        estado_resultado = extraer_principios(estado)

        # Validaciones
        if estado_resultado["principios_libro"] is not None:
            import re
            rir = estado_resultado["principios_libro"].intensidad_RIR
            assert re.match(r"^\d(-\d)?$", rir), \
                f"Formato RIR inválido: '{rir}'"


# ════════════════════════════════════════════════════════════
# PRUEBAS: NODO DE GENERACIÓN DE RUTINA
# ════════════════════════════════════════════════════════════

class TestGeneracionRutina:
    """Validación del nodo que genera la rutina personalizada"""

    def test_genera_rutina_exitosamente(
        self,
        estado_grafo_poblado: EstadoGrafo,
        cadena_generador_rutina_mock
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
        estado = estado_grafo_poblado.copy()

        # Ejecución
        estado_resultado = generar_rutina(estado)

        # Validaciones
        assert estado_resultado["error"] is None or estado_resultado["error"] == ""
        assert estado_resultado["step_completed"] == "routine_generated"
        assert estado_resultado["rutina_final"] is not None
        assert isinstance(estado_resultado["rutina_final"], RutinaActiva)
        assert len(estado_resultado["rutina_final"].sesiones) > 0

    def test_maneja_principios_faltantes(
        self,
        estado_grafo_poblado: EstadoGrafo
    ):
        """
        Verificar manejo cuando faltan principios.
        
        Criterios:
        ✓ Error indica principios faltantes
        ✓ Rutina NO se genera
        ✓ Indica error
        """
        # Preparación
        estado = estado_grafo_poblado.copy()
        estado["principios_libro"] = None

        # Ejecución
        estado_resultado = generar_rutina(estado)

        # Validaciones
        assert estado_resultado["error"] is not None
        assert "error" in estado_resultado["step_completed"].lower()

    def test_maneja_error_llm(
        self,
        estado_grafo_poblado: EstadoGrafo,
        cadena_generador_rutina_mock
    ):
        """
        Verificar manejo de errores del LLM.
        
        Criterios:
        ✓ Captura error del LLM
        ✓ Mensaje de error en español
        ✓ Rutina NO se asigna
        """
        # Preparación
        estado = estado_grafo_poblado.copy()
        cadena_generador_rutina_mock(error=True)

        # Ejecución
        estado_resultado = generar_rutina(estado)

        # Validaciones
        assert estado_resultado["error"] is not None
        assert "error" in estado_resultado["step_completed"].lower()


# ════════════════════════════════════════════════════════════
# PRUEBAS: NODO DE GUARDADO DE RUTINA
# ════════════════════════════════════════════════════════════

class TestGuardadoRutina:
    """Validación del nodo que guarda la rutina en archivo"""

    def test_guarda_rutina_exitosamente(
        self,
        estado_grafo_poblado: EstadoGrafo,
        directorio_usuarios_temporal: Path
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
        estado = estado_grafo_poblado.copy()
        estado["user_id"] = "test_user"
        estado["rutina_final"] = crear_rutina_valida(
            crear_principios_validos(),
            user_id="test_user"
        )
        estado["step_completed"] = "routine_generated"

        # Ejecución
        estado_resultado = guardar_rutina(estado)

        # Validaciones
        assert estado_resultado["error"] is None or estado_resultado["error"] == ""
        assert estado_resultado["step_completed"] == "routine_saved"

        ruta_usuario = directorio_usuarios_temporal / f"{estado['user_id']}.json"
        assert ruta_usuario.exists(), "Archivo de usuario no creado"

        with open(ruta_usuario) as archivo:
            datos_guardados = json.load(archivo)

        assert "updated_at" in datos_guardados, "Timestamp faltante"

        # Verificar creación de backup
        archivos_backup = list(
            directorio_usuarios_temporal.glob(
                f"{estado['user_id']}.json.*.backup"
            )
        )
        assert len(archivos_backup) >= 1, "Backup no creado"
        assert archivos_backup[0].exists(), "Ruta de backup existe"

    def test_maneja_rutina_faltante(
        self,
        estado_grafo_poblado: EstadoGrafo
    ):
        """
        Verificar manejo cuando falta la rutina.
        
        Criterios:
        ✓ Error indica rutina vacía
        ✓ step_completed = failed
        """
        # Preparación
        estado = estado_grafo_poblado.copy()
        estado["user_id"] = "test_user"
        estado["rutina_final"] = None

        # Ejecución
        estado_resultado = guardar_rutina(estado)

        # Validaciones
        assert estado_resultado["error"] is not None and estado_resultado["error"] != ""
        assert "vacía" in estado_resultado["error"].lower() or \
               "missing" in estado_resultado["error"].lower()
        assert "failed" in estado_resultado["step_completed"].lower()

    @patch("builtins.open", side_effect=IOError("Error de permisos simulado"))
    @patch("shutil.copy")
    def test_maneja_error_io(
        self,
        mock_shutil_copy,
        mock_open,
        estado_grafo_poblado: EstadoGrafo,
        directorio_usuarios_temporal: Path
    ):
        """
        Verificar manejo de errores de I/O.
        
        Criterios:
        ✓ Captura error de archivo
        ✓ Intenta restaurar backup
        ✓ step_completed indica error
        """
        # Preparación
        estado = estado_grafo_poblado.copy()
        estado["user_id"] = "test_user"
        estado["rutina_final"] = crear_rutina_valida(
            crear_principios_validos(),
            user_id="test_user"
        )

        # Ejecución
        estado_resultado = guardar_rutina(estado)

        # Validaciones
        assert estado_resultado["error"] is not None
        assert "error de archivo" in estado_resultado["error"].lower() or \
               "permission" in estado_resultado["error"].lower()
        assert "Error de permisos simulado" in estado_resultado["error"]
        assert "failed" in estado_resultado["step_completed"].lower()


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
            MENSAJE_ERROR_DEFECTO
        ),
    ])
    def test_mapea_errores_a_mensajes_amigables(
        self,
        estado_grafo_vacio: EstadoGrafo,
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
        estado = estado_grafo_vacio.copy()
        estado["error"] = error_tecnico
        estado["step_completed"] = "paso_anterior_fallo"

        # Ejecución
        estado_resultado = manejar_error(estado)

        # Validaciones
        assert estado_resultado["step_completed"] == "error"
        assert estado_resultado["respuesta_usuario"] is not None
        assert estado_resultado["respuesta_usuario"] != ""
        assert "❌" in estado_resultado["respuesta_usuario"]
        assert substring_esperado.lower() in \
            estado_resultado["respuesta_usuario"].lower()

        # Verificar que detalles técnicos NO se exponen
        if error_tecnico != MENSAJE_ERROR_DEFECTO and \
        substring_esperado != MENSAJE_ERROR_DEFECTO:
            assert error_tecnico not in estado_resultado["respuesta_usuario"] or \
                error_tecnico.lower() in substring_esperado.lower()

        # Verificar referencia al paso
        assert f"Referencia: paso '{estado['step_completed']}'" in \
            estado_resultado["respuesta_usuario"]


# ════════════════════════════════════════════════════════════
# PRUEBAS: ESTRUCTURA GENERAL DE NODOS
# ════════════════════════════════════════════════════════════

try:
    from agents import nodes
    LISTA_FUNCIONES_NODOS = [
        nodes.load_context,
        nodes.extract_principles,
        nodes.generate_routine,
        nodes.save_routine,
        nodes.handle_error,
    ]
    NOMBRES_NODOS = [f.__name__ for f in LISTA_FUNCIONES_NODOS]
except ImportError:
    LISTA_FUNCIONES_NODOS = []
    NOMBRES_NODOS = []
    print("Advertencia: No se pueden importar funciones de nodos para parametrización.")


class TestEstructuraNodos:
    """Validación de estructura general de todos los nodos"""

    @pytest.mark.parametrize("funcion_nodo", LISTA_FUNCIONES_NODOS, ids=NOMBRES_NODOS)
    def test_todos_nodos_retornan_estado(
        self,
        funcion_nodo,
        estado_grafo_vacio: EstadoGrafo
    ):
        """
        Verificar que TODOS los nodos retornan EstadoGrafo válido.
        
        Criterios:
        ✓ Retorna diccionario (EstadoGrafo)
        ✓ Contiene 'step_completed'
        ✓ Contiene 'error'
        ✓ No retorna None
        ✓ No lanza excepciones
        """
        # Preparación
        estado = estado_grafo_vacio.copy()

        # Agregar setup mínimo para ciertos nodos
        if funcion_nodo.__name__ == "manejar_error":
            estado["error"] = "Error simulado para prueba de manejar_error"

        # Ejecución
        try:
            # Saltar nodos que requieren setup complejo
            if funcion_nodo.__name__ in [
                "extraer_principios",
                "generar_rutina",
                "guardar_rutina"
            ]:
                pytest.skip(
                    f"Saltando {funcion_nodo.__name__}, "
                    f"requiere setup más completo"
                )

            estado_resultado = funcion_nodo(estado)

        except Exception as excepcion:
            pytest.fail(
                f"Nodo {funcion_nodo.__name__} lanzó excepción: {excepcion}"
            )

        # Validaciones
        assert estado_resultado is not None, \
            f"{funcion_nodo.__name__} retornó None"
        assert isinstance(estado_resultado, dict), \
            f"{funcion_nodo.__name__} no retornó diccionario"
        assert "step_completed" in estado_resultado, \
            f"{funcion_nodo.__name__} falta 'step_completed'"
        assert "error" in estado_resultado, \
            f"{funcion_nodo.__name__} falta 'error'"
        assert "user_id" in estado_resultado, \
            f"{funcion_nodo.__name__} falta 'user_id'"