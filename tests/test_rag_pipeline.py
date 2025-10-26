# ========================================
# PRUEBAS DE INTEGRACIÓN - SISTEMA RAG
# ========================================

# Importaciones de la librería estándar
import os
import re
from typing import List, Dict, Any

# Importaciones de terceros
import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# Importaciones del proyecto
try:
    from rag.vectorstore_manager import VectorStoreManager
    from rag.principle_extractor import PrincipleExtractor
    from rag.models import PrincipiosExtraidos
    from config.settings import Config
except ImportError as e:
    pytest.exit(f"Error al importar módulos RAG: {e}", 1)


# ========================================
# CONFIGURACIÓN GENERAL DE PRUEBAS
# ========================================

SALTAR_SI_BD_AUSENTE = False  # Asume que la BD Chroma existe o se crea en fixture

pytestmark = pytest.mark.skipif(
    SALTAR_SI_BD_AUSENTE,
    reason="Base de datos Chroma no encontrada"
)


# ========================================
# PRUEBAS DE VECTORSTORE (Base de Datos)
# ========================================

class TestVectorStore:
    """Validación de la tienda vectorial (Chroma)"""

    def test_vectorstore_carga_correctamente(self, vectorstore):
        """
        Verificar que el vectorstore se carga e inicializa correctamente.
        
        Requisitos:
        - Debe ser una instancia válida de Chroma (no None)
        - Debe tener métodos esenciales como similarity_search
        """
        # Verificaciones
        assert vectorstore is not None, "El vectorstore no se inicializó correctamente"
        assert hasattr(vectorstore, 'similarity_search'), \
            "Falta el método similarity_search"
        assert hasattr(vectorstore, 'as_retriever'), \
            "No puede crear un retriever"

    def test_retriever_retorna_documentos(self, vectorstore):
        """
        Verificar que el retriever devuelve documentos válidos.
        
        Criterios:
        - Retorna una lista de documentos
        - Cada documento tiene page_content válido
        - Incluye metadatos completos
        """
        # Preparación
        retriever: VectorStoreRetriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Top 3 resultados
        )
        consulta = "¿Cuál es el RIR recomendado para hipertrofia en intermedios?"

        # Ejecución
        resultados: List[Document] = retriever.invoke(consulta)

        # Validaciones
        assert isinstance(resultados, list), \
            "El retriever no retornó una lista"
        assert len(resultados) > 0, \
            f"No hay documentos para: '{consulta}'"

        # Validar primer documento
        primer_doc = resultados[0]
        assert isinstance(primer_doc, Document), \
            "El resultado no es un documento válido"
        assert hasattr(primer_doc, 'page_content'), \
            "Falta el contenido del documento"
        assert isinstance(primer_doc.page_content, str), \
            "El contenido no es string"
        assert len(primer_doc.page_content) > 10, \
            "El contenido es muy corto"
        assert hasattr(primer_doc, 'metadata'), \
            "Falta los metadatos"
        assert isinstance(primer_doc.metadata, dict), \
            "Los metadatos no son un diccionario"


# ========================================
# PRUEBAS DEL EXTRACTOR DE PRINCIPIOS
# ========================================

class TestExtractorPrincipios:
    """Validación de la extracción de principios con citas"""

    def test_cadena_extrae_principios_con_citas(
        self,
        principle_extractor: PrincipleExtractor,
        user_profile_sample: Dict[str, Any]
    ):
        """
        Verificar que la cadena de extracción retorna principios con citas.
        
        Criterios Críticos:
        - Retorna un objeto PrincipiosExtraidos válido
        - DEBE contener citas de fuente (no alucinaciones)
        - Las citas tienen formato válido (páginas/referencias)
        - Campos esenciales están poblados
        """
        # Preparación
        cadena = principle_extractor.get_extraction_chain()

        # Ejecución
        resultado: PrincipiosExtraidos = cadena.invoke(user_profile_sample)

        # Validaciones básicas
        assert resultado is not None, \
            "La cadena retornó None"
        assert isinstance(resultado, PrincipiosExtraidos), \
            f"Tipo incorrecto: {type(resultado)}"

        # Validar citas (CRÍTICO: prevenir alucinaciones)
        self._validar_citas_fuente(resultado)

        # Validar intensidad RIR
        self._validar_intensidad_rir(resultado)

        # Validar rango de repeticiones
        assert hasattr(resultado, 'rango_repeticiones'), \
            "Falta rango_repeticiones"
        assert resultado.rango_repeticiones is not None, \
            "rango_repeticiones está vacío"

    def test_estructura_principios_completa(
        self,
        principle_extractor: PrincipleExtractor,
        user_profile_sample: Dict[str, Any]
    ):
        """
        Verificar que todos los campos del modelo están presentes.
        
        Valida:
        - Todos los campos esperados existen
        - Los tipos de datos coinciden con el modelo
        """
        # Preparación
        cadena = principle_extractor.get_extraction_chain()

        # Ejecución
        resultado: PrincipiosExtraidos = cadena.invoke(user_profile_sample)

        # Validación de estructura
        assert isinstance(resultado, PrincipiosExtraidos), \
            "No es un PrincipiosExtraidos"

        campos_esperados = PrincipiosExtraidos.model_fields.keys()

        # Verificar que todos los campos existen
        for campo in campos_esperados:
            assert hasattr(resultado, campo), \
                f"Falta el campo '{campo}'"

        # Validar tipos específicos
        assert isinstance(resultado.intensidad_RIR, str), \
            "intensidad_RIR debe ser string"
        assert isinstance(resultado.descanso_series_s, int), \
            "descanso_series_s debe ser int"
        assert isinstance(resultado.ECI_recomendados, list), \
            "ECI_recomendados debe ser lista"
        assert isinstance(resultado.citas_fuente, list), \
            "citas_fuente debe ser lista"

    @pytest.mark.parametrize("nivel,patron_rir_esperado", [
        ("principiante", r"^[2-4](-[3-5])?$"),  # RIR más alto para principiantes
        ("intermedio", r"^[1-3](-[2-4])?$"),    # RIR medio
        ("avanzado", r"^[0-2](-[1-3])?$"),      # RIR más bajo para avanzados
    ])
    def test_adaptacion_segun_nivel(
        self,
        principle_extractor: PrincipleExtractor,
        user_profile_sample: Dict[str, Any],
        nivel: str,
        patron_rir_esperado: str
    ):
        """
        Verificar que la extracción se adapta al nivel del usuario.
        
        Expectativas por Nivel:
        - Principiante: RIR más alto (entrenamientos más cortos)
        - Intermedio: RIR medio
        - Avanzado: RIR más bajo (entrenamientos más largos)
        """
        # Preparación
        perfil = user_profile_sample.copy()
        perfil["level"] = nivel
        cadena = principle_extractor.get_extraction_chain()

        # Ejecución
        resultado: PrincipiosExtraidos = cadena.invoke(perfil)

        # Validaciones
        assert resultado is not None, \
            f"Resultado None para nivel '{nivel}'"
        assert isinstance(resultado, PrincipiosExtraidos), \
            "Tipo de resultado incorrecto"
        assert len(resultado.citas_fuente) > 0, \
            f"Sin citas para nivel '{nivel}' - posible alucinación"

        # Validar que el RIR coincide con el patrón esperado
        assert re.match(patron_rir_esperado, resultado.intensidad_RIR), \
            f"Nivel '{nivel}': esperaba RIR '{patron_rir_esperado}', " \
            f"obtuvo '{resultado.intensidad_RIR}'"

    # ========================================
    # MÉTODOS AUXILIARES DE VALIDACIÓN
    # ========================================

    @staticmethod
    def _validar_citas_fuente(resultado: PrincipiosExtraidos) -> None:
        """
        Validar que las citas tienen formato válido y no son alucinaciones.
        
        Lanza:
        - AssertionError si no hay citas o formato inválido
        """
        assert hasattr(resultado, 'citas_fuente'), \
            "Falta atributo citas_fuente"
        assert isinstance(resultado.citas_fuente, list), \
            "citas_fuente no es lista"
        assert len(resultado.citas_fuente) > 0, \
            "CRÍTICO: Sin citas de fuente - posible alucinación del LLM"

        # Validar formato de citas (debe contener 'Página' o números)
        citaciones_validas = any(
            "ágina" in cita or any(char.isdigit() for char in cita)
            for cita in resultado.citas_fuente
        )
        assert citaciones_validas, \
            f"Formato de citas inválido: {resultado.citas_fuente}"

    @staticmethod
    def _validar_intensidad_rir(resultado: PrincipiosExtraidos) -> None:
        """
        Validar que el RIR tiene formato y valores válidos.
        
        Formato Válido: "2" o "2-3" (números de 0-5)
        
        Lanza:
        - AssertionError si el formato no es válido
        """
        assert hasattr(resultado, 'intensidad_RIR'), \
            "Falta intensidad_RIR"
        assert isinstance(resultado.intensidad_RIR, str), \
            "intensidad_RIR debe ser string"
        assert len(resultado.intensidad_RIR) > 0, \
            "intensidad_RIR está vacío"

        # Patrón: dígito único o rango (ej: "2" o "2-3")
        es_formato_valido = re.match(
            r"^\d(-\d)?$",
            resultado.intensidad_RIR
        )
        assert es_formato_valido, \
            f"Formato RIR inválido: '{resultado.intensidad_RIR}' " \
            f"(esperado: '2' o '2-3')"