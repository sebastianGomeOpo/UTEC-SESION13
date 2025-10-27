from pathlib import Path
from typing import Dict, Any

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableLambda

# Project Imports
from config.settings import Config
from rag.models import PrincipiosExtraidos
from rag.vectorstore_manager import VectorStoreManager
from utils.logger import setup_logger


class PrincipleExtractor:
    """
    Handles the RAG chain creation for extracting training principles based on user profile.
    """

    def __init__(self):
        """
        Initializes the PrincipleExtractor.
        """
        self.logger = setup_logger(__name__)
        self.config = Config()
        self.llm = ChatOpenAI(
            model=self.config.LLM_MODEL_EXTRACT,
            temperature=self.config.LLM_TEMPERATURE
        )
        self.vector_store_manager = VectorStoreManager()
        self.parser = PydanticOutputParser(pydantic_object=PrincipiosExtraidos)
        self.logger.info("PrincipleExtractor initialized.")

    def _load_prompt_template(self) -> str:
        """
        Loads the prompt template text from the specified file.

        Returns:
            str: The raw prompt template string.

        Raises:
            FileNotFoundError: If the prompt file does not exist.
        """
        prompt_file_path = self.config.PROMPTS_DIR / "rag_principle_extractor.txt"
        self.logger.info(f"Loading prompt template from: {prompt_file_path}")
        if not prompt_file_path.exists():
            self.logger.error(f"Prompt file not found: {prompt_file_path}")
            raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")

        try:
            raw_prompt = prompt_file_path.read_text(encoding="utf-8")
            self.logger.info("Prompt template loaded successfully.")
            return raw_prompt
        except Exception as e:
            self.logger.error(f"Error reading prompt file {prompt_file_path}: {e}", exc_info=True)
            raise

    def _build_retrieval_query(self, perfil_usuario: Dict[str, Any]) -> str:
        """
        Construye una query textual para el retriever basándose en el perfil del usuario.
        
        Args:
            perfil_usuario: Diccionario con datos del usuario
            
        Returns:
            str: Query optimizada para búsqueda semántica
        """
        # Extraer campos relevantes
        nivel = perfil_usuario.get("level", "")
        objetivo = perfil_usuario.get("objetivo", "")
        restricciones = perfil_usuario.get("restricciones", [])
        
        # Construir query
        query_parts = []
        
        if nivel:
            query_parts.append(f"nivel {nivel}")
        
        if objetivo:
            query_parts.append(f"objetivo {objetivo}")
        
        if restricciones:
            restricciones_str = " ".join(restricciones)
            query_parts.append(f"restricciones {restricciones_str}")
        
        query = " ".join(query_parts)
        
        self.logger.debug(f"Built retrieval query: '{query}'")
        return query

    def _expand_profile(self, perfil_usuario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expande el perfil del usuario en variables individuales para el template.
        
        Args:
            perfil_usuario: Diccionario con datos del usuario
            
        Returns:
            Dict con campos expandidos (nivel, objetivo, restricciones)
        """
        restricciones = perfil_usuario.get("restricciones", [])
        restricciones_str = ", ".join(restricciones) if restricciones else "ninguna"
        
        expanded = {
            "nivel": perfil_usuario.get("level", "intermedio"),
            "objetivo": perfil_usuario.get("objetivo", "hipertrofia"),
            "restricciones": restricciones_str
        }
        
        self.logger.debug(f"Expanded profile: {expanded}")
        return expanded

    def get_extraction_chain(self) -> RunnableSequence:
        """
        Builds and returns the LangChain Expression Language (LCEL) chain
        for extracting training principles.

        Returns:
            RunnableSequence: The compiled LCEL chain.
        """
        self.logger.info("Building the principle extraction RAG chain...")

        try:
            # 1. Load Raw Prompt Template
            raw_prompt = self._load_prompt_template()

            # 2. Get Retriever
            vector_store = self.vector_store_manager.get_or_create_vectorstore()
            retriever = vector_store.as_retriever()
            self.logger.info("Retriever obtained from vector store.")

            # 3. Create Prompt Template Instance
            # input_variables deben coincidir con las variables en el template
            prompt_template = PromptTemplate(
                template=raw_prompt,
                input_variables=["contexto_libro", "nivel", "objetivo", "restricciones"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()}
            )
            self.logger.info("PromptTemplate instance created.")

            # ════════════════════════════════════════════════════════════════
            # 4. CADENA LCEL CORREGIDA
            # ════════════════════════════════════════════════════════════════
            # Paso 1: Agregar contexto_libro mediante retrieval
            # Paso 2: Expandir perfil en variables individuales (nivel, objetivo, restricciones)
            # Paso 3: Formatear prompt
            # Paso 4: Invocar LLM
            # Paso 5: Parsear respuesta
            
            chain = (
                # Paso 1: Agregar contexto del libro
                RunnablePassthrough.assign(
                    contexto_libro=RunnableLambda(self._build_retrieval_query) | retriever
                )
                # Paso 2: Expandir perfil en variables individuales
                | RunnableLambda(lambda x: {
                    "contexto_libro": x["contexto_libro"],
                    **self._expand_profile(x)
                })
                # Paso 3-5: Prompt → LLM → Parser
                | prompt_template
                | self.llm
                | self.parser
            )
            
            self.logger.info("✅ Principle extraction RAG chain built successfully.")
            return chain

        except Exception as e:
            self.logger.error(f"Error building the extraction chain: {e}", exc_info=True)
            raise