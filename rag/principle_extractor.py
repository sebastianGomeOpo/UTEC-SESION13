from pathlib import Path

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

# Project Imports
from config.settings import Config
from rag.models import PrincipiosExtraidos  # Assuming PrincipiosExtraidos is in rag.models
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
            prompt_template = PromptTemplate(
                template=raw_prompt,
                input_variables=["perfil_usuario", "contexto_libro"], # Corrected input variable name
                partial_variables={"format_instructions": self.parser.get_format_instructions()}
            )
            self.logger.info("PromptTemplate instance created.")

            # 4. Define the LCEL Chain
            # Passes user profile directly, retrieves context based on user profile query implicitly
            chain = (
                {
                    "contexto_libro": retriever, # Context fetched via retriever
                    "perfil_usuario": RunnablePassthrough() # User profile passed through
                 }
                | prompt_template
                | self.llm
                | self.parser
            )
            self.logger.info("✅ Principle extraction RAG chain built successfully.")
            return chain

        except Exception as e:
            self.logger.error(f"Error building the extraction chain: {e}", exc_info=True)
            raise