import os
from pathlib import Path

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Project Imports
from config.settings import Config
from utils.logger import setup_logger
from rag.chunking_strategy import get_semantic_text_splitter

class VectorStoreManager:
    """
    Manages the creation, loading, and accessing of the Chroma vector store.
    """

    def __init__(self):
        """
        Initializes the VectorStoreManager.
        """
        self.logger = setup_logger(__name__)
        self.config = Config()
        self.text_splitter = get_semantic_text_splitter()
        self.embedding_function = OpenAIEmbeddings()

    def build_vectorstore(self):
        """
        Loads the book, splits it into chunks, creates embeddings,
        and persists them into a Chroma vector store.
        """
        self.logger.info(f"Starting vector store build from: {self.config.BOOK_PATH}")
        if not self.config.BOOK_PATH.exists():
             self.logger.error(f"Book PDF not found at {self.config.BOOK_PATH}")
             raise FileNotFoundError(f"Book PDF not found at {self.config.BOOK_PATH}")

        try:
            # Load PDF
            loader = PyPDFLoader(str(self.config.BOOK_PATH))
            docs = loader.load()
            if not docs:
                self.logger.error("No documents loaded from PDF. Check PDF content and integrity.")
                return
            self.logger.info(f"Loaded {len(docs)} pages from PDF.")

            # Split Documents
            chunks = self.text_splitter.split_documents(docs)
            if not chunks:
                 self.logger.error("Text splitting resulted in zero chunks.")
                 return
            self.logger.info(f"Split documents into {len(chunks)} chunks.")

            # Create and Persist Vector Store
            self.logger.info(f"Creating and persisting vector store at: {self.config.CHROMA_DIR}")
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_function,
                persist_directory=str(self.config.CHROMA_DIR)
            )
            self.logger.info("✅ Vector store built and persisted successfully.")

        except Exception as e:
            self.logger.error(f"Error building vector store: {e}", exc_info=True)
            raise

    def get_or_create_vectorstore(self) -> Chroma:
        """
        Loads the existing Chroma vector store if it exists, otherwise builds it.

        Returns:
            Chroma: An instance of the loaded or newly created vector store.
        """
        persist_dir_str = str(self.config.CHROMA_DIR)
        if not os.path.exists(persist_dir_str):
            self.logger.warning(f"Vector store not found at {persist_dir_str}. Building now...")
            self.build_vectorstore()
        else:
            self.logger.info(f"Loading existing vector store from: {persist_dir_str}")

        # Load (or re-load after building)
        try:
            vector_store = Chroma(
                persist_directory=persist_dir_str,
                embedding_function=self.embedding_function
            )
            self.logger.info("✅ Vector store loaded successfully.")
            return vector_store
        except Exception as e:
            self.logger.error(f"Error loading vector store from {persist_dir_str}: {e}", exc_info=True)
            raise
