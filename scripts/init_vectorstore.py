import sys
import argparse
from pathlib import Path

# Añadir raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.settings import Config
    from utils.logger import setup_logger
    # Importar el manager ahora que existe
    from rag.vectorstore_manager import VectorStoreManager
except ImportError as e:
    print(f"Error: Faltan dependencias. Asegúrate de instalar requirements.txt y que los archivos de Fases previas existan. Error: {e}")
    sys.exit(1)

logger = setup_logger(__name__)

def init_vectorstore(force: bool = False) -> bool:
    """
    Inicializa o carga el vector store usando VectorStoreManager.
    """
    try:
        vector_store_manager = VectorStoreManager()

        # Verificar si ya existe (la lógica está ahora DENTRO del manager)
        chroma_dir_str = str(vector_store_manager.config.CHROMA_DIR)
        if Path(chroma_dir_str).exists() and not force:
            logger.warning(f"Vector store ya existe en {chroma_dir_str}")
            logger.info("Usa --force para re-vectorizar. Cargando existente...")
            # Igualmente llamamos a get_or_create para asegurar que se cargue
            vector_store_manager.get_or_create_vectorstore()
            return True # Consideramos éxito si ya existe y no se fuerza

        logger.info(f"Inicializando vector store desde: {vector_store_manager.config.BOOK_PATH}")

        # La lógica de creación/carga ahora está encapsulada
        vector_store_manager.get_or_create_vectorstore() # Esto llamará a build_vectorstore si es necesario

        logger.info(f"✅ Vector store inicializado/cargado en: {chroma_dir_str}")
        return True

    except FileNotFoundError as e:
         logger.error(f"Error: Archivo no encontrado - {e}")
         return False
    except Exception as e:
        logger.error(f"Error al inicializar vector store: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Inicializa el Vector Store ChromaDB")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-vectorizar aunque el directorio Chroma ya exista"
    )
    args = parser.parse_args()

    logger.info("Iniciando script de inicialización del Vector Store...")
    success = init_vectorstore(force=args.force)

    if not success:
        logger.error("❌ Inicialización del Vector Store FALLIDA.")
        sys.exit(1)
    else:
        # El mensaje de éxito específico (creado o cargado) ya lo da el manager
        logger.info("✅ Script de inicialización finalizado.")
        sys.exit(0)

if __name__ == "__main__":
    main()