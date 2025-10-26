import sys
import argparse
from pathlib import Path

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config.settings import Config
    from utils.logger import setup_logger
except ImportError as e:
    print(f"Error: Faltan dependencias de Fase 0 (config/logger). Error: {e}")
    sys.exit(1)

logger = setup_logger(__name__)

def init_vectorstore(force: bool = False) -> bool:
    """
    Inicializa el vector store.
    En Fase 0, este script está bloqueado y fallará intencionalmente.
    """
    
    try:
        # Verificar si ya existe
        if Path(Config.CHROMA_DIR).exists() and not force:
            logger.warning(f"Vector store ya existe en {Config.CHROMA_DIR}")
            logger.info("Usa --force para re-vectorizar")
            return True

        logger.info(f"Inicializando vector store desde: {Config.BOOK_PATH}")

        # BLOQUEO DE FASE 0
        # TODO: Implementar en Fase 2
        try:
            # Este import fallará hasta que Fase 2 esté completa
            from rag.vectorstore_manager import VectorStoreManager

            # Si el import funciona por error, fallar igualmente
            logger.error("❌ ERROR INESPERADO: VectorStoreManager existe, pero la lógica de Fase 0 no debe ejecutarse.")
            logger.info("Este script se ejecutará después de completar Fase 2")
            return False

        except ImportError:
            logger.error("❌ VectorStoreManager no implementado aún (esperado en Fase 2)")
            logger.info("Este script se ejecutará después de completar Fase 2")
            return False

    except Exception as e:
        logger.error(f"Error al inicializar vector store: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Inicializa el Vector Store")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-vectorizar aunque ya exista"
    )
    args = parser.parse_args()

    success = init_vectorstore(force=args.force)

    if not success:
        logger.error("❌ Inicialización FALLIDA (esperado en Fase 0)")
        sys.exit(1)
    else:
        logger.info("✅ Inicialización EXITOSA (o ya existía)")
        sys.exit(0)

if __name__ == "__main__":
    main()