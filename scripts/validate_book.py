import sys
from pathlib import Path

# Añadir raíz del proyecto al path para importar config y utils
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config.settings import Config
    from utils.logger import setup_logger
    from langchain_community.document_loaders import PyPDFLoader
except ImportError as e:
    print(f"Error: Faltan dependencias. Asegúrate de instalar requirements.txt. Error: {e}")
    sys.exit(1)

logger = setup_logger(__name__)

def validate_book(book_path: str) -> bool:
   
    try:
        logger.info(f"Validando libro: {book_path}")

        # 1. Verificar existencia
        if not Path(book_path).exists():
            logger.error(f"Libro no encontrado en: {book_path}")
            return False

        # 2. Cargar con PyPDFLoader
        loader = PyPDFLoader(book_path)
        pages = loader.load()

        if not pages:
            logger.error("PDF no contiene páginas legibles o está corrupto.")
            return False

        # 3. Verificar número mínimo de páginas
        if len(pages) < 50:
            logger.warning(f"PDF tiene solo {len(pages)} páginas (esperado: 50+)")
            # No es un error fatal, solo un warning.

        # 4. Verificar que hay texto extraíble (CRÍTICO para RAG)
        # Revisar las primeras 10 páginas
        total_text = "".join([p.page_content for p in pages[:10]])

        if len(total_text.strip()) < 100:
            logger.error("PDF parece estar escaneado (sin texto extraíble en las primeras 10 páginas)")
            return False

        # 5. Buscar keywords críticas
        keywords = ["Tabla", "RIR", "entrenamiento", "ejercicio"]
        found_keywords = [kw for kw in keywords if kw.lower() in total_text.lower()]

        if len(found_keywords) < 2:
            logger.warning(f"Solo se encontraron {len(found_keywords)} keywords críticas de {len(keywords)} en las primeras 10 páginas.")
            # No es un error fatal, solo un warning.

        logger.info(f"[+] Validación exitosa: {len(pages)} páginas, {len(total_text)} caracteres extraídos (primeras 10 págs)")
        logger.info(f"[+] Keywords encontradas: {found_keywords}")

        return True

    except Exception as e:
        logger.error(f"Error en validación: {e} (¿PDF corrupto o protegido?)")
        return False

if __name__ == "__main__":
    # Usar la constante de config/settings.py
    success = validate_book(str(Config.BOOK_PATH))

    if not success:
        logger.error("[!] Validación FALLIDA - No continuar con vectorización")
        sys.exit(1)
    else:
        logger.info("[!] Validación EXITOSA - Proceder con vectorización")
        sys.exit(0)