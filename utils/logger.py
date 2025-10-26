import logging
import os
import sys # Importa sys
from datetime import datetime

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar duplicación de handlers
    if logger.handlers:
        return logger
    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Handler para archivo
    if log_file is None:
        log_file = os.path.join("logs", "entrenador.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # Handler de archivo (generalmente maneja bien UTF-8 por defecto, pero especificar no hace daño)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler para consola (opcional)
    # FORZAR UTF-8 para StreamHandler
    console_handler = logging.StreamHandler(sys.stdout) # Usa sys.stdout
    console_handler.setFormatter(formatter)
    # Establecer explícitamente la codificación UTF-8 para la salida de consola
    # Esto ayuda en entornos Windows donde la codificación por defecto puede ser otra
    try:
        # Python 3.7+
        console_handler.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1))
    except Exception:
         # Fallback por si lo anterior falla (menos probable que funcione en Windows cmd legacy)
        pass # Podrías intentar omitir la configuración de encoding si falla

    logger.addHandler(console_handler)

    return logger
