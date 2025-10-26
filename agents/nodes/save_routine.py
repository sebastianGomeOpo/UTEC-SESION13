import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Project Imports
from agents.graph_state import GraphState
from rag.models import RutinaActiva # Assuming RutinaActiva for type hint
from config.settings import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

def save_routine(state: GraphState) -> GraphState:
    """
    Nodo: save_routine

    Persiste la rutina generada (rutina_final) en el archivo JSON del usuario,
    creando un backup antes de escribir.

    Args:
      state (GraphState): Estado con user_id y rutina_final llenos.

    Returns:
      GraphState: Estado actualizado con respuesta_usuario o error.

    Raises:
      Ninguno (errores van a state["error"]).
    """
    logger.info("--- Entering Save Routine Node ---")
    user_id = state.get("user_id")
    rutina_final: RutinaActiva | None = state.get("rutina_final")

    if not user_id:
        logger.error("User ID missing in state.")
        state["error"] = "User ID no disponible para guardar rutina."
        state["step_completed"] = "save_routine_error"
        return state
    if not rutina_final:
        logger.error("'rutina_final' missing or None in state.")
        state["error"] = "Rutina final vacía, no se puede guardar."
        state["step_completed"] = "save_routine_error"
        return state

    backup_path = None
    original_content = None
    config = Config() # Instantiate config to access USERS_DIR
    user_file_path = config.USERS_DIR / f"{user_id}.json"

    try:
        logger.info(f"Attempting to save routine for user {user_id} to {user_file_path}")

        if not user_file_path.exists():
            # This should ideally be caught by load_context, but double-check
            logger.error(f"User file {user_file_path} not found. Cannot save routine.")
            state["error"] = f"Archivo de usuario no encontrado en {user_file_path}"
            state["step_completed"] = "save_routine_error"
            return state

        # --- Transaction Start ---
        # 1. Read current content
        with open(user_file_path, "r", encoding="utf-8") as f:
            original_content = f.read()
            user_data = json.loads(original_content) # Load into dict

        # 2. Create backup
        timestamp_backup = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = user_file_path.parent / f"{user_file_path.stem}.{timestamp_backup}.backup"
        shutil.copy(str(user_file_path), str(backup_path))
        logger.info(f"Created backup at: {backup_path}")

        # 3. Update data
        # Use model_dump for Pydantic V2 serialization
        user_data["rutina_activa"] = rutina_final.model_dump(mode='json')
        user_data["updated_at"] = datetime.now().isoformat()
        logger.debug("User data updated with new routine.")


        # 4. Write updated data back to original file
        with open(user_file_path, "w", encoding="utf-8") as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully wrote updated data to {user_file_path}")

        # 5. Verification (Optional but recommended)
        with open(user_file_path, "r", encoding="utf-8") as f:
            written_data = json.load(f)
        if written_data.get("rutina_activa", {}).get("fecha_creacion") != rutina_final.fecha_creacion:
             # Basic check, compare a key field
             raise IOError("Verification failed: Written data does not match expected routine.")
        logger.info("Post-write verification passed.")
        # --- Transaction End ---


        # Success
        state["respuesta_usuario"] = "✅ Rutina guardada exitosamente en tu perfil."
        state["step_completed"] = "saved" # Final successful state
        # Optionally remove old backups here if needed

    except (IOError, OSError, shutil.Error) as e:
        logger.exception(f"File system error saving routine for {user_id}: {e}")
        state["error"] = f"Error de archivo guardando rutina: {str(e)}"
        state["step_completed"] = "save_routine_failed"
        # Attempt to restore from backup
        if backup_path and backup_path.exists() and original_content:
            try:
                logger.warning(f"Attempting to restore original file {user_file_path} from backup {backup_path}")
                # Option 1: Copy backup over original
                # shutil.copy(str(backup_path), str(user_file_path))
                # Option 2: Write original content back (safer if original_content was read successfully)
                with open(user_file_path, "w", encoding="utf-8") as f_restore:
                     f_restore.write(original_content)
                logger.info("Restored original file content.")
            except Exception as restore_e:
                logger.error(f"CRITICAL: Failed to restore from backup after save error: {restore_e}")
                state["error"] += f" | ADVERTENCIA: No se pudo restaurar el backup: {restore_e}"
    except json.JSONDecodeError as e:
         logger.exception(f"Error decoding existing JSON for user {user_id}: {e}")
         state["error"] = f"Archivo de usuario existente está corrupto: {str(e)}"
         state["step_completed"] = "save_routine_failed"
    except Exception as e:
        logger.exception(f"An unexpected error occurred saving routine for user {user_id}: {e}")
        state["error"] = f"Error inesperado guardando rutina: {str(e)}"
        state["step_completed"] = "save_routine_failed"
         # Attempt restore here too if backup was created before unexpected error
        if backup_path and backup_path.exists() and original_content:
             try:
                 logger.warning(f"Attempting restore due to unexpected error...")
                 with open(user_file_path, "w", encoding="utf-8") as f_restore:
                     f_restore.write(original_content)
                 logger.info("Restored original file content.")
             except Exception as restore_e:
                 logger.error(f"CRITICAL: Failed to restore from backup: {restore_e}")
                 state["error"] += f" | ADVERTENCIA: No se pudo restaurar: {restore_e}"


    logger.info("--- Exiting Save Routine Node ---")
    return state