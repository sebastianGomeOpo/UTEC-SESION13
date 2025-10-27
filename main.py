# -----------------------------------------------------------------------------
# Fase 5: CLI Principal
# Este archivo ejecuta el loop principal de la aplicación,
# determina el tipo de request, invoca el grafo y muestra la respuesta.
# -----------------------------------------------------------------------------

import sys
from datetime import datetime

# Importar el grafo compilado de Fase 5
try:
    from agents.entrenador import graph
except ImportError as e:
    print(f"Error: No se pudo importar el grafo. ¿Están todos los nodos definidos? {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error compilando el grafo: {e}", file=sys.stderr)
    sys.exit(1)

# Imports del proyecto
from agents.graph_state import create_initial_state, GraphState
from config.settings import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Mapeo de keywords para determinar la intención del usuario
REQUEST_TYPE_KEYWORDS = {
    "crear_rutina": ["rutina", "crea", "genera", "nueva", "plan", "programa"],
    # Se movió consultar_historial ANTES para priorizar sus keywords
    "consultar_historial": ["historial", "muéstrame", "qué hice", "registro", "pasado", "hice"], # Añadido "hice"
    "registrar_ejercicio": ["registra", "anota", "ejercicio", "sentadilla", "press", "kg", "peso muerto"],
}

def determinar_request_type(user_input: str) -> str:
    """
    Determina qué tipo de request es basado en keywords.

    Args:
        user_input: El texto crudo del usuario.

    Returns:
        "crear_rutina" | "registrar_ejercicio" | "consultar_historial" | "unknown"
    """
    user_input_lower = user_input.lower()

    # --- CORRECCIÓN ---
    # Verificar tipos en orden de especificidad o prioridad
    # 1. Consultar historial (keywords más específicas como "historial", "muéstrame")
    if any(kw in user_input_lower for kw in REQUEST_TYPE_KEYWORDS["consultar_historial"]):
        logger.debug(f"Input '{user_input}' clasificado como: consultar_historial")
        return "consultar_historial"

    # 2. Registrar ejercicio (keywords como "registra", "anota", "kg")
    if any(kw in user_input_lower for kw in REQUEST_TYPE_KEYWORDS["registrar_ejercicio"]):
        logger.debug(f"Input '{user_input}' clasificado como: registrar_ejercicio")
        return "registrar_ejercicio"

    # 3. Crear rutina (keywords más generales como "rutina", "plan")
    if any(kw in user_input_lower for kw in REQUEST_TYPE_KEYWORDS["crear_rutina"]):
        logger.debug(f"Input '{user_input}' clasificado como: crear_rutina")
        return "crear_rutina"

    # Fallback si no coincide ninguno de los anteriores
    logger.debug(f"Input '{user_input}' no clasificado, tipo: unknown")
    return "unknown"

def main():
    """
    Loop principal de la CLI para el agente C-R-G.
    """
    print("\n" + "="*50)
    print("🏋️  Sistema de Entrenamiento C-R-G (Fase 5)")
    print("="*50)
    print("Comandos: 'login [user_id]' | 'exit'")
    
    user_id = None
    
    # Validar que USERS_DIR existe (buena práctica)
    if not Config.USERS_DIR.exists():
        logger.error(f"El directorio de usuarios no existe: {Config.USERS_DIR}")
        print(f"❌ ERROR: El directorio de usuarios '{Config.USERS_DIR}' no se encontró.")
        sys.exit(1)
    else:
        logger.info(f"Directorio de usuarios verificado: {Config.USERS_DIR}")

    while True:
        try:
            # 1. Leer input del usuario
            if user_id:
                prompt = f"\n[{user_id}] >>> "
            else:
                prompt = "\n[No Logueado] >>> "
                
            user_input = input(prompt).strip()

            if not user_input:
                continue

            # 2. Manejar comandos especiales
            if user_input.lower().startswith("login"):
                parts = user_input.split()
                if len(parts) == 2:
                    new_user_id = parts[1]
                    # Validar que el usuario existe antes de loguear
                    user_file = Config.USERS_DIR / f"{new_user_id}.json"
                    if user_file.exists():
                        user_id = new_user_id
                        print(f"✅ Logueado como: {user_id}")
                        logger.info(f"Usuario logueado: {user_id}")
                    else:
                        print(f"❌ Error: Usuario '{new_user_id}' no encontrado en {Config.USERS_DIR}")
                        logger.warning(f"Intento de login fallido: {new_user_id}")
                else:
                    print("Formato incorrecto. Usa: 'login [user_id]' (ej: login user_001)")
                continue

            if user_input.lower() == "exit":
                print("\n👋 ¡Hasta luego! 💪")
                break

            if not user_id:
                print("❌ Debes hacer login primero. Usa: 'login [user_id]'")
                continue

            # 3. Determinar request_type
            request_type = determinar_request_type(user_input)
            
            # 4. Preparar estado inicial
            initial_state = create_initial_state(user_id, request_type)
            initial_state["user_message"] = user_input # Para nodos legacy y potencialmente otros

            # 5. Invocar el grafo
            logger.info(f"Invocando grafo para {user_id} | Tipo: {request_type} | Msg: '{user_input}'")
            
            # Configuración de checkpoint (opcional pero bueno para debug)
            config = {"configurable": {"thread_id": f"user_session_{user_id}_{datetime.now().timestamp()}"}}

            final_state: GraphState = graph.invoke(initial_state, config=config)

            # 6. Mostrar resultado
            if final_state.get("error"):
                # handle_error ya formatea respuesta_usuario
                response = final_state.get("respuesta_usuario", "Ocurrió un error desconocido.")
                print(f"\n🤖 {response}")
                logger.error(f"Invocación fallida. User: {user_id}. Error: {final_state.get('error')}. Step: {final_state.get('step_completed')}")
            else:
                response = final_state.get("respuesta_usuario", "Operación completada sin mensaje.")
                print(f"\n🤖 {response}")
                logger.info(f"Invocación exitosa. User: {user_id}. Step: {final_state.get('step_completed')}")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupción por usuario. Saliendo...")
            break
        except Exception as e:
            print(f"\n❌ ERROR INESPERADO EN EL LOOP PRINCIPAL: {e}")
            logger.exception("Error fatal en el loop de main.py")
            # Considerar si salir o continuar el loop
            # break # Descomentar para salir en caso de error fatal

if __name__ == "__main__":
    main()
