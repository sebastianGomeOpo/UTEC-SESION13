"""
Agente Entrenador Personal
"""
import sys
from agents.entrenador import EntrenadorAgent
from config.settings import Config

def main():
    """Inicia el chatbot"""
    print("=" * 60)
    print("🏋️  ENTRENADOR PERSONAL AI")
    print("=" * 60)
    
    # Seleccionar usuario
    user_id = input("\nID de usuario (Enter para 'default'): ").strip()
    if not user_id:
        user_id = None  # Usará default
    
    # ========================================
    # ALTERNATIVA: Detectar --verbose desde línea de comandos
    # ========================================
    verbose = "--verbose" in sys.argv
    
    # Cargar configuración desde JSONs
    config = Config(user_id=user_id)
    
    # Crear agente pasando el flag verbose
    entrenador = EntrenadorAgent(config, verbose=verbose)
    
    print("\nComandos: 'salir' para terminar")
    if verbose:
        print("🔍 Modo VERBOSE activado")
    entrenador.run()

if __name__ == "__main__":
    main()