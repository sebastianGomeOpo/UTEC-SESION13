"""
Factory: Crea herramientas personalizadas por usuario.
Inyecta el contexto de usuario en todas las herramientas.
"""
from ._base import UserContext, set_user_context
from .calculo import calcular_1rm
from .registro import registrar_ejercicio
from .historial import consultar_historial


class TrainingToolsFactory:
    """
    Factory que crea y configura herramientas personalizadas por usuario.
    
    Patrón: Strategy + Dependency Injection
    """
    
    def __init__(self, user_id: str):
        """
        Inicializa la factory con el user_id.
        
        Args:
            user_id: ID del usuario
        """
        self.user_id = user_id
        # Crear contexto del usuario
        self.user_context = UserContext(user_id)
        # Inyectar contexto globalmente
        set_user_context(self.user_context)
    
    def get_tools(self):
        """
        Retorna las herramientas personalizadas.
        
        Returns:
            Lista de herramientas @tool con contexto de usuario
        """
        return [
            calcular_1rm,              # ✅ Sin cambios
            registrar_ejercicio,       # ✅ Usa UserContext
            consultar_historial       # ✅ Usa UserContex       # ✅ Usa UserContext
        ]