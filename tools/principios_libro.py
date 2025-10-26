# Standard Library Imports
import logging
from typing import Dict, Any, List, Optional

# Project Imports
# Assuming PrincipleExtractor is correctly placed and importable
try:
    from rag.principle_extractor import PrincipleExtractor
    from rag.models import PrincipiosExtraidos
except ImportError:
    logging.warning("Could not import RAG components for PrincipiosLibroTool. Tool will be non-functional.")
    PrincipleExtractor = None
    PrincipiosExtraidos = None


class PrincipiosLibroTool:
    """
    Wrapper tool for extracting training principles using PrincipleExtractor.
    Intended primarily for debugging, testing, or potential future reuse outside the main graph.
    """
    def __init__(self):
        """Initializes the tool."""
        self.logger = logging.getLogger(__name__)
        self.extractor_instance = None
        if PrincipleExtractor:
            try:
                self.extractor_instance = PrincipleExtractor()
                self.logger.info("PrincipiosLibroTool initialized with PrincipleExtractor.")
            except Exception as e:
                self.logger.error(f"Failed to initialize PrincipleExtractor in tool: {e}", exc_info=True)
        else:
            self.logger.warning("PrincipleExtractor not available. PrincipiosLibroTool is non-functional.")

    def execute(self, perfil_usuario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts training principles based on the user profile.

        Args:
            perfil_usuario: Dictionary containing user level, objective, restrictions, etc.

        Returns:
            dict: A dictionary containing:
                - success (bool): True if extraction succeeded, False otherwise.
                - principios (dict, optional): The extracted principles as a dictionary if successful.
                - citas (List[str], optional): Source citations if successful.
                - error (str, optional): Error message if extraction failed.
        """
        if not self.extractor_instance or not PrincipiosExtraidos:
            return {
                "success": False,
                "error": "PrincipleExtractor component is not available or failed to initialize."
            }

        self.logger.info(f"Executing PrincipiosLibroTool for profile: {perfil_usuario.get('user_id', 'unknown')}")
        try:
            chain = self.extractor_instance.get_extraction_chain()
            # The chain input is just the profile dict based on RunnablePassthrough
            result = chain.invoke(perfil_usuario)

            if not result:
                return {"success": False, "error": "Extraction resulted in None."}

            if not result.citas_fuente:
                # Even though the node handles this, the tool should also report it
                return {"success": False, "error": "Alucinación detectada: sin citas de fuente."}


            return {
                "success": True,
                # Use model_dump for Pydantic V2
                "principios": result.model_dump(mode='json'),
                "citas": result.citas_fuente,
                # Add confidence if available in your model
                # "confianza": getattr(result, 'confianza', None)
            }
        except Exception as e:
            self.logger.exception(f"Error during principle extraction in tool: {e}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }