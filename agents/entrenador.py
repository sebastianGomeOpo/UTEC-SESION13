"""
Agente Entrenador - Debug Modo Completo FUNCIONAL
Usa stream_mode="debug" con extracción correcta (keys en minúsculas)
"""
import logging
import pprint
from typing import Optional, Dict, Any, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import TrainingToolsFactory
from .base import Agent
from .prompts import PromptLoader
from .presenters import ConsolePresenter, VerbosePresenter, Presenter

logger = logging.getLogger(__name__)


class EntrenadorAgent(Agent):
    """
    Agente entrenador con auditoría DEBUG completa.
    
    - Si verbose=False: Modo normal, sin debug crudo
    - Si verbose=True: Imprime debug crudo + presentación normal
    """

    # ==================== INICIALIZACIÓN ====================

    def __init__(self, config, verbose: bool = False):
        """Inicializa EntrenadorAgent.

        Args:
            config: Objeto con configuración de aplicación y usuario.
            verbose: Si True, imprime debug crudo además de presentar respuesta.
                     Si False, solo presentación normal.
        """
        try:
            self.config = config
            self.config.validate()
            self.verbose = verbose

            self._inicializar_presentador(verbose)
            self._inicializar_modelo()
            self._inicializar_prompt_sistema()
            self._inicializar_memoria()
            self._inicializar_herramientas()
            self._inicializar_agente()
            self._inicializar_estado_sesion()

        except Exception as e:
            logger.error(
                f"❌ Error crítico durante la inicialización: {e}",
                exc_info=True
            )
            raise

    def _inicializar_presentador(self, verbose: bool) -> None:
        """Selecciona presentador según verbose."""
        self.presenter: Presenter = (
            VerbosePresenter() if verbose else ConsolePresenter()
        )
        tipo = "Detallado" if verbose else "Consola"
        debug_str = "🐛 DEBUG Crudo ACTIVADO" if verbose else "Desactivado"
        logger.info(f"✓ Presentador: {tipo} | {debug_str}")

    def _inicializar_modelo(self) -> None:
        """Inicializa el modelo LLM."""
        self.llm = ChatOpenAI(
            model=self.config.MODEL_NAME,
            temperature=self.config.TEMPERATURE,
            max_tokens=getattr(self.config, "MAX_TOKENS", 2048),
            timeout=getattr(self.config, "TIMEOUT", 30),
        )
        logger.info(f"✓ Modelo LLM: {self.config.MODEL_NAME}")

    def _inicializar_prompt_sistema(self) -> None:
        """Carga el prompt del sistema."""
        prompt_loader = PromptLoader()
        self.system_prompt = prompt_loader.get_adaptive_prompt(self.config)
        logger.info("✓ Prompt del sistema cargado")

    def _inicializar_memoria(self) -> None:
        """Configura memoria."""
        self.checkpointer = MemorySaver()
        logger.info("✓ MemorySaver inicializado")

    def _inicializar_herramientas(self) -> None:
        """Crea herramientas del usuario."""
        tools_factory = TrainingToolsFactory(user_id=self.config.user_id)
        self.all_tools = tools_factory.get_tools()
        cantidad = len(self.all_tools)
        logger.info(f"✓ {cantidad} herramienta(s) cargada(s)")

    def _inicializar_agente(self) -> None:
        """Ensambla el agente."""
        self.agent = create_agent(
            model=self.llm,
            tools=self.all_tools,
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
        )
        logger.info("✓ Agente creado exitosamente")

    def _inicializar_estado_sesion(self) -> None:
        """Inicializa estado de sesión."""
        self.conversation_history: List[Dict[str, str]] = []
        logger.info("✓ Historial inicializado")

    # ==================== CONFIGURACIÓN ====================

    def _get_thread_config(self) -> Dict[str, Any]:
        """Obtiene configuración del thread."""
        thread_id = f"session_{self.config.user_id}"
        return {"configurable": {"thread_id": thread_id}}

    def set_verbose(self, verbose: bool) -> None:
        """Cambia modo verbose en runtime."""
        self.verbose = verbose
        self._inicializar_presentador(verbose)

    # ==================== EXTRACCIÓN DE RESPUESTA ====================

    def _extract_final_response_from_debug(self, all_steps: List[Any]) -> Optional[str]:
        """Extrae respuesta final del stream DEBUG.
        
        IMPORTANTE: Las claves en el debug stream son MINÚSCULAS:
        - "type" (no "TYPE")
        - "payload" (no "PAYLOAD")
        - "values" (no "VALUES")
        """
        if not all_steps:
            logger.warning("Lista de steps vacía")
            return None

        try:
            final_checkpoint_payload = None
            
            # Buscar en reversa el último checkpoint
            for step_event_data in reversed(all_steps):
                # CRUCIAL: Usar keys en MINÚSCULAS
                if (isinstance(step_event_data, dict) and
                        step_event_data.get("type") == "checkpoint" and  # ✅ minúscula
                        "payload" in step_event_data):                    # ✅ minúscula

                    payload = step_event_data.get("payload")
                    
                    # Validar estructura
                    if isinstance(payload, dict) and "values" in payload:  # ✅ minúscula
                        values = payload.get("values")
                        
                        if isinstance(values, dict) and "messages" in values:
                            final_checkpoint_payload = payload
                            logger.debug("Checkpoint final encontrado")
                            break

            if final_checkpoint_payload is None:
                logger.warning("No se encontró checkpoint final válido")
                return None

            # Extraer mensajes
            final_values = final_checkpoint_payload.get("values", {})
            if not isinstance(final_values, dict):
                logger.warning("Campo 'values' no es dict")
                return None

            final_messages = final_values.get("messages", [])
            if not isinstance(final_messages, list):
                logger.warning("Campo 'messages' no es lista")
                return None

            # Buscar el último AIMessage con contenido
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage):
                    content = getattr(msg, "content", "")
                    if isinstance(content, str) and content.strip():
                        tool_calls = getattr(msg, "tool_calls", [])
                        # Retornar si tiene contenido O si no tiene tool calls
                        if not tool_calls or content:
                            logger.info("✓ Respuesta extraída del debug stream")
                            return content

            logger.warning("No se encontró AIMessage final con contenido")
            return None

        except Exception as e:
            logger.error(f"Error extrayendo respuesta: {e}", exc_info=True)
            return None

    # ==================== PROCESAMIENTO ====================

    def _process_user_input(self, user_input: str, config: Dict[str, Any]) -> None:
        """Procesa entrada del usuario.
        
        Siempre usa stream_mode="debug" internamente.
        Si verbose=True, imprime debug crudo.
        """
        try:
            all_steps = []
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"🐛 INICIANDO DEBUG: '{user_input}'")
                print(f"{'='*70}")

            # SIEMPRE usar stream_mode="debug" para máxima información
            for chunk in self.agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
                stream_mode="debug"  # ✅ Auditoría completa
            ):
                all_steps.append(chunk)

                # Imprimir debug crudo SOLO si verbose está activo
                if self.verbose:
                    print("\n--- RAW DEBUG CHUNK ---")
                    pprint.pprint(chunk, indent=2, width=120)
                    print("----------------------")

            if self.verbose:
                print(f"\n{'='*70}")
                print(f"🏁 FIN DEBUG ({len(all_steps)} eventos)")
                print(f"{'='*70}\n")

            # Extraer respuesta (con keys correctas en minúsculas)
            ai_response = self._extract_final_response_from_debug(all_steps)

            # Presentar respuesta usando el presenter
            if ai_response:
                self.presenter.print_final_response(ai_response)
                self._save_conversation_turn(user_input, ai_response)
            else:
                logger.warning("No se pudo extraer respuesta")
                self.presenter.print_error("No se generó respuesta")

        except Exception as e:
            logger.error(f"Error procesando entrada: {e}", exc_info=True)
            self.presenter.print_error(str(e))

    def _save_conversation_turn(self, user_input: str, ai_response: str) -> None:
        """Guarda turno de conversación."""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "agent": ai_response,
            "user_id": self.config.user_id
        })

    # ==================== LOOP PRINCIPAL ====================

    def run(self) -> None:
        """Loop principal del agente."""
        config = self._get_thread_config()

        user_context = {
            "USER_NAME": self.config.USER_NAME,
            "OBJETIVO": self.config.OBJETIVO,
            "USER_LEVEL": self.config.USER_LEVEL
        }
        self.presenter.print_user_context(user_context)

        if self.verbose:
            print("\n🐛 MODO DEBUG ACTIVADO - Verás debug crudo en cada consulta\n")

        logger.info(f"Sesión iniciada: {self.config.user_id}")

        while True:
            try:
                user_input = input("\n💬 Tú: ").strip()

                if not user_input:
                    print("⚠️  Por favor, escribe algo...")
                    continue

                if user_input.lower() in ["salir", "exit", "quit"]:
                    self._handle_exit()
                    break

                self._process_user_input(user_input, config)

            except KeyboardInterrupt:
                self._handle_exit()
                break

            except Exception as e:
                logger.error(f"Error en loop: {e}", exc_info=True)
                self.presenter.print_error(str(e))

    def _handle_exit(self) -> None:
        """Salida limpia."""
        print("\n\n👋 Entrenador: ¡Hasta luego! 💪")
        self.config.save_user_data()
        total = len(self.conversation_history)
        logger.info(f"Sesión finalizada: {total} conversaciones")

    # ==================== GETTERS ====================

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Retorna copia del historial."""
        return self.conversation_history.copy()

    def get_agent(self) -> Any:
        """Retorna el agente."""
        return self.agent

    def get_presenter(self) -> Presenter:
        """Retorna el presenter."""
        return self.presenter

    def get_config(self) -> Any:
        """Retorna la configuración."""
        return self.config

    def get_session_info(self) -> Dict[str, Any]:
        """Retorna información de sesión."""
        return {
            "user_id": self.config.user_id,
            "user_name": self.config.USER_NAME,
            "total_conversaciones": len(self.conversation_history),
            "modelo": self.config.MODEL_NAME,
            "modo_verbose": isinstance(self.presenter, VerbosePresenter),
            "modo_debug_crudo": self.verbose,
        }

    # ==================== UTILIDADES ====================

    def clear_history(self) -> None:
        """Limpia historial."""
        self.conversation_history.clear()
        logger.info(f"Historial limpiado: {self.config.user_id}")

    def export_history(self, filepath: str) -> None:
        """Exporta historial a JSON."""
        import json
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            logger.info(f"Historial exportado: {filepath}")
        except IOError as e:
            logger.error(f"Error exportando: {e}")
            raise