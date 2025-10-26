"""
Schemas Pydantic para structured output del sistema RAG.

Estos modelos definen los contratos de datos entre:
- RAG Tool (extrae principios del libro)
- Assembler Tool (genera rutina final)
- GraphState (estado del grafo LangGraph)

IMPORTANTE: Todos los modelos usan Field() con descripciones para guiar al LLM.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# MODELOS DE EXTRACCIÓN RAG (Paso R: Recuperar Principios)
# ============================================================================

class ECI(BaseModel):
    """
    Ejercicio Complementario Individualizado.
    
    Extraído del libro cuando el usuario tiene restricciones físicas.
    Ejemplo: "Puente de Glúteo" para "rodilla izquierda".
    """
    nombre_ejercicio: str = Field(
        description="Nombre completo del ejercicio compensatorio del libro"
    )
    motivo: str = Field(
        description="Por qué se recomienda este ECI (ej: 'Para compensar rodilla izquierda')"
    )
    fuente_cita: str = Field(
        description="Página exacta del libro (ej: 'Página 107, Tabla 16')"
    )
    sets: int = Field(
        default=3,
        description="Número de series recomendadas para ECI"
    )
    reps: str = Field(
        default="8-12",
        description="Rango de repeticiones (formato string '8-12')"
    )
    
    class Config:
        frozen = False  # Permitir modificación (ej. ajustar sets)


class PrincipiosExtraidos(BaseModel):
    """
    Variables de entrenamiento extraídas del libro mediante RAG.
    
    Este modelo GARANTIZA que el LLM extraiga los principios correctos
    y los devuelva con citas verificables.
    
    CRÍTICO: Cada campo debe tener una fuente en citas_fuente.
    """
    intensidad_RIR: str = Field(
        description="Rango de RIR (Reserve In Reserve) según el libro. Formato: '1-2' o '2-3'"
    )
    rango_repeticiones: str = Field(
        description="Rango de repeticiones recomendado. Formato: '6-15' o '8-12'"
    )
    descanso_series_s: int = Field(
        description="Descanso entre series en segundos (ej: 90, 120)"
    )
    cadencia_tempo: str = Field(
        description="Tempo de ejecución. Formato: 'excéntrica:pausa:concéntrica:pausa' (ej: '3:0:1:1')"
    )
    frecuencia_semanal: str = Field(
        description="Frecuencia recomendada por semana. Formato: '3-5 días' o '4 días'"
    )
    ECI_recomendados: List[ECI] = Field(
        default_factory=list,
        description="Lista de ejercicios compensatorios individualizados según restricciones del usuario"
    )
    citas_fuente: List[str] = Field(
        description="Páginas exactas del libro de donde se extrajeron estos principios. Mínimo 1 cita."
    )
    
    class Config:
        frozen = True  # Inmutable - los principios del libro no cambian


# ============================================================================
# MODELOS DE RUTINA (Paso G: Generar Rutina)
# ============================================================================

class Ejercicio(BaseModel):
    """
    Ejercicio individual en una sesión de entrenamiento.
    """
    nombre: str = Field(
        description="Nombre del ejercicio (ej: 'Sentadilla', 'Press de banca')"
    )
    tipo: str = Field(
        description="Tipo de ejercicio: 'principal', 'ECI' o 'accesorio'"
    )
    sets: int = Field(
        ge=1,
        le=10,
        description="Número de series (1-10)"
    )
    reps: str = Field(
        description="Repeticiones por serie. Formato: '8-12' o '5' (puede ser rango o número fijo)"
    )
    RIR: str = Field(
        description="Reserve In Reserve. Formato: '1-2' o '3'. DEBE coincidir con principios_libro"
    )
    tempo: str = Field(
        description="Cadencia de ejecución. Formato: '3:0:1:1'. DEBE coincidir con principios_libro"
    )
    descanso_s: int = Field(
        ge=30,
        le=300,
        description="Descanso después de este ejercicio en segundos (30-300)"
    )
    notas: Optional[str] = Field(
        default=None,
        description="Notas adicionales del entrenador (ej: 'Enfocarse en forma', 'Progresar peso semanalmente')"
    )


class Sesion(BaseModel):
    """
    Sesión de entrenamiento (un día de entreno).
    """
    dia_semana: str = Field(
        description="Día de la semana para esta sesión (ej: 'lunes', 'martes')"
    )
    enfoque_muscular: str = Field(
        description="Grupo muscular principal (ej: 'tren superior', 'pierna', 'full body')"
    )
    ejercicios: List[Ejercicio] = Field(
        min_items=1,
        description="Lista de ejercicios en esta sesión. Mínimo 1 ejercicio."
    )
    duracion_estimada_min: int = Field(
        ge=20,
        le=120,
        description="Duración estimada de la sesión en minutos (20-120)"
    )


class RutinaActiva(BaseModel):
    """
    Rutina de entrenamiento completa generada por el sistema.
    
    CRÍTICO: Esta rutina incluye los principios_aplicados con citas,
    permitiendo auditoría de por qué se eligió cada parámetro.
    """
    nombre: str = Field(
        default="Rutina Personalizada",
        description="Nombre de la rutina"
    )
    sesiones: List[Sesion] = Field(
        min_items=1,
        description="Lista de sesiones de entrenamiento. Mínimo 1 sesión."
    )
    principios_aplicados: PrincipiosExtraidos = Field(
        description="Principios del libro que se usaron para crear esta rutina. INCLUYE CITAS."
    )
    fecha_creacion: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp ISO de cuándo se creó la rutina"
    )
    validez_semanas: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Cuántas semanas es válida esta rutina antes de re-evaluar (1-12)"
    )
    
    class Config:
        # Permitir modificación (ej. extender validez)
        frozen = False


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def validate_routine_against_principles(
    rutina: RutinaActiva,
    principios: PrincipiosExtraidos
) -> bool:
    """
    Valida que la rutina generada respete los principios del libro.
    
    Verificaciones:
    - RIR de ejercicios principales coincide con principios
    - Tempo coincide con principios
    - ECIs recomendados están presentes en la rutina
    
    Args:
        rutina: Rutina generada
        principios: Principios extraídos del libro
    
    Returns:
        True si válido, False si hay inconsistencias
    
    Raises:
        ValueError: Si encuentra inconsistencias críticas
    """
    errors = []
    
    # Verificar RIR en ejercicios principales
    for sesion in rutina.sesiones:
        for ejercicio in sesion.ejercicios:
            if ejercicio.tipo == "principal":
                if ejercicio.RIR != principios.intensidad_RIR:
                    errors.append(
                        f"Ejercicio '{ejercicio.nombre}' tiene RIR={ejercicio.RIR} "
                        f"pero principios dicen RIR={principios.intensidad_RIR}"
                    )
                
                if ejercicio.tempo != principios.cadencia_tempo:
                    errors.append(
                        f"Ejercicio '{ejercicio.nombre}' tiene tempo={ejercicio.tempo} "
                        f"pero principios dicen tempo={principios.cadencia_tempo}"
                    )
    
    # Verificar que ECIs recomendados estén presentes
    eci_names = [eci.nombre_ejercicio for eci in principios.ECI_recomendados]
    rutina_ejercicios = [
        ej.nombre 
        for sesion in rutina.sesiones 
        for ej in sesion.ejercicios
    ]
    
    for eci_name in eci_names:
        if eci_name not in rutina_ejercicios:
            errors.append(
                f"ECI obligatorio '{eci_name}' no está en la rutina generada"
            )
    
    if errors:
        raise ValueError(
            "Rutina inconsistente con principios:\n" + "\n".join(errors)
        )
    
    return True


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ECI",
    "PrincipiosExtraidos",
    "Ejercicio",
    "Sesion",
    "RutinaActiva",
    "validate_routine_against_principles"
]