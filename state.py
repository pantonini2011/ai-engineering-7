"""Estado compartido del orquestador y contratos de datos de cada agente.

Reglas del estado (Unidad 3, "Estado compartido y comunicación asíncrona"):

- **Aislamiento de escritura**: cada agente es dueño de una parcela.
  `investigaciones` solo la escribe el investigador, `analisis` solo el analista,
  `respuesta_final` solo el sintetizador. El supervisor escribe los campos de control.
- **Trazabilidad**: todos los nodos agregan una `Contribucion` a la bitácora
  `contribuciones` (quién hizo qué, en qué turno y paso, con qué fuentes).
- **Validación antes de escribir**: lo que produce un LLM entra al estado solo
  después de validarse contra un modelo Pydantic (`Investigacion`, `Analisis`,
  `DecisionSupervisor`).
- **Sin estado mutable compartido**: los nodos nunca hacen `.append()` sobre el
  estado; devuelven listas nuevas y el reducer `operator.add` las concatena.
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Literal, TypeVar

from langgraph.graph import MessagesState
from pydantic import BaseModel, ConfigDict, Field

Especialista = Literal["investigador", "analista"]
# Literal de las aristas condicionales del supervisor: los nombres de los nodos destino.
DestinoSupervisor = Literal["investigador", "analista", "validacion"]
DestinoValidacion = Literal["supervisor", "sintetizador"]
Agente = Literal["inicio", "supervisor", "investigador", "analista", "validacion", "sintetizador"]


class _Contrato(BaseModel):
    """Base de los contratos: rechaza campos que no estén en el esquema."""

    model_config = ConfigDict(extra="forbid")


# --- Salidas estructuradas de los LLM --------------------------------------------


class DecisionSupervisor(_Contrato):
    """Decisión de ruteo del supervisor (salida estructurada de su LLM)."""

    razonamiento: str = Field(
        description="Una o dos oraciones: qué falta para responder y por qué elegís ese paso."
    )
    siguiente: Literal["investigador", "analista", "FINALIZAR"] = Field(
        description="Quién interviene ahora, o FINALIZAR si la rúbrica de suficiencia se cumple."
    )
    instruccion: str = Field(
        default="",
        description=(
            "Instrucción AUTOCONTENIDA para el especialista elegido: el especialista no ve la "
            "conversación, así que incluí los datos concretos que necesita (cifras del usuario, "
            "qué buscar, qué calcular). Vacía si siguiente=FINALIZAR."
        ),
    )


class Hallazgo(_Contrato):
    dato: str = Field(
        min_length=3,
        description="Dato concreto encontrado en la documentación (cifra, umbral, procedimiento).",
    )
    fuente: str = Field(
        description="Archivo de origen tal como lo devolvió buscar_documentacion, ej. 'runbook_incidentes.md'."
    )


class Investigacion(_Contrato):
    """Salida del agente investigador."""

    resumen: str = Field(description="Resumen breve de lo encontrado, en español.")
    hallazgos: list[Hallazgo] = Field(
        default_factory=list,
        description="Datos relevantes, cada uno con su fuente. Solo lo que esté en los fragmentos recuperados.",
    )
    sin_informacion: bool = Field(
        default=False,
        description="true si la documentación NO contiene información para la instrucción.",
    )
    no_documentado: list[str] = Field(
        default_factory=list,
        description=(
            "Datos que pedía la instrucción, que buscaste y que NO figuran en la documentación "
            "(ej. 'severidad de la alerta de PgBouncer'). Vacía si encontraste todo."
        ),
    )


class Calculo(_Contrato):
    descripcion: str = Field(description="Qué se calcula, ej. 'uso del pool sobre el límite'.")
    expresion: str = Field(
        description="Expresión aritmética reproducible que usaste con la tool calcular, ej. '87 / 100 * 100'."
    )
    resultado: float = Field(description="Resultado numérico de la expresión.")


class Analisis(_Contrato):
    """Salida del agente analista."""

    resumen: str = Field(description="Conclusión del análisis en una o dos oraciones.")
    calculos: list[Calculo] = Field(default_factory=list)
    conclusiones: list[str] = Field(
        default_factory=list, description="Conclusiones concretas (ej. 'supera el umbral de alerta del 80%')."
    )
    datos_faltantes: list[str] = Field(
        default_factory=list,
        description=(
            "Datos de la DOCUMENTACIÓN que hicieron falta y no recibiste (el investigador puede "
            "buscarlos). Vacía si no faltó nada. No pongas acá estado en vivo del sistema ni "
            "datos que solo el usuario conoce: eso va en `supuestos`."
        ),
    )
    supuestos: list[str] = Field(
        default_factory=list,
        description=(
            "Supuestos del análisis o datos que solo el usuario puede confirmar (estado en vivo, "
            "cantidad de instancias, etc.). Se informan como limitación; no bloquean la respuesta."
        ),
    )
    inconsistencias: list[str] = Field(
        default_factory=list,
        description="Contradicciones detectadas entre los datos recibidos (vacía si no hay).",
    )


# --- Registros que se guardan en el estado ----------------------------------------


class RegistroInvestigacion(BaseModel):
    """Parcela del investigador: una entrada por cada vez que se le delega."""

    turno: int
    paso: int
    instruccion: str
    resultado: Investigacion | None = None
    # Fuentes que efectivamente devolvieron las tools (metadata de Chroma, no del LLM).
    fuentes_consultadas: list[str] = Field(default_factory=list)
    # Fuentes citadas por el LLM que las tools NO devolvieron (posible alucinación).
    fuentes_no_verificadas: list[str] = Field(default_factory=list)
    error: str | None = None


class VerificacionCalculo(BaseModel):
    expresion: str
    declarado: float
    recalculado: float | None
    ok: bool
    detalle: str = ""


class RegistroAnalisis(BaseModel):
    """Parcela del analista: una entrada por cada vez que se le delega."""

    turno: int
    paso: int
    instruccion: str
    resultado: Analisis | None = None
    # Cada cálculo se vuelve a evaluar en código: el LLM no "certifica" su propia aritmética.
    verificaciones: list[VerificacionCalculo] = Field(default_factory=list)
    error: str | None = None


class Contribucion(BaseModel):
    """Entrada de la bitácora: qué agente aportó qué, en qué turno y paso."""

    turno: int
    paso: int
    agente: Agente
    accion: str
    detalle: str = ""
    fuentes: list[str] = Field(default_factory=list)
    duracion_s: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


# --- Estado del grafo ------------------------------------------------------------


class OrquestadorState(MessagesState):
    """Estado compartido. Hereda `messages` (reducer `add_messages`) de MessagesState.

    `messages` guarda solo la conversación con el usuario (preguntas y respuestas
    finales): el razonamiento interno de cada especialista queda dentro de su propio
    subgrafo ReAct y no contamina el contexto de los demás.
    """

    # Control del turno (los reinicia el nodo `inicio`; los escribe el supervisor).
    turno: int
    pregunta: str
    next_agent: DestinoSupervisor
    instruccion: str
    delegaciones: dict[str, int]
    feedback_validacion: str
    rechazos_validacion: int
    limitaciones: list[str]
    task_completed: bool

    # Parcelas de cada agente (se acumulan entre turnos; cada registro lleva su `turno`).
    investigaciones: Annotated[list[RegistroInvestigacion], operator.add]
    analisis: Annotated[list[RegistroAnalisis], operator.add]
    respuesta_final: str

    # Bitácora de contribuciones de todos los nodos.
    contribuciones: Annotated[list[Contribucion], operator.add]


T = TypeVar("T", RegistroInvestigacion, RegistroAnalisis, Contribucion)


def del_turno(registros: list[T], turno: int) -> list[T]:
    """Filtra los registros del turno actual (las parcelas acumulan todos los turnos)."""
    return [r for r in registros if r.turno == turno]


def siguiente_paso(state: OrquestadorState) -> int:
    """Número de paso dentro del turno: cantidad de contribuciones del turno + 1."""
    return len(del_turno(state.get("contribuciones", []), state["turno"])) + 1


# Tipos que el checkpointer puede deserializar (allowlist explícita: LangGraph bloquea
# por defecto la deserialización de clases no registradas guardadas en un checkpoint).
TIPOS_SERIALIZABLES: list[type[BaseModel]] = [
    Hallazgo,
    Investigacion,
    Calculo,
    Analisis,
    RegistroInvestigacion,
    VerificacionCalculo,
    RegistroAnalisis,
    Contribucion,
]
