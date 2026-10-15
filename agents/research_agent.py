"""Especialista 1 — Agente de Búsqueda/Investigación.

Agente ReAct (`create_react_agent`) con herramientas ACOTADAS a la lectura de la
base vectorial (`tools/custom_tools.py`): no calcula ni recomienda (eso es del
analista). Recibe solo la instrucción del supervisor, no la conversación completa.

Escribe únicamente en su parcela del estado: `investigaciones`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel

from agents import crear_agente_react, ejecutar_especialista, es_error_fatal, salida_estructurada
from state import Contribucion, Investigacion, OrquestadorState, RegistroInvestigacion, siguiente_paso
from tools.custom_tools import build_research_tools, fuentes_consultadas, tools_usadas

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Sos el AGENTE INVESTIGADOR de un equipo de análisis técnico.
Tu única fuente es la documentación interna de la "Plataforma de Pedidos"
(arquitectura, monitoreo y alertas, normativa de despliegue, runbook de incidentes),
que consultás con la herramienta buscar_documentacion.

Reglas:
- Buscá antes de responder. Si la instrucción tiene varias partes, hacé una búsqueda por parte.
- Reportá SOLO datos que aparezcan en los fragmentos recuperados, cada uno con su `fuente`
  exactamente como la devolvió la herramienta. Copiá las cifras tal cual (umbrales, límites, tiempos).
- No hagas cálculos, comparaciones ni recomendaciones propias: eso lo hace el analista.
- Si la instrucción pide un dato que buscaste y no figura, listalo en no_documentado.
- Si ningún fragmento responde la instrucción, devolvé sin_informacion=true y hallazgos vacíos.
  Nunca completes con conocimiento general."""


def build_research_agent(llm: BaseChatModel, vectorstore: Chroma) -> Any:
    """Subgrafo ReAct del investigador (búsqueda en la base vectorial)."""
    return crear_agente_react(llm, build_research_tools(vectorstore), SYSTEM_PROMPT, "investigador")


def build_research_node(llm: BaseChatModel, vectorstore: Chroma) -> Any:
    agente = build_research_agent(llm, vectorstore)
    estructurador = salida_estructurada(llm, Investigacion)

    async def investigador(state: OrquestadorState) -> dict[str, Any]:
        turno, paso, instruccion = state["turno"], siguiente_paso(state), state["instruccion"]
        inicio = time.perf_counter()
        registro = RegistroInvestigacion(turno=turno, paso=paso, instruccion=instruccion)
        usadas: list[str] = []
        try:
            # Aislamiento de contexto: el especialista recibe SOLO la instrucción del supervisor.
            resultado, mensajes = await ejecutar_especialista(
                agente, estructurador, SYSTEM_PROMPT, instruccion, Investigacion
            )
            consultadas = fuentes_consultadas(mensajes)
            usadas = tools_usadas(mensajes)
            registro.resultado = resultado
            registro.fuentes_consultadas = consultadas
            registro.fuentes_no_verificadas = sorted(
                {h.fuente for h in resultado.hallazgos} - set(consultadas)
            )
        except Exception as e:  # el error queda en el estado y decide el supervisor
            if es_error_fatal(e):
                raise
            log.warning("investigador falló: %s", e)
            registro.error = f"{type(e).__name__}: {e}"

        if registro.resultado:
            r = registro.resultado
            detalle = "sin información en la documentación" if r.sin_informacion else r.resumen
            accion = f"investigó ({len(r.hallazgos)} hallazgos)"
        else:
            detalle, accion = registro.error or "", "falló"
        contribucion = Contribucion(
            turno=turno,
            paso=paso,
            agente="investigador",
            accion=accion,
            detalle=detalle + (f" · tools: {', '.join(usadas)}" if usadas else ""),
            fuentes=registro.fuentes_consultadas,
            duracion_s=round(time.perf_counter() - inicio, 2),
        )
        return {"investigaciones": [registro], "contribuciones": [contribucion]}

    return investigador
