"""Ensamblado del StateGraph: nodos, aristas condicionales y checkpointer.

Topología jerárquica con un Supervisor y dos especialistas:

    START → inicio → supervisor ─┬─> investigador ─> supervisor
                                 ├─> analista ─────> supervisor
                                 └─> validacion ─┬─> supervisor   (rechazo: refinar)
                                                 └─> sintetizador → END
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import aiosqlite
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyst_agent import build_analyst_node, build_synthesizer_node
from agents.research_agent import build_research_node
from agents.supervisor import build_supervisor_node, rutear_supervisor, rutear_validacion, validacion
from state import TIPOS_SERIALIZABLES, Contribucion, OrquestadorState
from tools.custom_tools import abrir_vectorstore

# Red de seguridad de LangGraph: techo de super-steps por invocación (los topes
# reales del flujo están en agents/supervisor.py).
RECURSION_LIMIT = 40

# =====================================================================================
# LLM por rol (pre-entregas 1 y 5: proveedor configurable por variables de entorno)
# =====================================================================================

Rol = Literal["supervisor", "investigador", "analista", "sintetizador"]
ROLES: tuple[Rol, ...] = ("supervisor", "investigador", "analista", "sintetizador")

CLAVES_POR_PROVEEDOR = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
MODELOS_POR_DEFECTO = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}


def proveedor_llm() -> str:
    return os.getenv("LLM_PROVIDER", "anthropic").strip().lower()


def verificar_configuracion() -> None:
    """Falla rápido, con un mensaje claro, si el proveedor o su API key no están configurados."""
    provider = proveedor_llm()
    if provider not in CLAVES_POR_PROVEEDOR:
        raise SystemExit(f"✖ LLM_PROVIDER={provider!r} no es válido: usá 'anthropic' u 'openai'.")
    clave = CLAVES_POR_PROVEEDOR[provider]
    if not os.getenv(clave):
        raise SystemExit(f"✖ Falta {clave} (LLM_PROVIDER={provider}). Completala en el .env.")


def modelo_llm(rol: Rol | None = None) -> str:
    """Modelo del rol: LLM_MODEL_<ROL> si está definido; si no, LLM_MODEL; si no, el default."""
    if rol and (especifico := os.getenv(f"LLM_MODEL_{rol.upper()}")):
        return especifico
    return os.getenv("LLM_MODEL") or MODELOS_POR_DEFECTO.get(proveedor_llm(), "")


def build_llm(rol: Rol | None = None) -> BaseChatModel:
    """Crea el LLM del rol según LLM_PROVIDER (anthropic | openai) y LLM_MODEL[_<ROL>]."""
    if proveedor_llm() == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=modelo_llm(rol), temperature=0)
    from langchain_anthropic import ChatAnthropic

    # Los modelos Claude actuales no aceptan `temperature` distinta del default.
    return ChatAnthropic(model_name=modelo_llm(rol))  # type: ignore[call-arg]


@dataclass
class Modelos:
    """Un LLM por rol: permite un supervisor más capaz y especialistas más baratos."""

    supervisor: BaseChatModel
    investigador: BaseChatModel
    analista: BaseChatModel
    sintetizador: BaseChatModel

    @classmethod
    def desde_entorno(cls) -> Modelos:
        return cls(
            supervisor=build_llm("supervisor"),
            investigador=build_llm("investigador"),
            analista=build_llm("analista"),
            sintetizador=build_llm("sintetizador"),
        )


# =====================================================================================
# Grafo
# =====================================================================================


async def inicio(state: OrquestadorState) -> dict[str, Any]:
    """Abre el turno: toma la última pregunta y reinicia los campos de control.

    Las parcelas (`investigaciones`, `analisis`, `contribuciones`) NO se borran: se
    acumulan en el thread y cada registro lleva su número de turno.
    """
    preguntas = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    turno = len(preguntas)
    pregunta = preguntas[-1].text
    return {
        "turno": turno,
        "pregunta": pregunta,
        "instruccion": "",
        "delegaciones": {"investigador": 0, "analista": 0},
        "feedback_validacion": "",
        "rechazos_validacion": 0,
        "limitaciones": [],
        "task_completed": False,
        "respuesta_final": "",
        "contribuciones": [
            Contribucion(turno=turno, paso=1, agente="inicio", accion="recibió la pregunta", detalle=pregunta)
        ],
    }


def build_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    modelos: Modelos | None = None,
    vectorstore: Chroma | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    m = modelos or Modelos.desde_entorno()
    builder = StateGraph(OrquestadorState)
    builder.add_node("inicio", inicio)
    builder.add_node("supervisor", build_supervisor_node(m.supervisor))
    builder.add_node("investigador", build_research_node(m.investigador, vectorstore or abrir_vectorstore()))
    builder.add_node("analista", build_analyst_node(m.analista))
    builder.add_node("validacion", validacion)
    builder.add_node("sintetizador", build_synthesizer_node(m.sintetizador))

    builder.add_edge(START, "inicio")
    builder.add_edge("inicio", "supervisor")
    # El supervisor decide dinámicamente; el Literal de rutear_supervisor son estos destinos.
    builder.add_conditional_edges(
        "supervisor",
        rutear_supervisor,
        {"investigador": "investigador", "analista": "analista", "validacion": "validacion"},
    )
    # Topología jerárquica: los especialistas siempre devuelven el control al supervisor.
    builder.add_edge("investigador", "supervisor")
    builder.add_edge("analista", "supervisor")
    builder.add_conditional_edges(
        "validacion", rutear_validacion, {"supervisor": "supervisor", "sintetizador": "sintetizador"}
    )
    builder.add_edge("sintetizador", END)
    return builder.compile(checkpointer=checkpointer)


# =====================================================================================
# Persistencia (pre-entrega 5)
# =====================================================================================


def serializador() -> JsonPlusSerializer:
    """Serializador del checkpoint con allowlist explícita de los tipos del estado."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[(t.__module__, t.__name__) for t in TIPOS_SERIALIZABLES]
    )


@asynccontextmanager
async def abrir_checkpointer(db_path: str) -> AsyncIterator[AsyncSqliteSaver]:
    """AsyncSqliteSaver con el serializador del estado del orquestador."""
    async with aiosqlite.connect(db_path) as conn:
        saver = AsyncSqliteSaver(conn, serde=serializador())
        await saver.setup()  # crea las tablas si la base es nueva
        yield saver
