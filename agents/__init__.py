"""Agentes del orquestador: supervisor, investigador y analista.

Este módulo reúne lo que comparten los especialistas: cómo se crea su ciclo ReAct,
cómo se les pide la salida estructurada y qué errores no conviene "reintentar".
No importa los submódulos (graph.py los importa directamente) para evitar ciclos.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.warnings import LangGraphDeprecatedSinceV10
from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)

# Pasos internos (LLM + tools) de cada especialista ReAct.
RECURSION_LIMIT_ESPECIALISTA = 12


def crear_agente_react(llm: BaseChatModel, tools: Sequence[BaseTool], prompt: str, name: str) -> Any:
    """`create_react_agent` de LangGraph, como pide la consigna.

    En LangGraph 1.x está deprecado en favor de `langchain.agents.create_agent` (se
    elimina en 2.0); se silencia solo ese aviso. Migrar es cambiar esta función.

    No se usa su `response_format`: ese paso le manda al LLM una conversación que
    termina en un mensaje del asistente, y los modelos Claude actuales la rechazan
    ("does not support assistant message prefill"). La salida estructurada se pide
    aparte, en `ejecutar_especialista`.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=LangGraphDeprecatedSinceV10)
        return create_react_agent(llm, tools, prompt=prompt, name=name)


def salida_estructurada(llm: BaseChatModel, schema: type[M]) -> Runnable[Any, Any]:
    """`with_structured_output` + `with_retry` (pre-entrega 2).

    Con Anthropic se usa `method="json_schema"` (salida estructurada nativa): los
    modelos Claude actuales no admiten el tool calling forzado del método por defecto.
    """
    kwargs: dict[str, Any] = {"method": "json_schema"} if type(llm).__name__ == "ChatAnthropic" else {}
    return llm.with_structured_output(schema, **kwargs).with_retry(stop_after_attempt=2)


PEDIDO_DE_FORMATO = (
    "Con la transcripción anterior, devolvé tu resultado final en el formato estructurado pedido. "
    "No agregues datos que no aparezcan en ella."
)


def transcripcion(mensajes: Sequence[BaseMessage]) -> str:
    """Texto plano del ciclo ReAct: instrucción, resultados de las tools y respuesta final.

    Se usa para pedir la salida estructurada en una llamada aparte. No se reenvían los
    mensajes originales porque pueden traer bloques de razonamiento (thinking) firmados
    para otra lista de tools, que la API de Anthropic rechaza en una llamada distinta.
    """
    partes: list[str] = []
    for m in mensajes:
        if isinstance(m, HumanMessage):
            partes.append(f"INSTRUCCIÓN:\n{m.text}")
        elif isinstance(m, ToolMessage):
            partes.append(f"RESULTADO DE {m.name}:\n{m.text}")
        elif m.text.strip():
            partes.append(f"TU RESPUESTA:\n{m.text}")
    return "\n\n".join(partes)


async def ejecutar_especialista(
    agente: Any, estructurador: Runnable[Any, Any], prompt: str, entrada: str, schema: type[M]
) -> tuple[M, list[BaseMessage]]:
    """Corre el ciclo ReAct del especialista y valida su resultado contra `schema`.

    Devuelve el resultado validado y los mensajes internos del ciclo, que NO pasan
    al estado compartido: solo se usan para auditar fuentes y tools usadas.
    """
    salida = await agente.ainvoke(
        {"messages": [HumanMessage(entrada)]}, {"recursion_limit": RECURSION_LIMIT_ESPECIALISTA}
    )
    mensajes: list[BaseMessage] = salida["messages"]
    crudo = await estructurador.ainvoke(
        [SystemMessage(prompt), HumanMessage(f"{transcripcion(mensajes)}\n\n{PEDIDO_DE_FORMATO}")]
    )
    return schema.model_validate(crudo), mensajes


# Fragmentos del mensaje de error cuando la cuenta del proveedor no tiene saldo
# (Anthropic: 400 "credit balance is too low"; OpenAI: 429 "insufficient_quota").
SIN_SALDO = ("credit balance is too low", "insufficient_quota")


def es_sin_saldo(error: BaseException) -> bool:
    return any(s in str(error) for s in SIN_SALDO)


def es_error_fatal(error: BaseException) -> bool:
    """Credenciales inválidas o cuenta sin saldo: ninguna llamada posterior va a funcionar,
    así que no tiene sentido que el supervisor "reintente" con otro agente."""
    return type(error).__name__ in {"AuthenticationError", "PermissionDeniedError"} or es_sin_saldo(error)
