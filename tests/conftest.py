"""Fixtures offline: LLMs falsos con respuestas guionadas y una Chroma con embeddings falsos.

Ningún test usa API keys ni descarga modelos: el grafo, los nodos, las tools, la
base vectorial (Chroma) y el checkpointer SQLite son los reales.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, PrivateAttr

from graph import Modelos
from tools.custom_tools import DOCS_DIR, ingestar


class FakeLLM(GenericFakeChatModel):
    """LLM falso: devuelve AIMessages guionados (texto o tool_calls) y, para
    with_structured_output, objetos Pydantic guionados. Registra lo que recibe
    para poder verificar el aislamiento de contexto."""

    _estructurados: list[Any] = PrivateAttr(default_factory=list)
    _recibidos: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeLLM:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        def responder(mensajes: Any) -> Any:
            self._recibidos.append(list(mensajes))
            salida = self._estructurados.pop(0)
            if isinstance(salida, Exception):
                raise salida
            return salida

        return RunnableLambda(responder)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self._recibidos.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    @property
    def recibidos(self) -> list[list[BaseMessage]]:
        return self._recibidos


def fake_llm(mensajes: Sequence[AIMessage] | None = None, estructurados: Sequence[Any] | None = None) -> FakeLLM:
    it: Iterator[AIMessage] = iter(list(mensajes or []))
    llm = FakeLLM(messages=it)
    llm._estructurados = list(estructurados or [])
    return llm


def tool_call(nombre: str, args: dict[str, Any], i: int = 0) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": nombre, "args": args, "id": f"call_{nombre}_{i}"}])


def modelos(
    supervisor: Sequence[BaseModel | Exception],
    investigador: tuple[Sequence[AIMessage], Sequence[BaseModel]] = ([], []),
    analista: tuple[Sequence[AIMessage], Sequence[BaseModel]] = ([], []),
    sintetizador: Sequence[AIMessage] | None = None,
) -> Modelos:
    return Modelos(
        supervisor=fake_llm(estructurados=list(supervisor)),
        investigador=fake_llm(*investigador),
        analista=fake_llm(*analista),
        sintetizador=fake_llm(list(sintetizador or [AIMessage("Respuesta final.")])),
    )


@pytest.fixture(scope="session")
def vectorstore(tmp_path_factory: pytest.TempPathFactory) -> Chroma:
    """Chroma real sobre los documentos reales, con embeddings determinísticos (sin red)."""
    directorio: Path = tmp_path_factory.mktemp("chroma")
    return ingestar(
        directorio=DOCS_DIR,
        persist_directory=str(directorio),
        collection_name="tests",
        embeddings=DeterministicFakeEmbedding(size=64),
        contar_tokens=lambda texto: len(texto) // 4,
    )
