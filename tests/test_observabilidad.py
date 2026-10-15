"""Tests offline de la instrumentación (Módulo 7): jerarquía de spans y falla de la base vectorial.

Usa un exportador en memoria en lugar de Phoenix: verifica que la recuperación en Chroma
quede DENTRO de la traza (TOOL → RETRIEVER → EMBEDDING / chroma.query), no como una
traza aparte, y que una caída de la base quede registrada como span con status ERROR.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import observability
from observability import VARIABLE_CAIDA_VECTORSTORE
from test_orquestador import PREGUNTA, correr, flujo_feliz


@pytest.fixture(scope="module")
def exportador() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer_original = observability._tracer
    observability._tracer = provider.get_tracer("orquestador.vectorstore")
    LangChainInstrumentor().instrument(tracer_provider=provider)
    yield exporter
    LangChainInstrumentor().uninstrument()
    observability._tracer = tracer_original


@pytest.fixture
def spans(exportador: InMemorySpanExporter) -> Iterator[InMemorySpanExporter]:
    exportador.clear()
    yield exportador


def _padre(span: ReadableSpan, por_id: dict[int, ReadableSpan]) -> str | None:
    if span.parent is None:
        return None
    padre = por_id.get(span.parent.span_id)
    return padre.name if padre else "(desconocido)"


def _por_nombre(spans: tuple[ReadableSpan, ...], nombre: str) -> list[ReadableSpan]:
    return [s for s in spans if s.name == nombre]


class TestJerarquiaDeSpans:
    def test_recuperacion_cuelga_de_la_tool_en_la_misma_traza(
        self, tmp_path: Path, vectorstore: Chroma, spans: InMemorySpanExporter
    ) -> None:
        correr(tmp_path, flujo_feliz(), vectorstore, [PREGUNTA])
        todos = spans.get_finished_spans()
        por_id = {s.context.span_id: s for s in todos}

        [retriever] = _por_nombre(todos, "chroma.retriever")
        assert _padre(retriever, por_id) == "buscar_documentacion"
        assert _padre(_por_nombre(todos, "embedding.consulta")[0], por_id) == "chroma.retriever"
        assert _padre(_por_nombre(todos, "chroma.query")[0], por_id) == "chroma.retriever"
        # Una sola traza por turno: todo cuelga de la raíz del grafo.
        assert len({s.context.trace_id for s in todos}) == 1
        assert [s.name for s in todos if s.parent is None] == ["LangGraph"]

    def test_retriever_registra_documentos_y_similitud(
        self, tmp_path: Path, vectorstore: Chroma, spans: InMemorySpanExporter
    ) -> None:
        correr(tmp_path, flujo_feliz(), vectorstore, [PREGUNTA])
        [retriever] = _por_nombre(spans.get_finished_spans(), "chroma.retriever")
        attrs = retriever.attributes or {}
        assert attrs["openinference.span.kind"] == "RETRIEVER"
        assert attrs["retrieval.documents.0.document.content"]
        assert "runbook_incidentes.md" in str(attrs["retrieval.documents.0.document.metadata"])
        assert isinstance(attrs["retrieval.documents.0.document.score"], float)


class TestFallaDeLaBaseVectorial:
    def test_caida_queda_como_error_en_la_traza(
        self, tmp_path: Path, vectorstore: Chroma, spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(VARIABLE_CAIDA_VECTORSTORE, "1")
        correr(tmp_path, flujo_feliz(), vectorstore, [PREGUNTA])
        todos = spans.get_finished_spans()
        [retriever] = _por_nombre(todos, "chroma.retriever")
        assert retriever.status.status_code == StatusCode.ERROR
        assert any(e.name == "exception" for e in retriever.events)
        # No llegó a calcular embeddings ni a consultar el índice.
        assert not _por_nombre(todos, "embedding.consulta")
        assert _por_nombre(todos, "buscar_documentacion")[0].status.status_code == StatusCode.ERROR
