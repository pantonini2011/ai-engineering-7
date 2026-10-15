"""Observabilidad (Módulo 7): trazas OpenInference → Arize Phoenix local.

Dos capas de instrumentación:

1. Automática (`LangChainInstrumentor`): grafo de LangGraph, nodos, agentes ReAct,
   tools y llamadas al LLM (con tokens de entrada/salida, que Phoenix usa para el costo).
2. Manual, sobre la base vectorial: `buscar_documentacion` consulta Chroma directo
   (no a través de un Retriever de LangChain), así que el instrumentador no la ve.
   Sin estos spans la recuperación sería una "mancha ciega" en la traza:

       TOOL buscar_documentacion
       └── RETRIEVER chroma.retriever          (documentos + similitud)
           ├── EMBEDDING embedding.consulta    (CPU local: sentence-transformers)
           └── CHAIN chroma.query              (búsqueda HNSW en Chroma)

   Separar embedding y query permite distinguir la latencia de procesamiento local
   de la latencia de red de las llamadas a la API del LLM.

Si no se llama a `configurar_trazas`, OpenTelemetry usa un tracer no-op: los spans
manuales no cuestan nada (tests, ejecuciones sin Phoenix).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.trace import Span

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

_tracer = trace.get_tracer("orquestador.vectorstore")

KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND

# Falla provocada para el ejercicio: con esta variable en "1" la base vectorial
# "se desconecta" y buscar_documentacion lanza un error que queda registrado en la traza.
VARIABLE_CAIDA_VECTORSTORE = "SIMULAR_CAIDA_VECTORSTORE"


def configurar_trazas(proyecto: str) -> TracerProvider:
    """Registra el exportador OTLP hacia Phoenix e instrumenta LangChain/LangGraph.

    El endpoint sale de PHOENIX_COLLECTOR_ENDPOINT (por defecto http://localhost:6006).
    Cada sesión de pruebas usa su propio proyecto para no mezclar métricas (p95, costos).
    """
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    # El SDK de Anthropic (1.x) emite su propio span por llamada HTTP. Como el instrumentador
    # de LangChain no deja sus spans en el contexto de OpenTelemetry, ese span queda huérfano:
    # una traza aparte por llamada, que duplica tokens y costo y distorsiona el p95. El span
    # ChatAnthropic de LangChain ya mide esa llamada (overhead medido: ~4 ms), así que se apaga.
    # Debe hacerse antes de crear los clientes (Modelos.desde_entorno).
    os.environ.setdefault("ANTHROPIC_OPEN_TELEMETRY", "false")
    provider = register(project_name=proyecto, batch=True, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    return provider


def vectorstore_caido() -> bool:
    return os.getenv(VARIABLE_CAIDA_VECTORSTORE) == "1"


@contextmanager
def span_retriever(consulta: str, top_k: int, filtro: dict[str, Any] | None) -> Iterator[Span]:
    """Span RETRIEVER de una búsqueda en Chroma. Las excepciones quedan como status ERROR.

    El instrumentador de LangChain no deja sus spans en el contexto de OpenTelemetry, así
    que el padre (el span TOOL de buscar_documentacion) se toma explícitamente con
    `get_current_span()`; sin esto el RETRIEVER sería la raíz de una traza aparte.
    """
    from openinference.instrumentation.langchain import get_current_span

    padre = get_current_span()
    with _tracer.start_as_current_span(
        "chroma.retriever",
        context=trace.set_span_in_context(padre) if padre else None,
        attributes={
            KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
            SpanAttributes.INPUT_VALUE: consulta,
            SpanAttributes.METADATA: json.dumps({"top_k": top_k, "filtro": filtro}, ensure_ascii=False),
        },
    ) as span:
        yield span


@contextmanager
def span_embedding(modelo: str, texto: str) -> Iterator[Span]:
    """Span EMBEDDING del vector de la consulta (cómputo local, sin red)."""
    prefijo = f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0"
    with _tracer.start_as_current_span(
        "embedding.consulta",
        attributes={
            KIND: OpenInferenceSpanKindValues.EMBEDDING.value,
            SpanAttributes.EMBEDDING_MODEL_NAME: modelo,
            f"{prefijo}.{EmbeddingAttributes.EMBEDDING_TEXT}": texto,
        },
    ) as span:
        yield span


@contextmanager
def span_query_chroma(coleccion: str) -> Iterator[Span]:
    """Span de la búsqueda de vecinos en el índice HNSW de Chroma."""
    with _tracer.start_as_current_span(
        "chroma.query",
        attributes={KIND: OpenInferenceSpanKindValues.CHAIN.value, "db.system": "chromadb", "db.collection": coleccion},
    ) as span:
        yield span


def registrar_documentos(span: Span, pares: Sequence[tuple[Document, float]]) -> None:
    """Documentos recuperados con su similitud (1 - distancia coseno), en formato OpenInference."""
    for i, (doc, distancia) in enumerate(pares):
        prefijo = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}"
        span.set_attribute(f"{prefijo}.{DocumentAttributes.DOCUMENT_CONTENT}", doc.page_content)
        span.set_attribute(f"{prefijo}.{DocumentAttributes.DOCUMENT_SCORE}", round(1 - distancia, 3))
        span.set_attribute(
            f"{prefijo}.{DocumentAttributes.DOCUMENT_METADATA}", json.dumps(doc.metadata, ensure_ascii=False)
        )
        if doc.id:
            span.set_attribute(f"{prefijo}.{DocumentAttributes.DOCUMENT_ID}", doc.id)
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"{len(pares)} fragmentos recuperados")
