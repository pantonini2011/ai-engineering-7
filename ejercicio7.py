"""Ejercicio del Módulo 7: instrumentación y análisis de trazas con Phoenix.

Script reproducible, en tres pasos:

1. Inicialización limpia de Phoenix con `px.launch_app()` (base temporal: solo quedan
   las trazas de esta corrida), en un proceso aparte.
2. `TracerProvider` de OpenTelemetry que exporta por OTLP a esa instancia, y
   `LangChainInstrumentor` configurado con ese provider.
3. Tres consultas variadas:
   - simple: una sola llamada al LLM, sin recuperación;
   - con recuperación documental: el orquestador busca en ChromaDB y calcula;
   - caso de error: la misma clase de consulta con la base vectorial "desconectada".

Al final imprime un resumen por traza (latencia, tokens, costo), lo guarda en
`analisis/ejercicio7/` y deja Phoenix abierto para sacar las capturas (Ctrl+C para cerrar).

Uso:
    python ejercicio7.py              # corre, resume y espera
    python ejercicio7.py --sin-esperar
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
import os
import sys
import time
from multiprocessing.synchronize import Event as EventoProceso
from pathlib import Path

from dotenv import load_dotenv

# El SDK de Anthropic 1.x trae su propio trazado OpenTelemetry: con un provider global
# genera un span huérfano por llamada que duplica tokens y costo. El span de LangChain
# ya mide esa llamada, así que se apaga. Tiene que ir antes de crear los clientes.
os.environ.setdefault("ANTHROPIC_OPEN_TELEMETRY", "false")

import phoenix as px  # noqa: E402
from phoenix.client import Client  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from openinference.instrumentation import using_attributes  # noqa: E402
from openinference.instrumentation.langchain import LangChainInstrumentor  # noqa: E402
from openinference.semconv.resource import ResourceAttributes  # noqa: E402
from opentelemetry import trace  # noqa: E402
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: E402
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: E402

from graph import abrir_checkpointer, build_graph, build_llm, verificar_configuracion  # noqa: E402
from main import _configurar_logs, ejecutar_turno  # noqa: E402
from observability import VARIABLE_CAIDA_VECTORSTORE  # noqa: E402
from tools.custom_tools import ingestar  # noqa: E402

PROYECTO = "ejercicio-7"
DB_PATH = "checkpoints.db"
SALIDA = Path("analisis") / "ejercicio7"
# USD por millón de tokens (claude-haiku-5-5, prompts < 100K).
PRECIO_ENTRADA, PRECIO_SALIDA = 0.10, 0.50

CONSULTA_SIMPLE = "En una oración: ¿para qué sirve un pool de conexiones como PgBouncer?"
CONSULTA_RECUPERACION = (
    "ej7-recuperacion",
    "Hoy PgBouncer tiene 87 conexiones activas sostenidas hace 3 minutos. ¿Ya debería haber saltado "
    "la alerta del pool de PostgreSQL? ¿A qué porcentaje del límite estamos y qué indica el runbook?",
)
CONSULTA_ERROR = (
    "ej7-error",
    "¿Cuál es el umbral de replication lag de las réplicas de lectura y cuánto lo superamos si hoy "
    "medimos 14 segundos?",
)


def servidor_phoenix(listo: EventoProceso, cerrar: EventoProceso) -> None:
    """Proceso hijo: inicialización limpia de Phoenix (base temporal, vacía en cada corrida).

    Va en su propio proceso porque, en un hilo del mismo proceso que el agente, el servidor
    compite por el GIL y deja de aceptar spans (responde 503). `launch_app(run_in_thread=False)`
    haría lo mismo, pero no funciona en Windows.
    """
    if px.launch_app(use_temp_dir=True) is None:
        return
    listo.set()
    try:
        cerrar.wait()
    except KeyboardInterrupt:
        pass
    px.close_app()


def configurar_tracer(url: str) -> TracerProvider:
    """TracerProvider propio que exporta a Phoenix + instrumentador de LangChain con ese provider."""
    provider = TracerProvider(resource=Resource({ResourceAttributes.PROJECT_NAME: PROYECTO}))
    # Exportación en lote, en un hilo aparte: no frena al agente mientras Phoenix procesa.
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(f"{url}v1/traces", timeout=30)))
    # Global: los spans manuales de observability.py (RETRIEVER, EMBEDDING) usan este provider.
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    return provider


async def consulta_simple() -> None:
    """Una sola llamada al LLM: la traza es un único span LLM (línea de base de latencia)."""
    print(f"\n[simple] {CONSULTA_SIMPLE}")
    with using_attributes(session_id="ej7-simple", tags=["simple"]):
        respuesta = await build_llm("sintetizador").ainvoke(
            [SystemMessage("Respondé en español, breve."), HumanMessage(CONSULTA_SIMPLE)]
        )
    print(f"→ {respuesta.text.strip()}")


async def consultas_orquestador() -> None:
    async with abrir_checkpointer(DB_PATH) as saver:
        for thread_id, _ in (CONSULTA_RECUPERACION, CONSULTA_ERROR):
            await saver.adelete_thread(thread_id)  # cada corrida arranca sin memoria previa
        graph = build_graph(saver)
        await ejecutar_turno(graph, *CONSULTA_RECUPERACION)
        os.environ[VARIABLE_CAIDA_VECTORSTORE] = "1"  # falla provocada: Chroma "no responde"
        try:
            await ejecutar_turno(graph, *CONSULTA_ERROR)
        finally:
            os.environ.pop(VARIABLE_CAIDA_VECTORSTORE, None)


def resumir(url: str) -> None:
    """Latencia, tokens y costo por traza, más el desglose LLM / recuperación."""
    df = Client(base_url=url).spans.get_spans_dataframe(project_identifier=PROYECTO, limit=5000, timeout=60)
    SALIDA.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[c for c in df.columns if df[c].map(type).eq(dict).any()]).to_csv(SALIDA / "spans.csv")
    df["ms"] = (df.end_time - df.start_time).dt.total_seconds() * 1000
    raices = df[df.parent_id.isna()].sort_values("start_time")
    lineas = ["traza | spans | latencia (s) | LLM (s) | recuperación (s) | tokens in/out | USD"]
    for _, raiz in raices.iterrows():
        t = df[df["context.trace_id"] == raiz["context.trace_id"]]
        llm = t[t.span_kind == "LLM"]
        ret = t[t.span_kind == "RETRIEVER"]
        tin = int(llm["attributes.llm.token_count.prompt"].fillna(0).sum())
        tout = int(llm["attributes.llm.token_count.completion"].fillna(0).sum())
        usd = (tin * PRECIO_ENTRADA + tout * PRECIO_SALIDA) / 1e6
        errores = int((t.status_code == "ERROR").sum())
        nombre = raiz.get("attributes.session.id") or raiz["name"]
        lineas.append(
            f"{nombre} | {len(t)} | {raiz.ms / 1000:.1f} | {llm.ms.sum() / 1000:.1f} | "
            f"{ret.ms.sum() / 1000:.2f} | {tin}/{tout} | {usd:.4f}" + (f" | {errores} spans ERROR" if errores else "")
        )
    texto = "\n".join(lineas)
    (SALIDA / "resumen.txt").write_text(texto + "\n", encoding="utf-8")
    print("\n" + texto + f"\n\n→ {SALIDA}/")


async def correr() -> None:
    await consulta_simple()
    await consultas_orquestador()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sin-esperar", action="store_true", help="cierra Phoenix al terminar")
    args = parser.parse_args()
    load_dotenv()
    _configurar_logs()
    verificar_configuracion()
    ingestar(forzar=False)  # indexa data/docs en Chroma solo si la colección está vacía

    listo, cerrar = mp.Event(), mp.Event()
    servidor = mp.Process(target=servidor_phoenix, args=(listo, cerrar))
    servidor.start()
    if not listo.wait(timeout=180):
        servidor.terminate()
        sys.exit("✖ No se pudo iniciar Phoenix (¿el puerto 6006 está ocupado por otro Phoenix?).")
    url = f"http://localhost:{os.getenv('PHOENIX_PORT', '6006')}/"
    provider = configurar_tracer(url)
    print(f"Phoenix: {url} · proyecto '{PROYECTO}'")
    asyncio.run(correr())
    provider.force_flush()
    time.sleep(2)  # que Phoenix termine de persistir los últimos spans
    resumir(url)

    if not args.sin_esperar:
        print(f"\nPhoenix sigue abierto en {url} para las capturas. Ctrl+C para cerrar.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    provider.shutdown()
    cerrar.set()
    servidor.join(timeout=30)


if __name__ == "__main__":
    main()
