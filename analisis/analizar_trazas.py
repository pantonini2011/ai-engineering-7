"""Análisis de las trazas de una sesión de Phoenix (Módulo 7).

Uso:
    python analisis/analizar_trazas.py m7-ejercicio-sesion2

Lee los spans del proyecto con el cliente de Phoenix y escribe en analisis/<proyecto>/:
  - resumen.json    métricas por turno, por tipo de span, por nodo y errores
  - spans.csv       los spans con duración, tokens y costo (para revisar a mano)
y muestra un resumen en consola.

Definiciones:
  - Latencia de RED (remota): spans LLM (ChatAnthropic). Medido en la sesión 1, el wrapper
    de LangChain agrega ~4 ms sobre la llamada HTTP pura, así que el span ≈ red + inferencia
    en los servidores de Anthropic.
  - Latencia de PROCESAMIENTO local: embedding de la consulta (CPU), búsqueda en Chroma y el
    resto del tiempo del turno que no está cubierto por llamadas al LLM ni por la
    recuperación (orquestación de LangGraph, validación, parseo, checkpoints SQLite).
  - Costo: tokens × precio de Claude Haiku 5.5 (USD 0,10 / 1M de entrada, 0,50 / 1M de
    salida, prompts < 100K tokens), el mismo que Phoenix tiene en su tabla de modelos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from phoenix.client import Client

PRECIO_ENTRADA = 0.10 / 1_000_000
PRECIO_SALIDA = 0.50 / 1_000_000
NODOS = {"inicio", "supervisor", "investigador", "analista", "validacion", "sintetizador"}
P, C = "attributes.llm.token_count.prompt", "attributes.llm.token_count.completion"


def cargar(proyecto: str) -> pd.DataFrame:
    df = Client().spans.get_spans_dataframe(project_identifier=proyecto, limit=20000, timeout=60)
    df = df.reset_index(drop=True)
    df["ms"] = (df.end_time - df.start_time).dt.total_seconds() * 1000
    for col in (P, C):
        df[col] = df[col].fillna(0) if col in df else 0
    df["costo_usd"] = df[P] * PRECIO_ENTRADA + df[C] * PRECIO_SALIDA
    return df


def nodo_de(df: pd.DataFrame) -> pd.Series:
    """Nodo del grafo (supervisor, investigador...) del que cuelga cada span."""
    padre = dict(zip(df["context.span_id"], df["parent_id"]))
    nombre = dict(zip(df["context.span_id"], df["name"]))

    def buscar(span_id: str) -> str:
        actual: Any = span_id
        while isinstance(actual, str):
            if nombre.get(actual) in NODOS:
                return str(nombre[actual])
            actual = padre.get(actual)
        return "-"

    return df["context.span_id"].map(buscar)


def union_ms(intervalos: list[tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    """Duración de la unión de intervalos (por si hubiera spans solapados)."""
    total, fin_actual, ini_actual = 0.0, None, None
    for ini, fin in sorted(intervalos):
        if fin_actual is None or ini > fin_actual:
            if fin_actual is not None:
                total += (fin_actual - ini_actual).total_seconds()
            ini_actual, fin_actual = ini, fin
        else:
            fin_actual = max(fin_actual, fin)
    if fin_actual is not None:
        total += (fin_actual - ini_actual).total_seconds()
    return total * 1000


def por_turno(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for trace_id, t in df.groupby("context.trace_id"):
        raiz = t[t.parent_id.isna()]
        if raiz.empty:
            continue
        raiz = raiz.iloc[0]
        llm = t[t.span_kind == "LLM"]
        recup = t[t.name == "chroma.retriever"]

        def reloj(spans: pd.DataFrame) -> float:
            # Tiempo de reloj: el investigador lanza búsquedas en paralelo, así que sumar
            # duraciones contaría dos veces el tiempo superpuesto.
            return union_ms(list(zip(spans.start_time, spans.end_time)))

        red, local_rec = reloj(llm), reloj(recup)
        filas.append(
            {
                "trace_id": trace_id,
                "sesion": raiz.get("attributes.session.id"),
                "inicio": raiz.start_time,
                "total_ms": raiz.ms,
                "red_llm_ms": red,
                "recuperacion_ms": local_rec,
                "embedding_ms": reloj(t[t.name == "embedding.consulta"]),
                "chroma_query_ms": reloj(t[t.name == "chroma.query"]),
                "orquestacion_ms": raiz.ms - reloj(pd.concat([llm, recup])),
                "busquedas_en_paralelo": int(recup.ms.sum() > local_rec * 1.05),
                "llamadas_llm": len(llm),
                "busquedas": len(recup),
                "tokens_entrada": int(llm[P].sum()),
                "tokens_salida": int(llm[C].sum()),
                "costo_usd": llm.costo_usd.sum(),
                "errores": int((t.status_code == "ERROR").sum()),
                "status_raiz": raiz.status_code,
            }
        )
    return pd.DataFrame(filas).sort_values("inicio").reset_index(drop=True)


def main(proyecto: str) -> None:
    salida = Path(__file__).parent / proyecto
    salida.mkdir(exist_ok=True)
    df = cargar(proyecto)
    df["nodo"] = nodo_de(df)
    turnos = por_turno(df)

    tipos = (
        df.groupby(["span_kind", "name"])
        .ms.agg(cantidad="count", total_ms="sum", media_ms="mean", p95_ms=lambda s: s.quantile(0.95), max_ms="max")
        .round(1)
        .sort_values("total_ms", ascending=False)
    )
    llm = df[df.span_kind == "LLM"]
    nodos = (
        llm.groupby("nodo")
        .agg(llamadas=("ms", "count"), ms=("ms", "sum"), entrada=(P, "sum"), salida=(C, "sum"), costo_usd=("costo_usd", "sum"))
        .sort_values("entrada", ascending=False)
    )
    # Solo spans de trabajo concreto: los CHAIN/AGENT contenedores solo suman a sus hijos.
    hojas = df[df.span_kind.isin(["LLM", "RETRIEVER", "EMBEDDING", "TOOL"]) | (df.name == "chroma.query")]
    top = hojas.nlargest(10, "ms")[["name", "span_kind", "nodo", "ms", P, C]]
    errores = df[df.status_code == "ERROR"][["name", "span_kind", "nodo", "status_message", "context.trace_id"]]

    # Razonamiento más largo: el turno con más tokens; desglose por llamada al LLM.
    mayor = turnos.loc[turnos.tokens_entrada.add(turnos.tokens_salida).idxmax()]
    llm_mayor = llm[llm["context.trace_id"] == mayor.trace_id].sort_values("start_time")
    detalle_mayor = llm_mayor[["nodo", "ms", P, C, "costo_usd"]].reset_index(drop=True)

    pd.set_option("display.width", 200)
    print(f"Proyecto {proyecto}: {len(df)} spans, {len(turnos)} trazas\n")
    print(turnos.drop(columns=["trace_id", "inicio"]).round(4).to_string(), "\n")
    print("p50/p95 latencia por turno (s):", round(turnos.total_ms.quantile(0.5) / 1000, 1), "/", round(turnos.total_ms.quantile(0.95) / 1000, 1))
    print("\nPor tipo de span:\n", tipos.to_string())
    print("\nLLM por nodo:\n", nodos.round(4).to_string())
    print("\nSpans más lentos:\n", top.round(0).to_string())
    print("\nErrores:\n", errores.to_string())
    print(f"\nRazonamiento más largo: sesión {mayor.sesion} ({mayor.llamadas_llm} llamadas)\n", detalle_mayor.round(4).to_string())

    resumen = {
        "proyecto": proyecto,
        "spans": len(df),
        "trazas": len(turnos),
        "latencia_turno_s": {
            "p50": turnos.total_ms.quantile(0.5) / 1000,
            "p95": turnos.total_ms.quantile(0.95) / 1000,
            "max": turnos.total_ms.max() / 1000,
        },
        "totales": {
            "tokens_entrada": int(turnos.tokens_entrada.sum()),
            "tokens_salida": int(turnos.tokens_salida.sum()),
            "costo_usd": turnos.costo_usd.sum(),
            "red_llm_ms": turnos.red_llm_ms.sum(),
            "recuperacion_ms": turnos.recuperacion_ms.sum(),
            "orquestacion_ms": turnos.orquestacion_ms.sum(),
        },
        "turnos": json.loads(turnos.to_json(orient="records", date_format="iso")),
        "por_tipo": json.loads(tipos.reset_index().to_json(orient="records")),
        "llm_por_nodo": json.loads(nodos.reset_index().to_json(orient="records")),
        "spans_mas_lentos": json.loads(top.to_json(orient="records")),
        "errores": json.loads(errores.to_json(orient="records")),
        "razonamiento_mas_largo": {
            "sesion": mayor.sesion,
            "trace_id": mayor.trace_id,
            "llamadas": json.loads(detalle_mayor.to_json(orient="records")),
        },
    }
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    cols = ["context.trace_id", "context.span_id", "parent_id", "name", "span_kind", "nodo", "status_code", "ms", P, C, "costo_usd"]
    df[cols].to_csv(salida / "spans.csv", index=False)
    print(f"\n→ {salida}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m7-ejercicio-sesion2")
