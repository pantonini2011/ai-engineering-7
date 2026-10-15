"""Herramientas (@tool) de los especialistas y la base vectorial que consultan.

- Investigador: `buscar_documentacion` y `listar_documentos`, sobre la base vectorial
  ChromaDB de la pre-entrega 3 (la "búsqueda sobre tu Vector DB de pre-entregas
  anteriores" que propone la consigna).
- Analista: `calcular`, `porcentaje` y `comparar_con_umbral`.

Cada especialista recibe SOLO sus tools: el investigador no puede calcular y el
analista no puede buscar.

Prácticas heredadas de la pre-entrega 5: `args_schema` Pydantic con `extra="forbid"`,
docstrings que dicen cuándo usar cada tool y cuándo NO, argumentos inválidos que
vuelven como observación (la tool no se ejecuta) y timeout en la consulta a la base.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import math
import operator
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.v1 import ValidationError as ValidationErrorV1

from observability import (
    registrar_documentos,
    span_embedding,
    span_query_chroma,
    span_retriever,
    vectorstore_caido,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =====================================================================================
# Contratos comunes de las tools
# =====================================================================================


class EntradaTool(BaseModel):
    """Base de los args_schema: sin campos extra y con espacios recortados."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _error_de_validacion(error: ValidationError | ValidationErrorV1) -> str:
    """Traduce el ValidationError a un mensaje que el LLM pueda usar para corregir los argumentos."""
    detalles = "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'argumentos'}: {e['msg']}" for e in error.errors()
    )
    return json.dumps(
        {"error": f"Argumentos inválidos ({detalles}). Corregilos y reintentá."}, ensure_ascii=False
    )


def preparar_tools(tools: Sequence[BaseTool]) -> list[BaseTool]:
    """Si el LLM manda argumentos inválidos, la tool no se ejecuta y el error vuelve como
    observación (en vez de lanzar), así el especialista corrige y reintenta."""
    for t in tools:
        t.handle_validation_error = _error_de_validacion
    return list(tools)


def observaciones(mensajes: Sequence[BaseMessage], tool_name: str) -> list[dict[str, Any]]:
    """Resultados (JSON) que devolvió una tool dentro del ciclo ReAct de un especialista."""
    salida: list[dict[str, Any]] = []
    for m in mensajes:
        if isinstance(m, ToolMessage) and m.name == tool_name:
            try:
                contenido = json.loads(str(m.content))
            except json.JSONDecodeError:
                continue
            if isinstance(contenido, dict):
                salida.append(contenido)
    return salida


def tools_usadas(mensajes: Sequence[BaseMessage]) -> list[str]:
    return [m.name for m in mensajes if isinstance(m, ToolMessage) and m.name]


# =====================================================================================
# Base vectorial ChromaDB (pre-entrega 3)
# =====================================================================================
#
# Mantiene las decisiones de la pre-entrega 3: fragmentación jerárquica por
# encabezado Markdown, el mismo modelo de embeddings para indexar y consultar,
# ingesta idempotente y la fuente de cada fragmento tomada de la metadata.

DOCS_DIR = PROJECT_ROOT / "data" / "docs"
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "vectorstore")
CHROMA_COLLECTION = "plataforma_pedidos"
# Multilingüe porque los documentos y las preguntas están en español.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Fragmentos por búsqueda: entre 3 y 5 para evitar el "contexto infinito".
TOP_K = 5

# El modelo de embeddings solo lee los primeros 128 tokens de cada texto
# (max_seq_length) e ignora el resto. Con fragmentos de 500 tokens (los de la
# pre-entrega 3), la regla de rollback del canario, al final de una sección de
# 282 tokens, quedaba fuera del embedding y la búsqueda no la encontraba.
# Por eso el tamaño se ajusta al modelo, y cada subfragmento lleva adelante la
# ruta de encabezados para no perder el contexto de su sección.
MAX_TOKENS_MODELO = 128
CHUNK_SIZE = 100  # tokens del cuerpo; la ruta de encabezados ocupa el resto
CHUNK_OVERLAP = 15  # 15%
HEADERS = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]


@lru_cache(maxsize=1)
def build_embeddings() -> Embeddings:
    """Embeddings locales (sin API key). Se cargan una sola vez por proceso."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _tokenizer_len() -> Callable[[str], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return lambda texto: len(tokenizer.encode(texto, add_special_tokens=False))


def ruta_de_encabezados(metadata: dict[str, object]) -> str:
    """'Documento > Sección > Subsección' a partir de la metadata del splitter."""
    return " > ".join(str(metadata[clave]) for _, clave in HEADERS if metadata.get(clave))


def fragmentar(documento: Document, contar_tokens: Callable[[str], int]) -> list[Document]:
    """Corta por encabezado Markdown; si una sección no entra en MAX_TOKENS_MODELO, la
    subdivide y antepone a cada parte la ruta de encabezados de la sección."""
    secciones = MarkdownHeaderTextSplitter(HEADERS, strip_headers=True).split_text(
        documento.page_content
    )
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=contar_tokens
    )
    fragmentos: list[Document] = []
    for seccion in secciones:
        metadata = {**documento.metadata, **seccion.metadata}
        ruta = f"[{ruta_de_encabezados(metadata)}]\n" if seccion.metadata else ""
        cuerpo = seccion.page_content
        partes = [cuerpo] if contar_tokens(ruta + cuerpo) <= MAX_TOKENS_MODELO else fallback.split_text(cuerpo)
        fragmentos.extend(Document(page_content=ruta + p, metadata=metadata) for p in partes)
    return fragmentos


def _cargar_documentos(directorio: Path) -> list[Document]:
    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in sorted(directorio.glob("*.md"))
    ]


def abrir_vectorstore(
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION,
    embeddings: Embeddings | None = None,
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings or build_embeddings(),
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )


def ingestar(
    directorio: Path = DOCS_DIR,
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION,
    embeddings: Embeddings | None = None,
    contar_tokens: Callable[[str], int] | None = None,
    forzar: bool = False,
) -> Chroma:
    """Indexa los .md de `directorio` en Chroma. Idempotente salvo `forzar=True`."""
    vectorstore = abrir_vectorstore(persist_directory, collection_name, embeddings)
    existentes = len(vectorstore.get(include=[])["ids"])
    if existentes and not forzar:
        log.info("Colección '%s' ya poblada (%d fragmentos): se omite la ingesta.", collection_name, existentes)
        return vectorstore
    if existentes:
        vectorstore.reset_collection()

    documentos = _cargar_documentos(directorio)
    if not documentos:
        raise ValueError(f"No hay archivos .md en '{directorio}'.")
    contar = contar_tokens or _tokenizer_len()
    fragmentos = [f for d in documentos for f in fragmentar(d, contar)]
    vectorstore.add_documents(fragmentos)
    log.info("Ingesta: %d fragmentos de %d documentos en '%s'.", len(fragmentos), len(documentos), collection_name)
    return vectorstore


def documentos_indexados(vectorstore: Chroma) -> dict[str, list[str]]:
    """{archivo: [secciones]} de lo indexado, para que el investigador sepa qué hay."""
    indice: dict[str, list[str]] = {}
    for meta in vectorstore.get(include=["metadatas"])["metadatas"] or []:
        secciones = indice.setdefault(str(meta.get("source", "?")), [])
        titulo = meta.get("Header 2") or meta.get("Header 1")
        if titulo and str(titulo) not in secciones:
            secciones.append(str(titulo))
    return dict(sorted(indice.items()))


# =====================================================================================
# Tools del investigador
# =====================================================================================

TIMEOUT_BUSQUEDA_S = 30.0
LARGO_MAXIMO_FRAGMENTO = 1500  # caracteres por fragmento devuelto al LLM


class BuscarDocumentacionInput(EntradaTool):
    consulta: str = Field(
        min_length=3,
        max_length=300,
        description="Qué buscar, en lenguaje natural. Ej.: 'umbral de alerta de conexiones de PgBouncer'.",
    )
    documento: str | None = Field(
        default=None,
        description="Opcional: limitar la búsqueda a un archivo, ej. 'runbook_incidentes.md'.",
    )


class ListarDocumentosInput(EntradaTool):
    pass


async def buscar_trazado(
    vectorstore: Chroma, consulta: str, filtro: dict[str, Any] | None
) -> list[tuple[Document, float]]:
    """Búsqueda en Chroma en dos pasos (embedding de la consulta + query al índice), cada
    uno con su span, para medir por separado el cómputo local y la búsqueda (Módulo 7)."""
    with span_retriever(consulta, TOP_K, filtro) as span:
        if vectorstore_caido():
            raise ConnectionError("Chroma no responde: base vectorial desconectada (falla simulada).")
        embeddings = vectorstore.embeddings
        assert embeddings is not None
        with span_embedding(getattr(embeddings, "model_name", type(embeddings).__name__), consulta):
            vector = await asyncio.to_thread(embeddings.embed_query, consulta)
        with span_query_chroma(vectorstore._collection.name):
            pares = await asyncio.to_thread(
                vectorstore.similarity_search_by_vector_with_relevance_scores, vector, k=TOP_K, filter=filtro
            )
        registrar_documentos(span, pares)
        return pares


def build_research_tools(vectorstore: Chroma) -> list[BaseTool]:
    """Tools del investigador, ligadas a una base vectorial concreta."""

    @tool(args_schema=BuscarDocumentacionInput)
    async def buscar_documentacion(consulta: str, documento: str | None = None) -> dict[str, Any]:
        """Búsqueda semántica en la documentación técnica de la Plataforma de Pedidos
        (base vectorial ChromaDB). Devuelve los fragmentos más parecidos a la consulta.

        Usala para encontrar umbrales, límites, procedimientos, causas de incidentes y
        reglas de despliegue. Hacé una búsqueda por cada tema distinto que necesites.

        NO la uses para hacer cuentas ni para temas ajenos a la plataforma (cotizaciones,
        noticias, precios de mercado): la base no tiene esa información.

        Returns:
            {"resultados": [{"fuente": str, "seccion": str, "similitud": float, "texto": str}, ...]}.
            Similitud cercana a 1 = muy relevante; por debajo de ~0.3 suele ser ruido.
            {"error": str} si la base no respondió a tiempo.
        """
        filtro = {"source": documento} if documento else None
        try:
            pares = await asyncio.wait_for(buscar_trazado(vectorstore, consulta, filtro), timeout=TIMEOUT_BUSQUEDA_S)
        except TimeoutError:
            return {"error": f"La base vectorial no respondió en {TIMEOUT_BUSQUEDA_S}s. Reintentá una vez."}
        return {
            "resultados": [
                {
                    "fuente": doc.metadata.get("source", "desconocida"),
                    "seccion": ruta_de_encabezados(doc.metadata),
                    # Chroma devuelve distancia coseno; se pasa a similitud (1 = idéntico).
                    "similitud": round(1 - distancia, 3),
                    "texto": doc.page_content[:LARGO_MAXIMO_FRAGMENTO],
                }
                for doc, distancia in pares
            ]
        }

    @tool(args_schema=ListarDocumentosInput)
    async def listar_documentos() -> dict[str, Any]:
        """Lista los documentos indexados y sus secciones principales.

        Usala si no sabés en qué documento puede estar un dato, para después filtrar
        buscar_documentacion por `documento`. NO devuelve contenido: para eso usá
        buscar_documentacion.

        Returns:
            {"documentos": {"archivo.md": ["sección", ...], ...}}
        """
        return {"documentos": await asyncio.to_thread(documentos_indexados, vectorstore)}

    return preparar_tools([buscar_documentacion, listar_documentos])


def fuentes_consultadas(mensajes: Sequence[BaseMessage]) -> list[str]:
    """Fuentes que devolvió efectivamente `buscar_documentacion` (metadata de Chroma)."""
    fuentes: list[str] = []
    for obs in observaciones(mensajes, "buscar_documentacion"):
        for r in obs.get("resultados", []):
            if r.get("fuente") and r["fuente"] not in fuentes:
                fuentes.append(r["fuente"])
    return fuentes


# =====================================================================================
# Tools del analista
# =====================================================================================

_OPERADORES: dict[type[ast.AST], Callable[..., int | float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_FUNCIONES: dict[str, Callable[..., int | float]] = {
    "round": round,
    "min": min,
    "max": max,
    "abs": abs,
    "sqrt": math.sqrt,
}
LARGO_MAXIMO_EXPRESION = 200


def evaluar_expresion(expresion: str) -> float:
    """Evalúa aritmética básica recorriendo el AST: solo números, + - * / // % **,
    paréntesis y round/min/max/abs/sqrt. Cualquier otra cosa (nombres, atributos,
    llamadas arbitrarias) se rechaza: el LLM nunca ejecuta código."""
    if len(expresion) > LARGO_MAXIMO_EXPRESION:
        raise ValueError(f"expresión demasiado larga (máximo {LARGO_MAXIMO_EXPRESION} caracteres)")

    # Los enteros se mantienen enteros hasta el final: round(x, 1) exige un int como 2.º argumento.
    def nodo(n: ast.AST) -> int | float:
        if isinstance(n, ast.Expression):
            return nodo(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPERADORES:
            izq, der = nodo(n.left), nodo(n.right)
            if isinstance(n.op, ast.Pow) and abs(der) > 10:
                raise ValueError("exponente fuera de rango (máximo 10)")
            return _OPERADORES[type(n.op)](izq, der)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPERADORES:
            return _OPERADORES[type(n.op)](nodo(n.operand))
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in _FUNCIONES
            and not n.keywords
        ):
            return _FUNCIONES[n.func.id](*(nodo(a) for a in n.args))
        raise ValueError(f"elemento no permitido: {ast.dump(n)[:60]}")

    try:
        return float(nodo(ast.parse(expresion, mode="eval")))
    except ZeroDivisionError:
        raise ValueError("división por cero") from None
    except SyntaxError:
        raise ValueError("sintaxis inválida (usá punto decimal: 1.5, no 1,5)") from None
    except (TypeError, OverflowError) as e:  # ej. round(1.5, 1.5) o min() sin argumentos
        raise ValueError(f"operación inválida: {e}") from None


class CalcularInput(EntradaTool):
    expresion: str = Field(
        min_length=1,
        max_length=LARGO_MAXIMO_EXPRESION,
        description="Expresión aritmética con punto decimal, ej. '87 / 100 * 100' o 'round(150 * 0.8)'.",
    )


class PorcentajeInput(EntradaTool):
    valor: float = Field(description="Valor medido, ej. 87 conexiones activas.")
    total: float = Field(gt=0, description="Total o límite de referencia, ej. 100 conexiones.")


class CompararInput(EntradaTool):
    valor: float = Field(description="Valor medido.")
    umbral: float = Field(description="Umbral de referencia (de la documentación).")
    condicion: Literal["mayor", "mayor_o_igual", "menor", "menor_o_igual"] = Field(
        description="Condición que dispara el umbral, ej. 'mayor' para 'alerta si supera el 80%'."
    )


_CONDICIONES: dict[str, Callable[[float, float], bool]] = {
    "mayor": operator.gt,
    "mayor_o_igual": operator.ge,
    "menor": operator.lt,
    "menor_o_igual": operator.le,
}


@tool(args_schema=CalcularInput)
async def calcular(expresion: str) -> dict[str, Any]:
    """Evalúa una expresión aritmética y devuelve el resultado exacto.

    Usala para CUALQUIER cuenta (sumas, restas, proporciones, conversiones). Admite
    + - * / // % **, paréntesis y round/min/max/abs/sqrt. Usá punto decimal.

    NO la uses con texto ni variables ("conexiones * 2"): reemplazá antes cada dato por su número.

    Returns:
        {"expresion": str, "resultado": float} o {"error": str}.
    """
    try:
        return {"expresion": expresion, "resultado": round(evaluar_expresion(expresion), 6)}
    except ValueError as e:
        return {"error": f"No se pudo calcular '{expresion}': {e}."}


@tool(args_schema=PorcentajeInput)
async def porcentaje(valor: float, total: float) -> dict[str, Any]:
    """Calcula qué porcentaje de `total` representa `valor` y cuánto margen queda.

    Usala para porcentajes de uso (ej. conexiones activas sobre el límite del pool).

    Returns:
        {"porcentaje": float, "margen_absoluto": float, "margen_porcentual": float}
    """
    pct = valor / total * 100
    return {
        "porcentaje": round(pct, 2),
        "margen_absoluto": round(total - valor, 4),
        "margen_porcentual": round(100 - pct, 2),
    }


@tool(args_schema=CompararInput)
async def comparar_con_umbral(
    valor: float, umbral: float, condicion: Literal["mayor", "mayor_o_igual", "menor", "menor_o_igual"]
) -> dict[str, Any]:
    """Compara un valor medido contra un umbral documentado y dice si lo dispara.

    Usala para decidir si se activa una alerta, un rollback o un límite (ej. ¿1.6% de
    errores supera el 2% de rollback? → condicion='mayor').

    Returns:
        {"dispara": bool, "diferencia": float}  (diferencia = valor - umbral)
    """
    return {"dispara": _CONDICIONES[condicion](valor, umbral), "diferencia": round(valor - umbral, 4)}


ANALYST_TOOLS = preparar_tools([calcular, porcentaje, comparar_con_umbral])
