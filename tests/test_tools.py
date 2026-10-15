"""Tests de las herramientas de cada especialista (acotadas, validadas y seguras)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import BaseTool

from agents.analyst_agent import verificar_calculos
from state import Analisis, Calculo
from tools.custom_tools import (
    ANALYST_TOOLS,
    DOCS_DIR,
    MAX_TOKENS_MODELO,
    build_research_tools,
    evaluar_expresion,
    fragmentar,
)


def _invocar(tool: BaseTool, args: dict[str, Any]) -> Any:
    salida = asyncio.run(tool.ainvoke(args))
    return json.loads(salida) if isinstance(salida, str) else salida


def _tool(tools: list[BaseTool], nombre: str) -> BaseTool:
    return next(t for t in tools if t.name == nombre)


class TestHerramientasDelAnalista:
    def test_calcular_resuelve_aritmetica(self) -> None:
        assert _invocar(_tool(ANALYST_TOOLS, "calcular"), {"expresion": "87 / 100 * 100"})["resultado"] == 87.0
        assert evaluar_expresion("round(150 * 0.8) - 87") == 33.0
        assert evaluar_expresion("round(87 / 150 * 100, 1)") == 58.0  # round con decimales (bug real del notebook)
        assert evaluar_expresion("max(87 - 80, 0) ** 2") == 49.0

    @pytest.mark.parametrize(
        "expresion",
        [
            "__import__('os').system('dir')", "open('x')", "(1).__class__", "x * 2", "9 ** 99", "1 / 0",
            "1,5 + 2", "round(1.55, 1.5)", "min()",
        ],
    )
    def test_calcular_rechaza_lo_que_no_es_aritmetica(self, expresion: str) -> None:
        salida = _invocar(_tool(ANALYST_TOOLS, "calcular"), {"expresion": expresion})
        assert "error" in salida

    def test_porcentaje_y_umbral(self) -> None:
        assert _invocar(_tool(ANALYST_TOOLS, "porcentaje"), {"valor": 87, "total": 150}) == {
            "porcentaje": 58.0, "margen_absoluto": 63.0, "margen_porcentual": 42.0,
        }
        umbral = _tool(ANALYST_TOOLS, "comparar_con_umbral")
        assert _invocar(umbral, {"valor": 1.6, "umbral": 2, "condicion": "mayor"})["dispara"] is False
        assert _invocar(umbral, {"valor": 1.6, "umbral": 1, "condicion": "mayor"})["dispara"] is True

    def test_argumentos_invalidos_vuelven_como_error(self) -> None:
        salida = _invocar(_tool(ANALYST_TOOLS, "porcentaje"), {"valor": 87, "total": 0})
        assert "Argumentos inválidos" in salida["error"]
        salida = _invocar(_tool(ANALYST_TOOLS, "calcular"), {"expresion": "1+1", "extra": True})
        assert "Argumentos inválidos" in salida["error"]

    def test_verificar_calculos_detecta_aritmetica_del_llm_incorrecta(self) -> None:
        analisis = Analisis(
            resumen="-",
            calculos=[
                Calculo(descripcion="ok", expresion="87 / 150 * 100", resultado=58.0),
                Calculo(descripcion="mal", expresion="87 / 150 * 100", resultado=85.0),
                Calculo(descripcion="ilegible", expresion="ochenta / 2", resultado=40.0),
            ],
        )
        ok, mal, ilegible = verificar_calculos(analisis)
        assert ok.ok and not mal.ok and mal.recalculado == 58.0
        assert not ilegible.ok and ilegible.recalculado is None


class TestHerramientasDelInvestigador:
    def test_busqueda_devuelve_fuente_y_seccion_de_la_metadata(self, vectorstore: Chroma) -> None:
        buscar = _tool(build_research_tools(vectorstore), "buscar_documentacion")
        salida = _invocar(buscar, {"consulta": "pool de conexiones", "documento": "runbook_incidentes.md"})
        assert 1 <= len(salida["resultados"]) <= 5  # top_k acotado
        for r in salida["resultados"]:
            assert r["fuente"] == "runbook_incidentes.md"
            assert r["seccion"].startswith("Runbook de incidentes")
            assert set(r) == {"fuente", "seccion", "similitud", "texto"}

    def test_listar_documentos(self, vectorstore: Chroma) -> None:
        listar = _tool(build_research_tools(vectorstore), "listar_documentos")
        documentos = _invocar(listar, {})["documentos"]
        assert sorted(documentos) == [
            "arquitectura_sistema.md", "monitoreo_alertas.md", "normativa_despliegue.md", "runbook_incidentes.md",
        ]
        assert "Incidente 1: Agotamiento del pool de conexiones a PostgreSQL" in documentos["runbook_incidentes.md"]

    def test_consulta_vacia_no_ejecuta_la_busqueda(self, vectorstore: Chroma) -> None:
        buscar = _tool(build_research_tools(vectorstore), "buscar_documentacion")
        assert "Argumentos inválidos" in _invocar(buscar, {"consulta": " "})["error"]

    def test_las_tools_de_cada_especialista_estan_acotadas(self, vectorstore: Chroma) -> None:
        investigador = {t.name for t in build_research_tools(vectorstore)}
        analista = {t.name for t in ANALYST_TOOLS}
        assert investigador == {"buscar_documentacion", "listar_documentos"}
        assert analista == {"calcular", "porcentaje", "comparar_con_umbral"}
        assert not investigador & analista


class TestFragmentacion:
    def test_fragmentos_entran_en_la_ventana_del_modelo_y_llevan_su_seccion(self) -> None:
        """El modelo de embeddings ignora lo que pasa de 128 tokens: ningún fragmento puede superarlo,
        y los subfragmentos de una sección larga llevan su ruta de encabezados."""
        texto = (DOCS_DIR / "normativa_despliegue.md").read_text(encoding="utf-8")

        def contar(t: str) -> int:  # aproximación de tokens, sin descargar el tokenizer
            return len(t.split())

        fragmentos = fragmentar(Document(page_content=texto, metadata={"source": "normativa_despliegue.md"}), contar)
        assert all(contar(f.page_content) <= MAX_TOKENS_MODELO for f in fragmentos)
        proceso = [f for f in fragmentos if f.metadata.get("Header 2") == "Proceso de despliegue a producción"]
        assert len(proceso) > 1
        assert all(f.page_content.startswith("[Normativa de despliegue — Plataforma de Pedidos > Proceso") for f in proceso)
        assert any("revierte automáticamente" in f.page_content for f in proceso)
