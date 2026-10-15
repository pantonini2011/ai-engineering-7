"""Tests offline del orquestador, agrupados por requisito de la consigna (una clase por requisito).

Los LLM son falsos (respuestas guionadas); el grafo, los nodos, las tools, Chroma y el
checkpointer SQLite son los reales.
"""

from __future__ import annotations

import asyncio
import logging
import typing
from pathlib import Path
from typing import Any

import pytest
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from conftest import FakeLLM, modelos, tool_call
from agents.supervisor import MAX_DELEGACIONES, MAX_POR_ESPECIALISTA, rutear_supervisor
from graph import Modelos, abrir_checkpointer, build_graph
from main import DEMO, config_de, demo
from observability import VARIABLE_CAIDA_VECTORSTORE
from state import (
    Analisis,
    Calculo,
    DecisionSupervisor,
    DestinoSupervisor,
    Hallazgo,
    Investigacion,
    RegistroAnalisis,
    RegistroInvestigacion,
)

RUNBOOK = "runbook_incidentes.md"
PREGUNTA = "Tenemos 87 conexiones activas en PgBouncer, ¿a qué % del límite estamos?"
INSTR_INV = "Buscá el límite de conexiones de PgBouncer y qué indica el runbook."
INSTR_AN = "Calculá qué porcentaje representan 87 conexiones sobre el límite."


def decidir(siguiente: str, instruccion: str = "") -> DecisionSupervisor:
    return DecisionSupervisor(razonamiento="test", siguiente=siguiente, instruccion=instruccion)  # type: ignore[arg-type]


def investigacion_ok(fuente: str = RUNBOOK) -> tuple[list[AIMessage], list[Investigacion]]:
    """Un ciclo ReAct del investigador: busca en el runbook y reporta un hallazgo."""
    return (
        [tool_call("buscar_documentacion", {"consulta": "límite de conexiones", "documento": RUNBOOK}), AIMessage("ok")],
        [Investigacion(resumen="El límite es 100.", hallazgos=[Hallazgo(dato="Límite de 100 conexiones", fuente=fuente)])],
    )


def analisis(resultado: float = 87.0) -> tuple[list[AIMessage], list[Analisis]]:
    """Un ciclo ReAct del analista: usa calcular y reporta el cálculo (con `resultado` declarado)."""
    return (
        [tool_call("calcular", {"expresion": "87 / 100 * 100"}), AIMessage("ok")],
        [
            Analisis(
                resumen="Uso del 87%.",
                calculos=[Calculo(descripcion="uso del pool", expresion="87 / 100 * 100", resultado=resultado)],
                conclusiones=["87% supera el umbral de alerta del 80%"],
            )
        ],
    )


def flujo_feliz() -> Modelos:
    return modelos(
        supervisor=[decidir("investigador", INSTR_INV), decidir("analista", INSTR_AN), decidir("FINALIZAR")],
        investigador=investigacion_ok(),
        analista=analisis(),
        sintetizador=[AIMessage("Están al 87% del límite [runbook_incidentes.md].")],
    )


def correr(
    tmp_path: Path, m: Modelos, vs: Chroma, preguntas: list[str], thread_id: str = "t1"
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Corre uno o más turnos (cada uno con una conexión NUEVA al .db) y devuelve
    el estado final y los updates de cada nodo."""

    async def run() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        updates: list[dict[str, Any]] = []
        for pregunta in preguntas:
            async with abrir_checkpointer(str(tmp_path / "cp.db")) as saver:
                graph = build_graph(saver, m, vs)
                async for u in graph.astream(
                    {"messages": [HumanMessage(pregunta)]}, config_de(thread_id), stream_mode="updates"
                ):
                    updates.append(u)
        async with abrir_checkpointer(str(tmp_path / "cp.db")) as saver:
            estado = (await build_graph(saver, m, vs).aget_state(config_de(thread_id))).values
        return estado, updates

    return asyncio.run(run())


def agentes(estado: dict[str, Any], turno: int = 1) -> list[str]:
    return [c.agente for c in estado["contribuciones"] if c.turno == turno]


class TestFlujoDeDelegacion:
    """Supervisor → Investigador → Supervisor → Analista → Supervisor → Validación → Síntesis → END."""

    def test_delega_investigador_luego_analista_y_finaliza(self, tmp_path: Path, vectorstore: Chroma) -> None:
        estado, _ = correr(tmp_path, flujo_feliz(), vectorstore, [PREGUNTA])
        assert agentes(estado) == [
            "inicio", "supervisor", "investigador", "supervisor", "analista", "supervisor", "validacion", "sintetizador",
        ]
        assert estado["task_completed"] is True
        assert estado["respuesta_final"] == "Están al 87% del límite [runbook_incidentes.md]."
        # `messages` solo guarda la conversación con el usuario: pregunta y respuesta final.
        assert [type(m) for m in estado["messages"]] == [HumanMessage, AIMessage]
        assert estado["messages"][-1].name == "sintetizador"

    def test_aristas_condicionales_usan_literal_con_nombres_de_nodos(self) -> None:
        destinos = typing.get_args(typing.get_type_hints(rutear_supervisor)["return"])
        assert set(destinos) == set(typing.get_args(DestinoSupervisor)) == {"investigador", "analista", "validacion"}
        nodos = build_graph(modelos=flujo_feliz(), vectorstore=_vs_vacio()).get_graph().nodes
        assert set(destinos) <= set(nodos)

    def test_el_supervisor_puede_ir_directo_al_analista(self, tmp_path: Path, vectorstore: Chroma) -> None:
        """No hay orden fijo: el supervisor decide (acá pide el análisis antes de investigar)."""
        m = modelos(
            supervisor=[decidir("analista", INSTR_AN), decidir("investigador", INSTR_INV),
                        decidir("analista", INSTR_AN), decidir("FINALIZAR")],
            investigador=investigacion_ok(),
            analista=([*analisis()[0], *analisis()[0]], [*analisis()[1], *analisis()[1]]),
        )
        estado, _ = correr(tmp_path, m, vectorstore, [PREGUNTA])
        assert agentes(estado)[1:6] == ["supervisor", "analista", "supervisor", "investigador", "supervisor"]


class TestEstadoCompartido:
    """El estado rastrea qué agente aportó qué, y cada agente escribe solo su parcela."""

    PARCELAS = {
        "inicio": {"turno", "pregunta", "instruccion", "delegaciones", "feedback_validacion", "rechazos_validacion",
                   "limitaciones", "task_completed", "respuesta_final", "contribuciones"},
        "supervisor": {"next_agent", "instruccion", "delegaciones", "limitaciones", "contribuciones"},
        "investigador": {"investigaciones", "contribuciones"},
        "analista": {"analisis", "contribuciones"},
        "validacion": {"task_completed", "feedback_validacion", "rechazos_validacion", "limitaciones", "contribuciones"},
        "sintetizador": {"messages", "respuesta_final", "task_completed", "contribuciones"},
    }

    def test_cada_agente_escribe_solo_su_parcela(self, tmp_path: Path, vectorstore: Chroma) -> None:
        _, updates = correr(tmp_path, flujo_feliz(), vectorstore, [PREGUNTA])
        for update in updates:
            for nodo, escrito in update.items():
                assert set(escrito) <= self.PARCELAS[nodo], f"{nodo} escribió fuera de su parcela"

    def test_bitacora_registra_agente_paso_y_fuentes(self, tmp_path: Path, vectorstore: Chroma) -> None:
        estado, _ = correr(tmp_path, flujo_feliz(), vectorstore, [PREGUNTA])
        contribuciones = estado["contribuciones"]
        assert [c.paso for c in contribuciones] == list(range(1, len(contribuciones) + 1))
        inv = next(c for c in contribuciones if c.agente == "investigador")
        assert inv.fuentes == [RUNBOOK]  # sale de la metadata de Chroma, no del LLM
        assert "buscar_documentacion" in inv.detalle
        registro = estado["investigaciones"][0]
        assert isinstance(registro, RegistroInvestigacion)
        assert registro.instruccion == INSTR_INV and registro.fuentes_no_verificadas == []

    def test_fuente_inventada_queda_marcada_y_no_llega_al_analista(self, tmp_path: Path, vectorstore: Chroma) -> None:
        m = modelos(
            supervisor=[decidir("investigador", INSTR_INV), decidir("analista", INSTR_AN)] + [decidir("FINALIZAR")] * 4,
            investigador=investigacion_ok(fuente="manual_inventado.md"),
            analista=analisis(),
        )
        estado, _ = correr(tmp_path, m, vectorstore, [PREGUNTA])
        assert estado["investigaciones"][0].fuentes_no_verificadas == ["manual_inventado.md"]
        entrada_analista = _textos(m.analista)[0]
        assert "manual_inventado.md" not in entrada_analista
        assert any("no devolvió" in c.detalle for c in estado["contribuciones"] if c.agente == "validacion")


class TestAislamientoDeContexto:
    """Cada especialista recibe solo su instrucción y el contexto necesario, no todo el historial."""

    def test_investigador_recibe_solo_la_instruccion(self, tmp_path: Path, vectorstore: Chroma) -> None:
        m = flujo_feliz()
        correr(tmp_path, m, vectorstore, [PREGUNTA])
        primera = _llm(m.investigador).recibidos[0]
        assert [type(x) for x in primera] == [SystemMessage, HumanMessage]
        assert primera[1].text == INSTR_INV
        assert PREGUNTA not in "".join(_textos(m.investigador))

    def test_analista_recibe_instruccion_y_hallazgos_sin_salida_cruda_de_tools(
        self, tmp_path: Path, vectorstore: Chroma
    ) -> None:
        m = flujo_feliz()
        correr(tmp_path, m, vectorstore, [PREGUNTA])
        entrada = _llm(m.analista).recibidos[0][1].text
        assert entrada.startswith(INSTR_AN)
        assert "Límite de 100 conexiones [fuente: runbook_incidentes.md]" in entrada
        assert '"similitud"' not in entrada  # el JSON de la búsqueda no viaja al analista
        assert PREGUNTA not in entrada

    def test_supervisor_ve_un_resumen_no_los_mensajes_internos(self, tmp_path: Path, vectorstore: Chroma) -> None:
        m = flujo_feliz()
        correr(tmp_path, m, vectorstore, [PREGUNTA])
        for llamada in _llm(m.supervisor).recibidos:
            assert not any(isinstance(x, ToolMessage) for x in llamada)
            assert len(llamada) == 2  # system + vista resumida del estado
        ultima = _llm(m.supervisor).recibidos[-1][1].text
        assert "## Trabajo del turno" in ultima and "Límite de 100 conexiones" in ultima

    def test_analista_recibe_lo_ya_buscado_y_no_documentado(self, tmp_path: Path, vectorstore: Chroma) -> None:
        msgs, (inv,) = investigacion_ok()
        inv = inv.model_copy(update={"no_documentado": ["severidad de la alerta"]})
        m = modelos(
            supervisor=[decidir("investigador", INSTR_INV), decidir("analista", INSTR_AN), decidir("FINALIZAR")],
            investigador=(msgs, [inv]),
            analista=analisis(),
        )
        correr(tmp_path, m, vectorstore, [PREGUNTA])
        entrada = _llm(m.analista).recibidos[0][1].text
        assert "YA BUSCADO Y NO DOCUMENTADO" in entrada and "- severidad de la alerta" in entrada
        assert "no documentado: severidad de la alerta" in _llm(m.supervisor).recibidos[1][1].text


class TestValidacionYRefinamiento:
    """La validación revisa la rúbrica antes del END y puede pedir que un especialista refine."""

    def test_calculo_erroneo_se_rechaza_y_el_analista_lo_refina(self, tmp_path: Path, vectorstore: Chroma) -> None:
        malo, bueno = analisis(resultado=78.0), analisis(resultado=87.0)
        m = modelos(
            supervisor=[decidir("investigador", INSTR_INV), decidir("analista", INSTR_AN), decidir("FINALIZAR"),
                        decidir("analista", "Corregí el cálculo del uso del pool."), decidir("FINALIZAR")],
            investigador=investigacion_ok(),
            analista=([*malo[0], *bueno[0]], [*malo[1], *bueno[1]]),
        )
        estado, _ = correr(tmp_path, m, vectorstore, [PREGUNTA])
        validaciones = [c for c in estado["contribuciones"] if c.agente == "validacion"]
        assert validaciones[0].accion.startswith("rechazó") and "78.0" in validaciones[0].detalle
        assert validaciones[-1].accion == "aprobó"
        primer, segundo = estado["analisis"]
        assert isinstance(primer, RegistroAnalisis)
        assert primer.verificaciones[0].ok is False and primer.verificaciones[0].recalculado == 87.0
        assert segundo.verificaciones[0].ok is True
        assert estado["rechazos_validacion"] == 1
        # El supervisor recibió las observaciones de la validación antes de re-delegar.
        assert "Cálculo con error" in _llm(m.supervisor).recibidos[3][1].text

    def test_sin_informacion_finaliza_sin_analisis(self, tmp_path: Path, vectorstore: Chroma) -> None:
        m = modelos(
            supervisor=[decidir("investigador", "Buscá la cotización del dólar."), decidir("FINALIZAR")],
            investigador=([AIMessage("nada")], [Investigacion(resumen="No hay datos.", sin_informacion=True)]),
            sintetizador=[AIMessage("No lo sé: la documentación no cubre la cotización del dólar.")],
        )
        estado, _ = correr(tmp_path, m, vectorstore, ["¿Cuál es la cotización del dólar?"])
        assert "analista" not in agentes(estado)
        assert next(c for c in estado["contribuciones"] if c.agente == "validacion").accion == "aprobó"
        assert estado["respuesta_final"].startswith("No lo sé")

    def test_finalizar_sin_analisis_se_rechaza(self, tmp_path: Path, vectorstore: Chroma) -> None:
        m = modelos(
            supervisor=[decidir("investigador", INSTR_INV), decidir("FINALIZAR"),
                        decidir("analista", INSTR_AN), decidir("FINALIZAR")],
            investigador=investigacion_ok(),
            analista=analisis(),
        )
        estado, _ = correr(tmp_path, m, vectorstore, [PREGUNTA])
        rechazo = next(c for c in estado["contribuciones"] if c.agente == "validacion")
        assert "Falta el análisis" in rechazo.detalle
        assert agentes(estado)[-2:] == ["validacion", "sintetizador"]

    def test_observacion_atendida_no_se_vuelve_a_mostrar(self, tmp_path: Path, vectorstore: Chroma) -> None:
        # Bug visto con la API real: tras atender un rechazo, el supervisor seguía viendo la
        # observación vieja y volvía a delegar por un problema ya resuelto.
        m = modelos(
            supervisor=[decidir("FINALIZAR"), decidir("investigador", INSTR_INV),
                        decidir("analista", INSTR_AN), decidir("FINALIZAR")],
            investigador=investigacion_ok(),
            analista=analisis(),
        )
        estado, _ = correr(tmp_path, m, vectorstore, [PREGUNTA])
        vistas = [llamada[1].text for llamada in _llm(m.supervisor).recibidos]
        assert "No hay una investigación exitosa" in vistas[1]  # justo después del rechazo
        assert "No hay una investigación exitosa" not in vistas[2]  # ya investigó: no se repite
        assert agentes(estado)[-2:] == ["validacion", "sintetizador"]


class TestCondicionDeParada:
    """Anti "supervisor infinito": topes en código que el LLM no puede saltear."""

    def test_tope_por_especialista(self, tmp_path: Path, vectorstore: Chroma) -> None:
        n = MAX_POR_ESPECIALISTA
        msgs, estructurados = investigacion_ok(fuente="otro.md")
        m = modelos(
            supervisor=[decidir("investigador", INSTR_INV)] * 20,
            investigador=(msgs * n, estructurados * n),
        )
        estado, _ = correr(tmp_path, m, vectorstore, [PREGUNTA])
        assert estado["delegaciones"]["investigador"] == n
        assert agentes(estado)[-1] == "sintetizador"  # terminó, sin GraphRecursionError
        assert any("ya intervino" in lim for lim in estado["limitaciones"])
        # Tras un rechazo, el supervisor no pudo resolver nada nuevo y la validación encontró
        # los mismos problemas: no insiste otra vez, responde declarando las limitaciones.
        assert estado["rechazos_validacion"] == 1
        validaciones = [c.accion for c in estado["contribuciones"] if c.agente == "validacion"]
        assert validaciones == ["rechazó (1/2)", "aprobó con limitaciones"]

    def test_tope_total_de_delegaciones(self, tmp_path: Path, vectorstore: Chroma) -> None:
        mitad = MAX_DELEGACIONES // 2
        inv, an = investigacion_ok(), analisis()
        m = modelos(
            supervisor=[decidir("investigador", INSTR_INV), decidir("analista", INSTR_AN)] * 10,
            investigador=(inv[0] * mitad, inv[1] * mitad),
            analista=(an[0] * mitad, an[1] * mitad),
        )
        estado, _ = correr(tmp_path, m, vectorstore, [PREGUNTA])
        assert sum(estado["delegaciones"].values()) == MAX_DELEGACIONES
        assert any("tope" in lim for lim in estado["limitaciones"])
        assert agentes(estado)[-1] == "sintetizador"

    def test_fail_safe_si_el_supervisor_falla(self, tmp_path: Path, vectorstore: Chroma) -> None:
        m = modelos(supervisor=[ValueError("JSON inválido")] * 20)
        estado, _ = correr(tmp_path, m, vectorstore, [PREGUNTA])
        assert agentes(estado)[-1] == "sintetizador"
        assert estado["respuesta_final"]
        assert any("fail-safe" in c.detalle for c in estado["contribuciones"] if c.agente == "supervisor")

    def test_sin_saldo_corta_en_lugar_de_reintentar(self, tmp_path: Path, vectorstore: Chroma) -> None:
        sin_saldo = RuntimeError("Error code: 400 - Your credit balance is too low to access the Anthropic API.")
        m = modelos(supervisor=[sin_saldo] * 20)
        with pytest.raises(RuntimeError, match="credit balance"):
            correr(tmp_path, m, vectorstore, [PREGUNTA])
        # El intento + el reintento de .with_retry, y corta: sin fail-safe ni más delegaciones.
        assert len(_llm(m.supervisor).recibidos) == 2


class TestPersistencia:
    """Checkpointer SQLite + thread_id (pre-entrega 5) con el estado multi-agente."""

    def test_segundo_turno_recuerda_la_conversacion(
        self, tmp_path: Path, vectorstore: Chroma, caplog: pytest.LogCaptureFixture
    ) -> None:
        inv, an = investigacion_ok(), analisis()
        m = modelos(
            supervisor=[decidir("investigador", INSTR_INV), decidir("analista", INSTR_AN), decidir("FINALIZAR")] * 2,
            investigador=(inv[0] * 2, inv[1] * 2),
            analista=(an[0] * 2, an[1] * 2),
            sintetizador=[AIMessage("Primera respuesta."), AIMessage("Segunda respuesta.")],
        )
        with caplog.at_level(logging.WARNING):
            estado, _ = correr(tmp_path, m, vectorstore, [PREGUNTA, "¿Y si subimos el límite a 150?"])
        assert estado["turno"] == 2
        assert [r.turno for r in estado["investigaciones"]] == [1, 2]  # las parcelas acumulan por turno
        assert isinstance(estado["analisis"][0], RegistroAnalisis)  # los tipos sobreviven al checkpoint
        assert "unregistered type" not in caplog.text  # allowlist del serializador
        vista_turno_2 = _llm(m.supervisor).recibidos[3][1].text
        assert "Usuario: " + PREGUNTA in vista_turno_2 and "Equipo: Primera respuesta." in vista_turno_2

    def test_threads_distintos_no_comparten_estado(self, tmp_path: Path, vectorstore: Chroma) -> None:
        correr(tmp_path, flujo_feliz(), vectorstore, [PREGUNTA], thread_id="a")
        estado, _ = correr(tmp_path, flujo_feliz(), vectorstore, [PREGUNTA], thread_id="b")
        assert estado["turno"] == 1 and len(estado["investigaciones"]) == 1


# --- helpers -----------------------------------------------------------------------------


def _llm(modelo: Any) -> FakeLLM:
    assert isinstance(modelo, FakeLLM)
    return modelo


def _textos(modelo: Any) -> list[str]:
    return [m.text for llamada in _llm(modelo).recibidos for m in llamada if isinstance(m, HumanMessage)]


def _vs_vacio() -> Any:
    """Para tests que solo inspeccionan la estructura del grafo (no buscan)."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    return Chroma(collection_name="vacia", embedding_function=DeterministicFakeEmbedding(size=8))


class TestDemo:
    def test_demo_corre_los_turnos_sobre_una_base_nueva(self, tmp_path: Path, vectorstore: Chroma) -> None:
        """La demo de main.py (la que genera la traza) corre de punta a punta sin API."""
        n = len(DEMO)
        inv, an = investigacion_ok(), analisis()
        m = modelos(
            supervisor=[decidir("investigador", INSTR_INV), decidir("analista", INSTR_AN), decidir("FINALIZAR")] * n,
            investigador=(inv[0] * n, inv[1] * n),
            analista=(an[0] * n, an[1] * n),
            sintetizador=[AIMessage(f"Respuesta {i}") for i in range(n)],
        )
        turnos = asyncio.run(demo(m, str(tmp_path / "nueva.db"), vectorstore))
        assert [t["respuesta_final"] for t in turnos] == [f"Respuesta {i}" for i in range(n)]
        assert turnos[1]["turno"] == 2  # el 2.º turno de la demo es un seguimiento en el mismo thread
        assert [p["agente"] for p in turnos[0]["pasos"]][-2:] == ["validacion", "sintetizador"]


class TestCorreccionesModulo7:
    """Defectos vistos en las trazas de Phoenix del Módulo 7."""

    def test_tope_queda_como_bloqueo_en_la_bitacora(self, tmp_path: Path, vectorstore: Chroma) -> None:
        # El tope ya se imponía en código, pero la bitácora decía "envió a validación" y se
        # leía como una delegación más.
        n = MAX_POR_ESPECIALISTA
        msgs, estructurados = investigacion_ok(fuente="otro.md")
        m = modelos(supervisor=[decidir("investigador", INSTR_INV)] * 20, investigador=(msgs * n, estructurados * n))
        estado, _ = correr(tmp_path, m, vectorstore, [PREGUNTA])
        assert estado["delegaciones"]["investigador"] == n
        bloqueos = [c for c in estado["contribuciones"] if c.agente == "supervisor" and "bloqueó" in c.accion]
        assert bloqueos and "el código no lo permite" in bloqueos[0].detalle

    def test_datos_faltantes_no_rechazan_y_quedan_como_limitacion(self, tmp_path: Path, vectorstore: Chroma) -> None:
        # Visto en sesion-redis: rechazar por un dato que solo tiene el usuario gastaba una vuelta.
        msgs, [an] = analisis()
        an_con_faltante = an.model_copy(update={"datos_faltantes": ["cantidad de usuarios afectados"]})
        m = modelos(
            supervisor=[decidir("investigador", INSTR_INV), decidir("analista", INSTR_AN), decidir("FINALIZAR")],
            investigador=investigacion_ok(),
            analista=(msgs, [an_con_faltante]),
        )
        estado, _ = correr(tmp_path, m, vectorstore, [PREGUNTA])
        validaciones = [c.accion for c in estado["contribuciones"] if c.agente == "validacion"]
        assert validaciones == ["aprobó"]
        assert "Falta un dato para el análisis: cantidad de usuarios afectados" in estado["limitaciones"]

    def test_falla_tecnica_llega_al_sintetizador_sin_instrucciones_internas(
        self, tmp_path: Path, vectorstore: Chroma, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Visto en sesion-falla: el ConnectionError no llegaba al sintetizador y la limitación
        # que recibía era "delegá en el investigador", una instrucción para el supervisor.
        monkeypatch.setenv(VARIABLE_CAIDA_VECTORSTORE, "1")
        m = modelos(
            supervisor=[decidir("investigador", INSTR_INV), decidir("FINALIZAR"), decidir("FINALIZAR")],
            investigador=investigacion_ok(),
        )
        estado, _ = correr(tmp_path, m, vectorstore, [PREGUNTA])
        assert not any("delegá" in lim for lim in estado["limitaciones"])
        assert any("No se pudo consultar la documentación" in lim for lim in estado["limitaciones"])
        entrada_sintetizador = _llm(m.sintetizador).recibidos[-1][-1].text
        assert "Falla técnica" in entrada_sintetizador and "ConnectionError" in entrada_sintetizador
