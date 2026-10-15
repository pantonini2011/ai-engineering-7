"""Especialista 2 — Agente de Análisis/Cómputo, y la fase de síntesis final.

**Analista**: agente ReAct (`create_react_agent`) con herramientas ACOTADAS al
cálculo (`tools/custom_tools.py`): no tiene acceso a la base vectorial. Trabaja solo
con la instrucción del supervisor y los hallazgos verificados del investigador de
este turno. Cada cálculo que reporta se vuelve a evaluar en código antes de entrar
al estado (`verificaciones`). Escribe únicamente en `analisis`.

**Sintetizador**: redacta la respuesta al usuario con los resultados VALIDADOS del
turno. Es el único nodo que escribe `respuesta_final` y que agrega un AIMessage a
`messages`. Si el LLM falla, arma una respuesta determinística con los mismos datos.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents import crear_agente_react, ejecutar_especialista, es_error_fatal, salida_estructurada
from agents.supervisor import historial_conversacion
from state import (
    Analisis,
    Contribucion,
    OrquestadorState,
    RegistroAnalisis,
    RegistroInvestigacion,
    VerificacionCalculo,
    del_turno,
    siguiente_paso,
)
from tools.custom_tools import ANALYST_TOOLS, evaluar_expresion, tools_usadas

log = logging.getLogger(__name__)

# =====================================================================================
# Analista
# =====================================================================================

SYSTEM_PROMPT = """Sos el AGENTE ANALISTA de un equipo de análisis técnico.
Recibís una instrucción y los DATOS que encontró el investigador en la documentación.

Reglas:
- Usá SOLO los datos recibidos y las cifras que traiga la instrucción. No busques ni inventes datos.
- Hacé TODAS las cuentas con las herramientas (calcular, porcentaje, comparar_con_umbral); nunca de cabeza.
- En `calculos`, copiá en `expresion` exactamente la expresión que evaluaste con calcular
  (con punto decimal) y en `resultado` el número que devolvió.
- Si falta un dato que debería estar en la documentación, listalo en `datos_faltantes`
  en lugar de suponerlo. Lo que solo el usuario puede confirmar (estado en vivo del sistema,
  cantidad de instancias, etc.) va en `supuestos`, igual que lo que figure en
  "YA BUSCADO Y NO DOCUMENTADO": el investigador ya lo buscó y no está.
- Si dos datos se contradicen, listalo en `inconsistencias`.
- Las conclusiones deben ser concretas y verificables (ej. "87% supera el umbral de alerta del 80%")."""


def verificar_calculos(analisis: Analisis) -> list[VerificacionCalculo]:
    """Re-evalúa cada cálculo declarado por el LLM y compara (tolerancia 0,5% o 0,01)."""
    verificaciones = []
    for c in analisis.calculos:
        try:
            real = evaluar_expresion(c.expresion)
        except ValueError as e:
            verificaciones.append(
                VerificacionCalculo(
                    expresion=c.expresion, declarado=c.resultado, recalculado=None, ok=False, detalle=str(e)
                )
            )
            continue
        ok = math.isclose(real, c.resultado, rel_tol=0.005, abs_tol=0.01)
        verificaciones.append(
            VerificacionCalculo(
                expresion=c.expresion,
                declarado=c.resultado,
                recalculado=round(real, 4),
                ok=ok,
                detalle="" if ok else f"declaró {c.resultado} pero la expresión da {round(real, 4)}",
            )
        )
    return verificaciones


def build_analyst_agent(llm: BaseChatModel) -> Any:
    """Subgrafo ReAct del analista (herramientas de cálculo)."""
    return crear_agente_react(llm, ANALYST_TOOLS, SYSTEM_PROMPT, "analista")


def datos_para_analista(investigaciones: list[RegistroInvestigacion]) -> str:
    """Contexto mínimo para el analista: solo los hallazgos de este turno cuya fuente
    devolvió efectivamente la búsqueda (los de fuente no verificada se descartan), más lo
    que el investigador ya buscó sin encontrar, para que no se pida buscarlo de nuevo."""
    lineas = [
        f"- {h.dato} [fuente: {h.fuente}]"
        for r in investigaciones
        if r.resultado
        for h in r.resultado.hallazgos
        if h.fuente in r.fuentes_consultadas
    ]
    texto = "\n".join(lineas) or "(el investigador no aportó datos en este turno)"
    no_documentado = list(
        dict.fromkeys(d for r in investigaciones if r.resultado for d in r.resultado.no_documentado)
    )
    if no_documentado:
        texto += "\n\nYA BUSCADO Y NO DOCUMENTADO (va en `supuestos`, nunca en `datos_faltantes`):\n"
        texto += "\n".join(f"- {d}" for d in no_documentado)
    return texto


def build_analyst_node(llm: BaseChatModel) -> Any:
    agente = build_analyst_agent(llm)
    estructurador = salida_estructurada(llm, Analisis)

    async def analista(state: OrquestadorState) -> dict[str, Any]:
        turno, paso, instruccion = state["turno"], siguiente_paso(state), state["instruccion"]
        inicio = time.perf_counter()
        registro = RegistroAnalisis(turno=turno, paso=paso, instruccion=instruccion)
        datos = datos_para_analista(del_turno(state.get("investigaciones", []), turno))
        usadas: list[str] = []
        try:
            # Aislamiento de contexto: instrucción + datos de la investigación, nada más.
            entrada = f"{instruccion}\n\nDATOS DE LA INVESTIGACIÓN:\n{datos}"
            registro.resultado, mensajes = await ejecutar_especialista(
                agente, estructurador, SYSTEM_PROMPT, entrada, Analisis
            )
            registro.verificaciones = verificar_calculos(registro.resultado)
            usadas = tools_usadas(mensajes)
        except Exception as e:  # el error queda en el estado y decide el supervisor
            if es_error_fatal(e):
                raise
            log.warning("analista falló: %s", e)
            registro.error = f"{type(e).__name__}: {e}"

        if registro.resultado:
            fallidos = sum(not v.ok for v in registro.verificaciones)
            accion = f"analizó ({len(registro.resultado.calculos)} cálculos, {fallidos} con error)"
            detalle = registro.resultado.resumen
        else:
            accion, detalle = "falló", registro.error or ""
        contribucion = Contribucion(
            turno=turno,
            paso=paso,
            agente="analista",
            accion=accion,
            detalle=detalle + (f" · tools: {', '.join(usadas)}" if usadas else ""),
            duracion_s=round(time.perf_counter() - inicio, 2),
        )
        return {"analisis": [registro], "contribuciones": [contribucion]}

    return analista


# =====================================================================================
# Sintetizador (fase de síntesis final)
# =====================================================================================

SYSTEM_PROMPT_SINTESIS = """Sos el redactor final de un equipo de análisis técnico de la "Plataforma de Pedidos".
Con los RESULTADOS VALIDADOS del equipo, respondé la pregunta del usuario.

Reglas:
- Usá SOLO los datos de los resultados. No agregues conocimiento general.
- Citá la fuente de cada dato documental entre corchetes, ej. [runbook_incidentes.md].
- Mostrá las cuentas clave con su resultado (ej. 87 / 100 × 100 = 87%).
- Si el investigador no encontró información, decí claramente que no lo sabés porque la
  documentación no lo cubre, y no inventes nada.
- Si hay una "Falla técnica", NO digas que la documentación no lo cubre: explicá que no se pudo
  consultar (el dato puede existir), respondé lo que sí se sepa y sugerí reintentar.
- Si hay limitaciones o inconsistencias, mencionalas al final en una línea "Limitaciones: ...".
- Español rioplatense, claro y breve (máximo ~200 palabras)."""


def resultados_validados(state: OrquestadorState) -> str:
    """Resumen de lo que el equipo produjo en el turno, con verificaciones aplicadas."""
    turno = state["turno"]
    lineas: list[str] = []
    for r in del_turno(state.get("investigaciones", []), turno):
        if r.error:
            # Sin esto el sintetizador no se entera de la falla y la confunde con "no documentado".
            lineas.append(f"Falla técnica: el investigador no pudo consultar la documentación ({r.error})")
        if not r.resultado:
            continue
        if r.resultado.sin_informacion:
            lineas.append(f"Investigación: SIN INFORMACIÓN en la documentación ({r.resultado.resumen})")
        lineas += [
            f"Dato: {h.dato} [{h.fuente}]" for h in r.resultado.hallazgos if h.fuente in r.fuentes_consultadas
        ]
    lineas += [
        f"Falla técnica: el analista no pudo completar el cálculo ({r.error})"
        for r in del_turno(state.get("analisis", []), turno)
        if r.error
    ]
    analisis = [r for r in del_turno(state.get("analisis", []), turno) if r.resultado]
    if analisis:
        ultimo = analisis[-1]
        assert ultimo.resultado is not None
        for c, v in zip(ultimo.resultado.calculos, ultimo.verificaciones, strict=False):
            valor = v.recalculado if v.recalculado is not None else c.resultado
            estado = "" if v.ok else " (NO VERIFICADO)"
            lineas.append(f"Cálculo: {c.descripcion}: {c.expresion} = {valor}{estado}")
        lineas += [f"Conclusión: {c}" for c in ultimo.resultado.conclusiones]
        lineas += [f"Inconsistencia: {i}" for i in ultimo.resultado.inconsistencias]
        lineas += [f"Supuesto a confirmar: {s}" for s in ultimo.resultado.supuestos]
    lineas += [f"Limitación: {lim}" for lim in state.get("limitaciones", [])]
    return "\n".join(dict.fromkeys(lineas)) or "(el equipo no obtuvo resultados)"


def build_synthesizer_node(llm: BaseChatModel) -> Any:
    async def sintetizador(state: OrquestadorState) -> dict[str, Any]:
        turno, paso = state["turno"], siguiente_paso(state)
        inicio = time.perf_counter()
        resultados = resultados_validados(state)
        try:
            respuesta = await llm.ainvoke(
                [
                    SystemMessage(SYSTEM_PROMPT_SINTESIS),
                    HumanMessage(
                        f"## Conversación previa\n{historial_conversacion(state['messages'])}\n\n"
                        f"## Pregunta\n{state['pregunta']}\n\n## Resultados validados\n{resultados}"
                    ),
                ]
            )
            texto = respuesta.text.strip()
            accion = "redactó la respuesta final"
        except Exception as e:
            if es_error_fatal(e):
                raise
            log.warning("sintetizador falló, se usa la respuesta determinística: %s", e)
            texto = f"No pude redactar la respuesta final; estos son los resultados del equipo:\n{resultados}"
            accion = "respuesta de respaldo (falló el LLM)"
        return {
            "messages": [AIMessage(texto, name="sintetizador")],
            "respuesta_final": texto,
            "task_completed": True,
            "contribuciones": [
                Contribucion(
                    turno=turno,
                    paso=paso,
                    agente="sintetizador",
                    accion=accion,
                    detalle=texto[:160],
                    duracion_s=round(time.perf_counter() - inicio, 2),
                )
            ],
        }

    return sintetizador
