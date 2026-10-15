"""Nodo Supervisor (decisión dinámica con Literal) y nodo de Validación.

**Supervisor**: router inteligente y controlador de flujo. Coordina, no ejecuta: no
busca ni calcula. En cada paso lee un RESUMEN del trabajo del turno (no los mensajes
internos de los especialistas) y decide con salida estructurada (`DecisionSupervisor`)
quién interviene ahora o si es momento de finalizar, aplicando la rúbrica de
suficiencia de su prompt. La decisión se mapea a los nodos del grafo con un `Literal`
(`DestinoSupervisor`).

**Validación**: un LLM puede "creer" que terminó sin que sea así, así que antes del END
la rúbrica se vuelve a chequear EN CÓDIGO. Si algo falla, el turno vuelve al supervisor.

**Anti "supervisor infinito"**: topes en código que el LLM no puede saltear
(delegaciones por turno y por especialista, rechazos de la validación) y un fail-safe
si el LLM no devuelve una decisión válida.
"""

from __future__ import annotations

import logging
import time
from typing import Any, NamedTuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from agents import es_error_fatal, salida_estructurada
from state import (
    Contribucion,
    DecisionSupervisor,
    DestinoSupervisor,
    DestinoValidacion,
    OrquestadorState,
    RegistroAnalisis,
    RegistroInvestigacion,
    del_turno,
    siguiente_paso,
)

log = logging.getLogger(__name__)

# --- Límites (anti "supervisor infinito") -------------------------------------------

# Delegaciones totales por turno (investigador + analista). Al llegar al tope el
# supervisor deja de delegar y el flujo pasa a validación/síntesis con lo que haya.
MAX_DELEGACIONES = 6
# Veces que se puede delegar al MISMO especialista en un turno (1 + 2 refinamientos).
MAX_POR_ESPECIALISTA = 3
# Veces que la validación puede rechazar y devolver el turno al supervisor.
MAX_RECHAZOS_VALIDACION = 2
# Mensajes de la conversación (pregunta/respuesta final) que ve el supervisor.
MAX_HISTORIAL_SUPERVISOR = 6

# =====================================================================================
# Supervisor
# =====================================================================================

SYSTEM_PROMPT = f"""Sos el SUPERVISOR de un equipo de análisis técnico sobre la "Plataforma de Pedidos".
No investigás ni calculás: coordinás a dos especialistas.

- investigador: busca en la documentación interna (arquitectura, monitoreo y alertas, normativa
  de despliegue, runbook de incidentes) y devuelve datos con su fuente. No calcula.
- analista: hace cálculos, porcentajes y comparaciones contra umbrales usando los datos que
  encontró el investigador (se los pasamos automáticamente). No busca.

Dada la conversación actual y el trabajo hecho en este turno, ¿quién debe intervenir ahora,
o es momento de finalizar?

Flujo habitual: investigador → analista → FINALIZAR. Volvé a delegar en un especialista solo
para REFINAR algo concreto que la rúbrica marque como incompleto.

RÚBRICA DE SUFICIENCIA (elegí FINALIZAR solo si se cumple todo):
1. Cada parte de la pregunta que depende de la documentación tiene al menos un hallazgo con fuente.
2. Cada cifra, porcentaje o comparación que pide la pregunta fue calculada por el analista y
   ningún cálculo figura con error de verificación.
3. El análisis no tiene datos faltantes que el investigador pueda conseguir en la documentación.
   Los supuestos (estado en vivo, datos que solo el usuario conoce) no bloquean: se informan.
4. Si el investigador marcó "sin información" para el tema central, elegí FINALIZAR: la respuesta
   dirá que no se sabe. No insistas ni pidas al analista que suponga datos.
5. Si hay inconsistencias entre datos, pedí al investigador que las aclare UNA vez; si persisten,
   FINALIZAR (se informan como limitación).
6. Si la sección "Observaciones de la validación" trae problemas, resolvelos antes de finalizar.

La `instruccion` debe ser AUTOCONTENIDA: el especialista no ve la conversación. Incluí las cifras
que dio el usuario y, en preguntas de seguimiento, el contexto necesario de la conversación previa.
Hay un tope de {MAX_DELEGACIONES} delegaciones por turno y {MAX_POR_ESPECIALISTA} por especialista."""


def historial_conversacion(mensajes: list[AnyMessage]) -> str:
    """Conversación previa (preguntas y respuestas finales), sin la pregunta actual."""
    previos = [m for m in mensajes[:-1] if isinstance(m, (HumanMessage, AIMessage))]
    previos = previos[-MAX_HISTORIAL_SUPERVISOR:]
    return "\n".join(
        f"{'Usuario' if isinstance(m, HumanMessage) else 'Equipo'}: {m.text}" for m in previos
    ) or "(sin conversación previa)"


def _resumen_investigacion(r: RegistroInvestigacion) -> str:
    if r.error:
        return f"[paso {r.paso}] investigador FALLÓ: {r.error}"
    assert r.resultado is not None
    if r.resultado.sin_informacion:
        return f"[paso {r.paso}] investigador: SIN INFORMACIÓN en la documentación. {r.resultado.resumen}"
    lineas = [f"[paso {r.paso}] investigador: {r.resultado.resumen}"]
    lineas += [f"   - {h.dato} [{h.fuente}]" for h in r.resultado.hallazgos]
    lineas += [f"   - no documentado: {d}" for d in r.resultado.no_documentado]
    if r.fuentes_no_verificadas:
        lineas.append(f"   ! cita fuentes que la búsqueda no devolvió: {r.fuentes_no_verificadas}")
    return "\n".join(lineas)


def _resumen_analisis(r: RegistroAnalisis) -> str:
    if r.error:
        return f"[paso {r.paso}] analista FALLÓ: {r.error}"
    assert r.resultado is not None
    lineas = [f"[paso {r.paso}] analista: {r.resultado.resumen}"]
    for c, v in zip(r.resultado.calculos, r.verificaciones, strict=False):
        marca = "ok" if v.ok else f"ERROR DE VERIFICACIÓN: {v.detalle}"
        lineas.append(f"   - {c.descripcion}: {c.expresion} = {c.resultado} ({marca})")
    lineas += [f"   - conclusión: {c}" for c in r.resultado.conclusiones]
    lineas += [f"   - dato faltante: {d}" for d in r.resultado.datos_faltantes]
    lineas += [f"   - supuesto: {s}" for s in r.resultado.supuestos]
    lineas += [f"   - inconsistencia: {i}" for i in r.resultado.inconsistencias]
    return "\n".join(lineas)


def vista_del_supervisor(state: OrquestadorState) -> str:
    """Lo único que ve el LLM supervisor: un resumen compacto, no el historial interno."""
    turno = state["turno"]
    trabajo = sorted(
        [(r.paso, _resumen_investigacion(r)) for r in del_turno(state.get("investigaciones", []), turno)]
        + [(r.paso, _resumen_analisis(r)) for r in del_turno(state.get("analisis", []), turno)]
    )
    delegaciones = state.get("delegaciones", {})
    restantes = MAX_DELEGACIONES - sum(delegaciones.values())
    # La observación de la validación está pendiente solo si nadie actuó desde el rechazo.
    # Si un especialista ya intervino, mostrarla haría re-delegar por un problema ya
    # atendido (si persiste, la próxima validación lo vuelve a marcar).
    contribuciones = del_turno(state.get("contribuciones", []), turno)
    pendiente = bool(contribuciones) and contribuciones[-1].agente == "validacion"
    observaciones = (state.get("feedback_validacion") if pendiente else "") or "(ninguna)"
    return "\n\n".join(
        [
            f"## Conversación previa\n{historial_conversacion(state['messages'])}",
            f"## Pregunta actual\n{state['pregunta']}",
            "## Trabajo del turno\n" + ("\n".join(t for _, t in trabajo) or "(todavía nadie intervino)"),
            f"## Observaciones de la validación\n{observaciones}",
            f"## Presupuesto\nDelegaciones hechas: {delegaciones}. Restantes en el turno: {restantes}.",
        ]
    )


def build_supervisor_node(llm: BaseChatModel) -> Any:
    # Salida estructurada validada con Pydantic + reintento si el JSON llega mal (pre-entrega 2).
    decisor = salida_estructurada(llm, DecisionSupervisor)

    async def supervisor(state: OrquestadorState) -> dict[str, Any]:
        turno, paso = state["turno"], siguiente_paso(state)
        inicio = time.perf_counter()
        delegaciones = dict(state.get("delegaciones", {}))
        limitaciones = list(state.get("limitaciones", []))

        def a_validacion(
            motivo: str, limitacion: str | None = None, accion: str = "envió a validación"
        ) -> dict[str, Any]:
            if limitacion and limitacion not in limitaciones:
                limitaciones.append(limitacion)
            return {
                "next_agent": "validacion",
                "instruccion": "",
                "limitaciones": limitaciones,
                "contribuciones": [
                    Contribucion(
                        turno=turno,
                        paso=paso,
                        agente="supervisor",
                        accion=accion,
                        detalle=motivo,
                        duracion_s=round(time.perf_counter() - inicio, 2),
                    )
                ],
            }

        # Los topes se imponen acá, en código: el LLM no puede saltearlos. Cuando bloquean un
        # pedido, la bitácora lo dice explícitamente para que no se lea como una delegación.
        bloqueo = "bloqueó la delegación (tope) y envió a validación"
        if sum(delegaciones.values()) >= MAX_DELEGACIONES:
            tope = f"Se alcanzó el tope de {MAX_DELEGACIONES} delegaciones del turno."
            return a_validacion(tope, tope, bloqueo)

        try:
            salida = await decisor.ainvoke(
                [SystemMessage(SYSTEM_PROMPT), HumanMessage(vista_del_supervisor(state))]
            )
            decision = DecisionSupervisor.model_validate(salida)
        except Exception as e:
            if es_error_fatal(e):
                raise
            log.warning("supervisor sin decisión válida: %s", e)
            return a_validacion(
                f"fail-safe: el supervisor no devolvió una decisión válida ({type(e).__name__})",
                "El supervisor no pudo completar la coordinación; la respuesta usa lo obtenido hasta ahí.",
            )

        if decision.siguiente == "FINALIZAR":
            return a_validacion(f"decidió finalizar: {decision.razonamiento}")

        destino: DestinoSupervisor = decision.siguiente
        if delegaciones.get(destino, 0) >= MAX_POR_ESPECIALISTA:
            tope = f"El {destino} ya intervino {MAX_POR_ESPECIALISTA} veces en este turno."
            return a_validacion(
                f"el LLM pidió al {destino} otra vez; el código no lo permite: {tope[0].lower()}{tope[1:]}",
                tope,
                bloqueo,
            )

        delegaciones[destino] = delegaciones.get(destino, 0) + 1
        instruccion = decision.instruccion.strip() or state["pregunta"]
        return {
            "next_agent": destino,
            "instruccion": instruccion,
            "delegaciones": delegaciones,
            "contribuciones": [
                Contribucion(
                    turno=turno,
                    paso=paso,
                    agente="supervisor",
                    accion=f"delegó a {destino}",
                    detalle=f"{decision.razonamiento} → instrucción: {instruccion}",
                    duracion_s=round(time.perf_counter() - inicio, 2),
                )
            ],
        }

    return supervisor


def rutear_supervisor(state: OrquestadorState) -> DestinoSupervisor:
    """Arista condicional: el retorno es un Literal con los nombres de los nodos destino."""
    return state["next_agent"]


# =====================================================================================
# Validación (rúbrica en código antes del END)
# =====================================================================================


class Problema(NamedTuple):
    """Un incumplimiento de la rúbrica, con dos textos para dos lectores distintos."""

    instruccion: str  # para el supervisor: qué hacer para resolverlo
    limitacion: str  # para el usuario, si el turno se cierra sin resolverlo


def evaluar_rubrica(state: OrquestadorState) -> list[Problema]:
    """Devuelve la lista de problemas encontrados (vacía = el turno está completo).

    - hay investigación con al menos un hallazgo cuya fuente devolvió la búsqueda;
    - el investigador no citó fuentes que la búsqueda no devolvió;
    - hay un análisis posterior a la última investigación (si no, quedó desactualizado);
    - todos los cálculos del analista coinciden al re-evaluarlos.

    Los datos faltantes del análisis NO se rechazan: la validación corre cuando el
    supervisor ya decidió finalizar, o sea, cuando ya juzgó (rúbrica, punto 3) que el
    investigador no puede conseguirlos. Rechazar de nuevo solo gastaba una vuelta, por
    ejemplo con datos que solo tiene el usuario. Se informan como limitación.
    """
    turno = state["turno"]
    del_turno_inv = del_turno(state.get("investigaciones", []), turno)
    investigaciones = [r for r in del_turno_inv if r.resultado]
    if not investigaciones:
        fallas = [r.error for r in del_turno_inv if r.error]
        limitacion = (
            f"No se pudo consultar la documentación: el investigador falló {len(fallas)} "
            f"{'vez' if len(fallas) == 1 else 'veces'} ({fallas[-1]}). El dato puede existir; conviene reintentar."
            if fallas
            else "No se consultó la documentación en este turno."
        )
        return [Problema("No hay una investigación exitosa en este turno: delegá en el investigador.", limitacion)]
    if all(r.resultado and r.resultado.sin_informacion for r in investigaciones):
        return []  # la documentación no cubre el tema: se responde "no lo sé", sin analizar

    problemas: list[Problema] = []
    ultima_inv = investigaciones[-1]
    if ultima_inv.fuentes_no_verificadas:
        fuentes = ", ".join(ultima_inv.fuentes_no_verificadas)
        problemas.append(
            Problema(
                f"El investigador citó fuentes que la búsqueda no devolvió ({fuentes}): "
                "pedile que vuelva a buscar y cite solo fuentes recuperadas.",
                f"Se descartaron datos atribuidos a fuentes que la búsqueda no devolvió ({fuentes}).",
            )
        )
    verificados = [
        h
        for r in investigaciones
        if r.resultado
        for h in r.resultado.hallazgos
        if h.fuente in r.fuentes_consultadas
    ]
    if not verificados:
        problemas.append(
            Problema(
                "Ningún hallazgo tiene una fuente devuelta por la búsqueda.",
                "Ningún dato tiene una fuente documental verificada.",
            )
        )

    analisis = [r for r in del_turno(state.get("analisis", []), turno) if r.resultado]
    if not analisis:
        problemas.append(
            Problema(
                "Falta el análisis/cómputo de los datos: delegá en el analista.",
                "Los datos no pudieron analizarse ni calcularse en este turno.",
            )
        )
        return problemas
    ultimo = analisis[-1]
    assert ultimo.resultado is not None
    if ultimo.paso < ultima_inv.paso:
        problemas.append(
            Problema(
                "El análisis es anterior a la última investigación: pedile al analista que lo actualice.",
                "El análisis no incorpora la última búsqueda en la documentación.",
            )
        )
    problemas += [
        Problema(
            f"Cálculo con error ({v.expresion}): {v.detalle}. Pedile al analista que lo corrija.",
            f"Un cálculo no pudo verificarse ({v.expresion}).",
        )
        for v in ultimo.verificaciones
        if not v.ok
    ]
    if not ultimo.resultado.calculos and not ultimo.resultado.conclusiones:
        problemas.append(
            Problema("El análisis no tiene cálculos ni conclusiones.", "El análisis no llegó a conclusiones.")
        )
    return problemas


def datos_faltantes(state: OrquestadorState) -> list[str]:
    """Datos que el último análisis del turno marcó como faltantes, como limitaciones."""
    analisis = [r for r in del_turno(state.get("analisis", []), state["turno"]) if r.resultado]
    if not analisis or analisis[-1].resultado is None:
        return []
    return [f"Falta un dato para el análisis: {d}" for d in analisis[-1].resultado.datos_faltantes]


async def validacion(state: OrquestadorState) -> dict[str, Any]:
    """Aprueba, o devuelve el turno al supervisor con las observaciones.

    Rechaza como máximo MAX_RECHAZOS_VALIDACION veces y mientras quede presupuesto de
    delegaciones. Tampoco insiste si encuentra los mismos problemas que en el rechazo
    anterior: el supervisor ya los evaluó, los consideró no resolubles y finalizó igual.
    En esos casos aprueba con limitaciones: el flujo siempre termina. Las limitaciones
    usan el texto para el usuario, nunca la instrucción dirigida al supervisor.
    """
    turno, paso = state["turno"], siguiente_paso(state)
    problemas = evaluar_rubrica(state)
    rechazos = state.get("rechazos_validacion", 0)
    hay_presupuesto = sum(state.get("delegaciones", {}).values()) < MAX_DELEGACIONES

    def contribucion(accion: str, detalle: str) -> Contribucion:
        return Contribucion(turno=turno, paso=paso, agente="validacion", accion=accion, detalle=detalle)

    def limitaciones(nuevas: list[str]) -> list[str]:
        return list(dict.fromkeys([*state.get("limitaciones", []), *nuevas, *datos_faltantes(state)]))

    if not problemas:
        return {
            "task_completed": True,
            "feedback_validacion": "",
            "limitaciones": limitaciones([]),
            "contribuciones": [contribucion("aprobó", "la rúbrica de suficiencia se cumple")],
        }
    feedback = "\n".join(f"- {p.instruccion}" for p in problemas)
    repetido = rechazos > 0 and feedback == state.get("feedback_validacion")
    if rechazos < MAX_RECHAZOS_VALIDACION and hay_presupuesto and not repetido:
        return {
            "task_completed": False,
            "feedback_validacion": feedback,
            "rechazos_validacion": rechazos + 1,
            "contribuciones": [
                contribucion(
                    f"rechazó ({rechazos + 1}/{MAX_RECHAZOS_VALIDACION})", " | ".join(p.instruccion for p in problemas)
                )
            ],
        }
    return {
        "task_completed": True,
        "limitaciones": limitaciones([p.limitacion for p in problemas]),
        "contribuciones": [contribucion("aprobó con limitaciones", " | ".join(p.limitacion for p in problemas))],
    }


def rutear_validacion(state: OrquestadorState) -> DestinoValidacion:
    """Arista condicional de la validación: refinar (supervisor) o cerrar (sintetizador)."""
    return "sintetizador" if state["task_completed"] else "supervisor"
