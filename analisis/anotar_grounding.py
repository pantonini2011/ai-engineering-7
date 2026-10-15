"""Registra en Phoenix la evaluación de grounding de cada turno (anotación sobre el span raíz).

La evaluación se hizo verificando cada cifra, umbral, comando y criticidad de la respuesta
final contra los documentos de data/docs (ver el informe del ejercicio). Se registra como
annotator_kind="LLM" porque la revisión la hizo Claude, no una persona.
"""

import sys

from phoenix.client import Client

# proyecto → sesión → [(label, score, explicación)]. sesion-pool tiene dos turnos: se anotan en orden.
SESION2: dict[str, list[tuple[str, float, str]]] = {
    "sesion-pool": [
        ("grounded", 1.0, "100 conexiones, 80% durante 2 min, pasos del runbook (pg_stat_activity, pg_cancel_backend, 150) y criticidad alta coinciden con monitoreo_alertas.md y runbook_incidentes.md."),
        ("grounded", 1.0, "Usa el límite temporal 150 del runbook y explicita que la base del 80% tras el cambio no está documentada; calcula ambos escenarios sin inventar el dato."),
    ],
    "sesion-canario": [
        ("grounded", 1.0, "Reversión >2% (normativa_despliegue.md), alerta >1% por 5 min (monitoreo_alertas.md) y la ventana de 15 min del canario: todo trazable."),
    ],
    "sesion-fuera": [
        ("grounded", 1.0, "Abstención correcta: no inventa cotización ni precios. Sugiere consultar el BCRA (conocimiento general, como sugerencia y no como dato)."),
    ],
    "sesion-redis": [
        ("grounded", 1.0, "85% de 2 GB, causa allkeys-lru, criticidad baja a media y la contradicción con arquitectura_sistema.md son correctas. Validación pidió aclarar la inconsistencia (2 rechazos)."),
    ],
    "sesion-celery": [
        ("grounded", 0.9, "Umbral 1000, max_retries=3, criticidad media, canales y post-mortem coinciden. Afirma que el paso 2 del Incidente 2 'quedó cortado', limitación espuria del chunking aunque el contenido sí llegó."),
    ],
    "sesion-falla": [
        ("abstencion_por_error", 1.0, "Chroma desconectada: 3 intentos fallidos con ConnectionError registrado en los spans TOOL/RETRIEVER. La respuesta no inventa el umbral de replication lag."),
    ],
}

SESION3: dict[str, list[tuple[str, float, str]]] = {
    "sesion-pool": [
        ("grounded", 1.0, "80% por 2 min, límite 100, comandos del runbook y 150: todo en los fragmentos. Señala que el paso 2 llegó cortado (efecto del chunking), lo que costó una búsqueda extra."),
        ("grounded", 1.0, "58%, 120, 33, 63 y 22 son cálculos sobre 87 (memoria del thread), 150 y 80% (fragmentos)."),
    ],
    "sesion-canario": [
        ("grounded", 1.0, "Umbrales de 2% y de 1% por 5 min en los fragmentos; 0,4 y 0,6 son cálculos."),
    ],
    "sesion-fuera": [
        ("grounded", 1.0, "Abstención correcta. Agrega 'no hubo falla técnica' sin que se pregunte: efecto de la regla nueva del sintetizador."),
    ],
    "sesion-redis": [
        ("grounded", 1.0, "85% de 2 GB, allkeys-lru, TTL de 30 min y criticidad baja a media en los fragmentos; 1,7, 87,5 y 2,5 son cálculos. Sin rechazos de validación."),
    ],
    "sesion-celery": [
        ("grounded", 0.8, "Umbral 1000, max_retries=3, FAILURE, SendGrid y canales en los fragmentos. Se contradice: cita max_retries=3 y después dice que la cantidad de reintentos no está documentada."),
    ],
    "sesion-falla": [
        ("abstencion_por_error", 1.0, "Explica que la base vectorial no respondió (ConnectionError) y sugiere reintentar; no inventa el umbral ni dice que la documentación no lo cubre."),
    ],
}

EVALUACION = {"m7-ejercicio-sesion2": SESION2, "m7-ejercicio-sesion3": SESION3}


def main(proyecto: str) -> None:
    client = Client()
    df = client.spans.get_spans_dataframe(project_identifier=proyecto, root_spans_only=True, timeout=60)
    raices = df.sort_values("start_time")
    pendientes = {s: list(v) for s, v in EVALUACION[proyecto].items()}
    for span_id, fila in raices.iterrows():
        sesion = fila["attributes.session.id"]
        if not pendientes.get(sesion):
            continue
        label, score, explicacion = pendientes[sesion].pop(0)
        client.spans.add_span_annotation(
            span_id=str(fila["context.span_id"]),
            annotation_name="grounding",
            annotator_kind="LLM",
            label=label,
            score=score,
            explanation=explicacion,
            sync=True,
        )
        print(f"{sesion}: {label} ({score})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m7-ejercicio-sesion2")
