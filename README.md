# Instrumentación y análisis de trazas con Phoenix

Ejercicio del Módulo 7 de AI Engineering (Coderhouse).

Se instrumentó con **Arize Phoenix** (OpenTelemetry + OpenInference) el orquestador multiagente del
Módulo 6: un supervisor en LangGraph que delega en un investigador (búsqueda en ChromaDB) y en un
analista (cálculos), valida con una rúbrica y cierra con un sintetizador. Con las trazas se midieron
latencia, tokens, costo y grounding, se provocó una falla y se corrigieron tres defectos del grafo que
la traza dejó a la vista.

El análisis completo, con gráficos y conclusiones, está en el documento
[Instrumentación y análisis de trazas con Phoenix](https://claude.ai/code/artifact/f21f3287-4219-49ea-85d8-56fae96224fb).

## Qué se agregó al proyecto del Módulo 6

- **`observability.py`**: registra el tracer de Phoenix, instrumenta LangChain/LangGraph y agrega
  spans manuales para la recuperación (`chroma.retriever` de tipo RETRIEVER con los documentos
  devueltos, el embedding de la consulta y la query a Chroma por separado).
- **`main.py`**: un proyecto de Phoenix por sesión (`--proyecto`), `session.id` = `thread_id`,
  metadatos de versión de prompt y dos consultas nuevas (Redis y Celery).
- **Falla provocada** (`--falla`): agrega un turno con la base vectorial "desconectada"
  (`ConnectionError` en `tools/custom_tools.py`), que queda registrado como ERROR en los spans.
- **Correcciones del grafo** detectadas en la traza (`agents/supervisor.py`, `agents/analyst_agent.py`):
  1. Cuando el código bloquea una delegación por tope, la bitácora lo registra como bloqueo y no como
     delegación.
  2. El validador ya no rechaza por datos faltantes que el supervisor ya dio por imposibles de
     conseguir: pasan a limitaciones. El texto que lee el usuario es distinto de la instrucción interna.
  3. El sintetizador recibe las fallas técnicas y explica que no se pudo consultar la documentación,
     en vez de decir que el dato no está documentado.
- **`analisis/`**: scripts que leen las trazas desde Phoenix y los resultados de cada sesión.
- **Tests**: `tests/test_observabilidad.py` (jerarquía de spans, documentos del retriever, falla
  como ERROR) y `TestCorreccionesModulo7` en `tests/test_orquestador.py`.

## Resultados principales

| | Sesión 2 (línea de base) | Sesión 3 (con correcciones) |
|---|---|---|
| Spans | 722 | 791 |
| Latencia por turno p50 / p95 | 43,3 s / 87,2 s | 46,0 s / 116,6 s |
| Tokens entrada / salida | 305.281 / 60.745 | 349.159 / 74.676 |
| Costo | USD 0,061 | USD 0,072 |

- El LLM explica el 99 % del tiempo; la recuperación (embedding ~60 ms, query a Chroma ~3 ms) y la
  orquestación son marginales. La latencia de cada llamada crece con los tokens de salida
  (≈ 1,16 s + 3,7 ms por token, R² 0,93), no con los de entrada.
- El razonamiento más largo de la sesión 2 fue la consulta de Redis: 22 llamadas al LLM y 2 rechazos
  de validación. Con las correcciones bajó a 12 llamadas y 0 rechazos.
- Grounding contra los fragmentos que Chroma devolvió en cada traza: no se encontraron cifras
  inventadas. Las evaluaciones quedaron como anotaciones `grounding` en Phoenix.
- Con la base caída, la respuesta pasó de "no puedo responder" a explicar la falla y sugerir reintentar.

## Cómo correrlo

Requiere Python 3.12 y una API key de Anthropic.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # en Linux/macOS: .venv/bin/pip
cp .env.example .env                            # completar ANTHROPIC_API_KEY
```

En una terminal, levantar Phoenix (UI en http://localhost:6006):

```bash
python -m phoenix.server.main serve
```

En otra, correr la demo (la primera vez indexa `data/docs` en ChromaDB):

```bash
python main.py --proyecto m7-ejercicio-sesion3 --falla
```

Otras opciones: `--chat` (modo interactivo), `--thread-id`, `--sin-trazas`, `--reindexar`.
En Windows conviene definir `PYTHONUTF8=1`.

Análisis sobre un proyecto ya cargado en Phoenix:

```bash
python analisis/analizar_trazas.py m7-ejercicio-sesion3        # resumen.json y spans.csv
python analisis/grounding_vs_fragmentos.py m7-ejercicio-sesion3
python analisis/anotar_grounding.py m7-ejercicio-sesion3       # sube las anotaciones a Phoenix
```

Tests y tipos:

```bash
python -m pytest     # 45 tests, sin llamadas a la API
python -m mypy
```

## Notas

- `anthropic` 1.x trae su propio trazado OpenTelemetry, que al registrarse Phoenix genera spans
  huérfanos y duplica tokens y costo. `observability.py` lo desactiva con
  `ANTHROPIC_OPEN_TELEMETRY=false`.
- `configurar_trazas()` debe llamarse antes de crear los clientes del LLM.
- Phoenix 20.22 exige `pydantic` ≥ 2.14 (por eso la versión en `requirements.txt`).
- `traces/` guarda la bitácora de delegación y los logs de las corridas; los spans exportados de la
  sesión 2 están en `traces/*.jsonl`.
