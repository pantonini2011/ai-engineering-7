# Monitoreo y alertas — Plataforma de Pedidos

## Stack de observabilidad

El sistema usa Prometheus para recolectar métricas de cada componente y Grafana para
visualizarlas. Los logs estructurados (JSON) de FastAPI y de los workers de Celery se centralizan
con un stack de logging tipo ELK.

## Métricas clave por componente

### FastAPI / Nginx
- Latencia p50/p95/p99 por endpoint.
- Tasa de errores 5xx (umbral de alerta: > 1% durante 5 minutos consecutivos).
- Requests por segundo, para detectar picos de tráfico anómalos.

### PostgreSQL / PgBouncer
- Conexiones activas vs. límite configurado (100). Alerta al superar el 80% de uso sostenido
  durante más de 2 minutos, ya que este es el precursor directo del Incidente 1 del runbook
  (agotamiento del pool de conexiones).
- Duración de queries: alerta si alguna query individual supera los 5 segundos de ejecución.
- Replication lag de las réplicas de lectura (umbral: 10 segundos).

### Redis
- Uso de memoria de cada instancia (sesión y cola), con alerta al 85% del límite configurado
  (2 GB), anticipando el Incidente 3 del runbook.
- Longitud de las colas de Celery: una cola de más de 1000 mensajes pendientes dispara una
  alerta de "backlog anómalo", ya sea por un worker caído o por un proveedor externo (ej.
  SendGrid) devolviendo errores de forma sostenida.

### Celery
- Tasa de tareas en estado `FAILURE` tras agotar los reintentos.
- Tiempo de procesamiento promedio por tipo de tarea.

## Canales de alerta

Las alertas de severidad **alta** (ej. tasa de 5xx elevada, pool de PostgreSQL agotado) se envían
simultáneamente a Slack (#alertas-produccion) y como llamada telefónica automática al ingeniero de
guardia vía PagerDuty. Las alertas de severidad **media** y **baja** solo van a Slack, sin
llamada.

## Dashboards de referencia

- **Dashboard "Salud de API"**: latencia, tasa de error y throughput de FastAPI por endpoint.
- **Dashboard "Base de datos"**: conexiones activas, queries lentas y replication lag de
  PostgreSQL/PgBouncer.
- **Dashboard "Colas asincrónicas"**: longitud de colas de Redis y tasa de éxito/fallo de
  tareas de Celery.

Todo incidente que dispare una alerta de severidad alta o media debe cerrarse con un post-mortem
breve, documentando causa raíz y acción correctiva, siguiendo el mismo criterio que las causas
raíz descriptas en el runbook de incidentes.
