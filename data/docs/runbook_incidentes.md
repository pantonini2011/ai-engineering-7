# Runbook de incidentes — Plataforma de Pedidos

Este documento describe los incidentes conocidos, cómo detectarlos y cómo resolverlos. Está
pensado para el equipo de guardia (on-call).

## Incidente 1: Agotamiento del pool de conexiones a PostgreSQL

**Síntoma**: el endpoint `POST /v1/pedidos` empieza a devolver errores 504 (Gateway Timeout) o
tarda más de 10 segundos en responder. En los logs de FastAPI aparece
`TimeoutError: connection pool exhausted` proveniente de PgBouncer.

**Causa raíz habitual**: picos de tráfico (más de 500 pedidos concurrentes) combinados con
queries lentas que retienen la conexión más tiempo del esperado — típicamente una query de
reporte administrativo ejecutada por error contra la base de producción en horario pico.

**Resolución**:
1. Verificar en el dashboard de PgBouncer cuántas conexiones están activas vs. el límite (100).
2. Si hay una query "colgada" reteniendo conexiones, identificarla con
   `SELECT * FROM pg_stat_activity WHERE state = 'active' ORDER BY query_start;` y, si corresponde,
   cancelarla con `pg_cancel_backend(pid)`.
3. Si el problema es puramente de volumen (no hay queries colgadas), escalar horizontalmente el
   número de réplicas de PgBouncer o subir temporalmente el límite de conexiones a 150.
4. Nivel de criticidad: **alta** (impacto directo en la capacidad de compra de los usuarios).

## Incidente 2: Emails de confirmación no se envían

**Síntoma**: el pedido se crea correctamente (`201 Created`), pero el cliente nunca recibe el
email de confirmación. En Celery Flower se ve la tarea `enviar_confirmacion` en estado `RETRY`
repetidas veces.

**Causa raíz habitual**: el proveedor de email externo (SendGrid) está devolviendo timeouts o
rate-limiting temporal.

**Resolución**:
1. Revisar el estado de SendGrid en su página de status pública.
2. Si SendGrid está caído, no hay acción inmediata posible: los reintentos de Celery
   (`max_retries=3`, backoff exponencial) ya están configurados y reintentarán solos. Si tras el
   tercer intento la tarea pasa a `FAILURE`, queda registrada en una tabla de notificaciones
   fallidas para reenvío manual posterior.
3. Nivel de criticidad: **media** (no bloquea la compra, pero afecta la experiencia del cliente).

## Incidente 3: Sesiones de usuario se pierden antes de tiempo

**Síntoma**: usuarios reportan que tienen que volver a loguearse con más frecuencia de la
esperada, antes de los 30 minutos configurados de TTL.

**Causa raíz habitual**: la instancia de Redis usada como caché de sesión alcanzó su límite de
memoria (2 GB) y la política `allkeys-lru` empezó a desalojar claves de sesión activas para
hacer lugar a las nuevas, en vez de solo expirar por TTL.

**Resolución**:
1. Revisar el uso de memoria de la instancia de Redis de sesión (no la de cola de Celery, son
   instancias separadas).
2. Si está cerca del límite, es señal de que hay más usuarios concurrentes de los previstos en el
   dimensionamiento original: evaluar subir el límite de memoria en vez de solo monitorear.
3. Nivel de criticidad: **baja a media**, según cuántos usuarios estén siendo afectados.

## Incidente 4: Condición de carrera en actualización de stock

**Síntoma**: el stock de un producto queda en un número negativo o inconsistente tras un pico de
compras simultáneas del mismo producto.

**Causa raíz habitual**: en versiones antiguas del código, la verificación de stock no usaba
`SELECT FOR UPDATE`, permitiendo que dos transacciones concurrentes leyeran el mismo stock
disponible antes de que ninguna de las dos confirmara su descuento.

**Resolución**: este incidente está mitigado desde que se introdujo el `SELECT FOR UPDATE` en el
paso 3 del flujo de creación de pedidos (ver `arquitectura_sistema.md`). Si reaparece, verificar
que ningún endpoint nuevo esté sorteando ese bloqueo con una query directa fuera del flujo
estándar.
