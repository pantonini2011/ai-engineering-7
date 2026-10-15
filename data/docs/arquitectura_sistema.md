# Arquitectura del sistema — Plataforma de Pedidos

## Visión general

La plataforma expone una API REST construida con **FastAPI** (Python 3.12) detrás de un
balanceador de carga Nginx. La API atiende tres dominios funcionales: catálogo, pedidos y
notificaciones. Cada dominio corre como un proceso independiente dentro del mismo cluster de
Kubernetes, pero comparten la misma base de datos relacional.

## Componentes principales

- **FastAPI**: capa de API. Expone endpoints REST versionados (`/v1/...`) y usa Pydantic para
  validar los payloads de entrada y salida. Corre con Uvicorn detrás de Gunicorn (4 workers por
  pod).
- **PostgreSQL 15**: base de datos principal. Almacena catálogo de productos, pedidos, usuarios y
  auditoría de cambios. Tiene un pool de conexiones gestionado con PgBouncer (modo `transaction`),
  con un límite de 100 conexiones por instancia.
- **Redis 7**: se usa para dos cosas distintas, en dos instancias separadas:
  1. **Caché de sesión**: guarda el token de sesión del usuario autenticado, con TTL de 30 minutos.
  2. **Cola de trabajos** (vía Celery): encola tareas asincrónicas como el envío de emails de
     confirmación de pedido y la actualización de stock.
- **Celery**: procesa las colas de Redis con workers dedicados. El worker de notificaciones tiene
  reintentos configurados (`max_retries=3`, backoff exponencial) para tareas que fallan por
  timeouts de red hacia el proveedor de email.

## Flujo de un pedido

1. El cliente autenticado hace `POST /v1/pedidos` con el carrito.
2. FastAPI valida el payload contra el schema Pydantic `PedidoCreate`.
3. Se verifica stock disponible contra PostgreSQL dentro de una transacción con `SELECT FOR
   UPDATE` para evitar condiciones de carrera entre pedidos concurrentes sobre el mismo producto.
4. Se persiste el pedido con estado `pendiente`.
5. Se encola una tarea Celery para notificar por email al cliente y otra para actualizar el
   stock agregado en la tabla de catálogo.
6. La API responde `201 Created` con el `pedido_id` antes de que las tareas asincrónicas
   terminen de procesarse.

## Consideraciones de escalabilidad

El cuello de botella histórico del sistema es el **pool de conexiones a PostgreSQL**: bajo carga
alta (más de 500 pedidos concurrentes), el límite de 100 conexiones de PgBouncer se satura y las
requests nuevas quedan esperando una conexión libre, generando timeouts intermitentes en el
endpoint `/v1/pedidos`. Este es el incidente más frecuente registrado en el runbook de
incidentes.

Redis, en cambio, no ha mostrado problemas de escalabilidad hasta el momento: el volumen de
sesiones activas y de mensajes en cola está muy por debajo de su capacidad configurada (2 GB de
memoria máxima, política `allkeys-lru`).
