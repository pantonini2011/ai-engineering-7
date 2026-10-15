# Normativa de despliegue — Plataforma de Pedidos

## Ambientes

El sistema tiene tres ambientes: `desarrollo`, `staging` y `producción`. Todo cambio de código
debe pasar por los tres, en ese orden, sin excepciones.

## Proceso de despliegue a producción

1. **Ventana de despliegue**: los despliegues a producción sólo se realizan de lunes a jueves,
   entre las 10:00 y las 16:00 (horario local). Está prohibido desplegar los viernes o en
   horario nocturno, salvo un hotfix de severidad crítica aprobado explícitamente por el líder
   técnico de guardia.
2. **Aprobación**: todo despliegue requiere al menos una aprobación de code review y que la
   suite de tests automatizados (`pytest`) haya pasado en el pipeline de CI sin excepciones.
3. **Migraciones de base de datos**: las migraciones de PostgreSQL se aplican con un paso previo
   de `staging` idéntico a producción en volumen de datos representativo, y siempre deben ser
   reversibles (se exige un script de rollback junto con cada migración).
4. **Rollout progresivo**: los cambios en la API de FastAPI se despliegan con rollout canario:
   primero al 10% de los pods, se observan métricas de error durante 15 minutos, y recién
   después se promueve al 100%.
5. **Rollback**: si la tasa de errores 5xx supera el 2% durante el canario, el despliegue se
   revierte automáticamente sin intervención manual.

## Gestión de configuración y secretos

- Las variables de entorno sensibles (credenciales de PostgreSQL, API key de SendGrid, credenciales
  de Redis) se gestionan con un vault centralizado, nunca en archivos de configuración versionados
  en el repositorio.
- El archivo `.env` es exclusivamente para desarrollo local y está excluido del control de
  versiones vía `.gitignore`.

## Política de versionado de la API

La API de FastAPI usa versionado explícito en la URL (`/v1/`, `/v2/`, etc.). Una versión de la
API se mantiene soportada durante un mínimo de 6 meses después de publicada la siguiente versión,
para dar tiempo a los clientes de migrar sus integraciones.

## Capacitación y checklist de guardia

Todo integrante que entra a la guardia (on-call) por primera vez debe:

1. Leer el runbook de incidentes completo.
2. Tener acceso configurado y probado al dashboard de PgBouncer y a Celery Flower.
3. Haber participado de al menos un simulacro de incidente con el equipo antes de tomar guardia
   en solitario.
