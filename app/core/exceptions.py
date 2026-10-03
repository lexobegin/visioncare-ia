"""
═══════════════════════════════════════════════════════════
VisionCare AI — Manejo global de errores

Formato IDÉNTICO al filtro HttpExceptionFilter del backend
NestJS, para que frontend y app móvil traten los errores de
ambos servicios con la misma lógica:

{
  "ok": false,
  "codigo": 422,
  "error": "Unprocessable Entity",
  "mensaje": "...",
  "detalles": ["..."],
  "ruta": "/api/v1/...",
  "timestamp": "2026-..."
}

Casos cubiertos:
  1. HTTPException (lanazas del código, ej: seguridad)
  2. RequestValidationError (validación de Pydantic/FastAPI)
  3. Errores no controlados (500) — detalle oculto en producción
═══════════════════════════════════════════════════════════
"""

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings

logger = logging.getLogger("visioncare-ia.excepciones")


def _cuerpo_error(
    codigo: int,
    error: str,
    mensaje: str,
    ruta: str,
    detalles: list[str] | None = None,
) -> dict[str, Any]:
    """Arma el cuerpo estándar de error (mismo contrato que NestJS)."""
    cuerpo: dict[str, Any] = {
        "ok": False,
        "codigo": codigo,
        "error": error,
        "mensaje": mensaje,
        "ruta": ruta,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }
    if detalles:
        cuerpo["detalles"] = detalles
    return cuerpo


def registrar_manejadores(app: FastAPI) -> None:
    """Registra los 3 manejadores globales (se llama desde main.py)."""

    # ─── Caso 1: HTTPException (errores lanzados por el código) ───
    @app.exception_handler(StarletteHTTPException)
    async def manejar_http(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        detalle = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=_cuerpo_error(
                codigo=exc.status_code,
                error=exc.__class__.__name__.replace("HTTPException", "Error"),
                mensaje=detalle,
                ruta=str(request.url.path),
            ),
        )

    # ─── Caso 2: Validación de entrada (Pydantic) ───
    @app.exception_handler(RequestValidationError)
    async def manejar_validacion(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        detalles = [
            f"{'.'.join(str(p) for p in err['loc'][1:]) or 'body'}: {err['msg']}"
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_cuerpo_error(
                codigo=status.HTTP_422_UNPROCESSABLE_ENTITY,
                error="Unprocessable Entity",
                mensaje="La petición contiene datos inválidos.",
                ruta=str(request.url.path),
                detalles=detalles,
            ),
        )

    # ─── Caso 3: Error no controlado ───
    @app.exception_handler(Exception)
    async def manejar_generico(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Error no controlado en %s: %s",
            request.url.path,
            exc,
            exc_info=True,
        )
        detalle_interno: list[str] = (
            [] if settings.es_produccion else [str(exc)]
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_cuerpo_error(
                codigo=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error="Internal Server Error",
                mensaje="Error interno del servicio de IA. Intenta nuevamente más tarde.",
                ruta=str(request.url.path),
                detalles=detalle_interno,
            ),
        )