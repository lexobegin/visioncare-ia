"""
═══════════════════════════════════════════════════════════
VisionCare AI — Seguridad del servicio IA (doble capa)

CAPA 1 — X-API-Key (servidor a servidor):
    Si IA_API_KEY está configurada, TODA petición debe enviar
    el header:  X-API-Key: <valor>
    Compara en tiempo constante (secrets.compare_digest) para
    evitar ataques de timing.

CAPA 2 — Bearer JWT (usuario final, reenviado por NestJS):
    Valida firma (mismo JWT_SECRET que NestJS), algoritmo y
    expiración; valida issuer solo si está configurado. Devuelve los datos para
    atribuir la ejecución en registros_ia (monitoreo).
═══════════════════════════════════════════════════════════
"""

import logging
import secrets
from typing import Any

import jwt  # PyJWT
from fastapi import Depends, Header, HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════ CAPA 1: API KEY ═══════════════════


async def verificar_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Exige X-API-Key SOLO si IA_API_KEY está configurada."""
    if not settings.api_key_activa:
        return  # desarrollo local sin API key → capa desactivada

    if not x_api_key or not secrets.compare_digest(
        x_api_key, settings.ia_api_key
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key ausente o inválida (header X-API-Key).",
        )


# ═══════════════════ CAPA 2: JWT ═══════════════════


async def obtener_usuario_actual(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """
    Valida el Bearer token (emitido por NestJS) y devuelve
    los datos del usuario autenticado.

    Uso en routers:
        @router.post("/detecciones/objetos")
        async def endpoint(
            usuario: dict = Depends(seguridad_reconocimiento), ...
        ):
            usuario["id_usuario"]
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token ausente. Envía el header: Authorization: Bearer <token>.",
        )

    partes = authorization.split()
    if len(partes) != 2 or partes[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Formato de autorización inválido. Usa: Bearer <token>.",
        )

    try:
        payload = jwt.decode(
            partes[1],
            settings.jwt_secret,
            algorithms=["HS256"],
            issuer=settings.jwt_issuer or None,
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError as error:
        if not settings.es_produccion:
            logger.warning(
                "JWT rechazado (%s): %s", type(error).__name__, error
            )

        if isinstance(error, jwt.ExpiredSignatureError):
            detail = "El token ha expirado."
        elif isinstance(error, jwt.InvalidIssuerError):
            detail = "El token no fue emitido por el issuer configurado."
        else:
            detail = "Token inválido."

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
        )

    sub = payload.get("sub")
    if sub is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token sin identificación de usuario.",
        )

    return {
        "id_usuario": int(sub),
        "correo": payload.get("correo", ""),
        "rol": payload.get("rol", ""),
    }


# ═══════════════════ DEPENDENCIA COMBINADA ═══════════════════


async def seguridad_reconocimiento(
    _api_key: None = Depends(verificar_api_key),
    usuario: dict[str, Any] = Depends(obtener_usuario_actual),
) -> dict[str, Any]:
    """
    Dependencia completa para los endpoints de análisis IA:
    aplica capa 1 + capa 2 y entrega el usuario autenticado.
    """
    return usuario