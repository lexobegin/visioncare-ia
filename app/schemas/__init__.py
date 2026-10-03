"""
VisionCare AI — Esquemas Pydantic (contratos request/response).
"""

from app.schemas.deteccion import (
    DeteccionItem,
    ModeloInfo,
    RespuestaDeteccion,
)

__all__ = ["DeteccionItem", "ModeloInfo", "RespuestaDeteccion"]