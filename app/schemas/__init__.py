"""
VisionCare AI — Esquemas Pydantic (contratos request/response).
S1: detección de objetos · S2: OCR, señales, productos,
colores, billetes, descripción.
"""

from app.schemas.deteccion import (
    DeteccionItem,
    ModeloInfo,
    RespuestaDeteccion,
)
from app.schemas.analisis import (
    ItemProducto,
    ItemSenal,
    ResultadoBillete,
    ResultadoColores,
    ResultadoDescripcion,
    ResultadoOcr,
    ResultadoProductos,
    ResultadoSenales,
)

__all__ = [
    # S1
    "DeteccionItem",
    "ModeloInfo",
    "RespuestaDeteccion",
    # S2
    "ItemSenal",
    "ItemProducto",
    "ResultadoOcr",
    "ResultadoSenales",
    "ResultadoProductos",
    "ResultadoColores",
    "ResultadoBillete",
    "ResultadoDescripcion",
]