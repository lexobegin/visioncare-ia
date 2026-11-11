"""
VisionCare AI — Servicios de orquestación (lógica de negocio IA).
"""

from app.services.billete_service import BilleteService, billete_service
from app.services.color_service import ColorService, color_service
from app.services.deteccion_service import DeteccionService, deteccion_service
from app.services.escena_service import EscenaService, escena_service
from app.services.monitoreo_service import monitoreo
from app.services.ocr_service import OcrService, ocr_service
from app.services.producto_service import ProductoService, producto_service
from app.services.senal_service import SenalService, senal_service

__all__ = [
    "BilleteService",
    "billete_service",
    "ColorService",
    "color_service",
    "DeteccionService",
    "deteccion_service",
    "monitoreo",
    "OcrService",
    "ocr_service",
    "ProductoService",
    "producto_service",
    "SenalService",
    "senal_service",
]