"""
VisionCare AI — Motores de IA (YOLO, OCR, color, productos,
billetes, señales, escena). Cada motor encapsula su lógica y
expone una API async simple.
"""

from app.ai_engine.billete_engine import BilleteEngine, ResultadoBilleteInterno
from app.ai_engine.color_engine import ColorAnalyzer, ResultadoColorInterno
from app.ai_engine.escena_engine import (
    EscenaEngine,
    MencionEscena,
    ResultadoEscenaInterno,
)
from app.ai_engine.object_detector import DetectorObjetos
from app.ai_engine.ocr_engine import LineaTexto, OcrReader, ResultadoOCR
from app.ai_engine.producto_engine import (
    ItemProductoInterno,
    ProductoEngine,
    ResultadoProductosInterno,
)
from app.ai_engine.senal_engine import (
    ItemSenalInterno,
    ResultadoSenalesInterno,
    SenalEngine,
)
from app.ai_engine.traducciones import (
    TRADUCCION_COCO_ES,
    categorizar_producto,
    color_desde_hsv,
    detectar_palabra_senal,
    traducir_etiqueta,
)

__all__ = [
    "BilleteEngine",
    "ResultadoBilleteInterno",
    "ColorAnalyzer",
    "ResultadoColorInterno",
    "EscenaEngine",
    "MencionEscena",
    "ResultadoEscenaInterno",
    "DetectorObjetos",
    "LineaTexto",
    "OcrReader",
    "ResultadoOCR",
    "ItemProductoInterno",
    "ProductoEngine",
    "ResultadoProductosInterno",
    "ItemSenalInterno",
    "ResultadoSenalesInterno",
    "SenalEngine",
    "TRADUCCION_COCO_ES",
    "traducir_etiqueta",
    "detectar_palabra_senal",
    "categorizar_producto",
    "color_desde_hsv",
]