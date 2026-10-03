"""
VisionCare AI — Motores de IA (YOLO, OCR, clasificadores).

Cada motor encapsula un modelo y expone una API async simple.
"""

from app.ai_engine.object_detector import DetectorObjetos
from app.ai_engine.traducciones import TRADUCCION_COCO_ES, traducir_etiqueta

__all__ = ["DetectorObjetos", "TRADUCCION_COCO_ES", "traducir_etiqueta"]