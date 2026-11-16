"""
═══════════════════════════════════════════════════════════
VisionCare AI — Suite de Pruebas de Resiliencia para ProductoEngine
Sprint 2: Pruebas unitarias para el motor de productos (CU13)
con fotos de alta resolución en CPU.
═══════════════════════════════════════════════════════════
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Asegurar path para importar app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai_engine.object_detector import DetectorObjetos, Deteccion, ResultadoDeteccion
from app.ai_engine.ocr_engine import OcrReader, LineaTexto, ResultadoOCR
from app.ai_engine.producto_engine import ProductoEngine
from tests.test_ocr_performance import _crear_imagen_bytes


class TestProductoResilience(unittest.TestCase):
    """Pruebas de resiliencia del motor de productos con imágenes de alta resolución."""

    def setUp(self) -> None:
        self.mock_detector = MagicMock(spec=DetectorObjetos)
        self.mock_reader = MagicMock(spec=OcrReader)
        self.engine = ProductoEngine(
            detector=self.mock_detector,
            reader_ocr=self.mock_reader,
        )

    def test_producto_con_deteccion_yolo_y_ocr_recorte(self) -> None:
        """Cuando YOLO detecta un objeto comercial, se recorta y se analiza con OCR sin error."""
        imagen_bytes = _crear_imagen_bytes(4000, 3000, "COCA COLA 500ML")

        # Mock YOLO detecta una botella (clase comercial COCO)
        self.mock_detector._inferir_sincrono.return_value = ResultadoDeteccion(
            modelo_nombre="yolov8n",
            modelo_version="1.0",
            duracion_ms=50,
            detecciones=[
                Deteccion(
                    etiqueta="bottle",
                    categoria="botella",
                    confianza=0.88,
                    x_min=500,
                    y_min=500,
                    x_max=3500,
                    y_max=2800,
                )
            ],
        )

        # Mock OCR del recorte
        self.mock_reader._leer_sincrono.return_value = ResultadoOCR(
            texto="COCA COLA 500ML",
            confianza=0.95,
            lineas=[
                LineaTexto(
                    texto="COCA COLA 500ML",
                    confianza=0.95,
                    y_centro=600,
                    x_inicio=600,
                    alto=100,
                )
            ],
        )

        resultado = self.engine._analizar_sincrono(imagen_bytes)

        self.assertEqual(len(resultado.items), 1)
        item = resultado.items[0]
        self.assertIn("coca", (item.marca or "").lower())
        self.assertEqual(self.mock_reader._leer_sincrono.call_count, 1)

    def test_producto_fallback_marca_en_foto_completa(self) -> None:
        """Cuando YOLO no detecta clase comercial, ejecuta fallback OCR sobre la foto completa."""
        imagen_bytes = _crear_imagen_bytes(4000, 3000, "PEPSI")

        # Mock YOLO sin detecciones comerciales
        self.mock_detector._inferir_sincrono.return_value = ResultadoDeteccion(
            modelo_nombre="yolov8n",
            modelo_version="1.0",
            duracion_ms=50,
            detecciones=[],
        )

        # Mock OCR foto completa reconoce marca conocida
        self.mock_reader._leer_sincrono.return_value = ResultadoOCR(
            texto="PEPSI 2L",
            confianza=0.92,
            lineas=[
                LineaTexto(
                    texto="PEPSI 2L",
                    confianza=0.92,
                    y_centro=1500,
                    x_inicio=1000,
                    alto=200,
                )
            ],
        )

        resultado = self.engine._analizar_sincrono(imagen_bytes)

        self.assertEqual(len(resultado.items), 1)
        item = resultado.items[0]
        self.assertIn("pepsi", (item.marca or "").lower())
        self.assertEqual(self.mock_reader._leer_sincrono.call_count, 1)


if __name__ == "__main__":
    unittest.main()
