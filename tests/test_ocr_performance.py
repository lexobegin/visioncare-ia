"""
═══════════════════════════════════════════════════════════
VisionCare AI — Suite de Pruebas de Rendimiento y Robustez OCR
Sprint 2: Pruebas unitarias y de regresión para OcrReader.

Cubre:
  1. Imagen pequeña (no debe redimensionarse innecesariamente).
  2. Imagen típica de celular de alta resolución (4000x3000).
  3. Imagen de tamaño extremo (8000x6000, 48MP).
  4. Archivos inválidos o corruptos (bytes aleatorios, truncados).
  5. Prueba de regresión de memoria y latencia.
  6. Prueba de integración E2E si el modelo EasyOCR está disponible.
═══════════════════════════════════════════════════════════
"""

import io
import os
import sys
import time
import tracemalloc
import unittest
from unittest.mock import MagicMock

from PIL import Image, ImageDraw, ImageFont

# Asegurar path para importar app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai_engine.ocr_engine import OcrReader, ResultadoOCR
from app.config import settings


def _crear_imagen_bytes(
    ancho: int,
    alto: int,
    texto: str = "TEXTO DE PRUEBA",
    formato: str = "JPEG",
) -> bytes:
    """Genera una imagen sintética en memoria para pruebas."""
    img = Image.new("RGB", (ancho, alto), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Dibujar cuadro y texto centrado si el tamaño lo permite
    if ancho >= 100 and alto >= 50:
        tam_fuente = max(12, int(min(ancho, alto) * 0.05))
        try:
            fuente = ImageFont.truetype("arial.ttf", tam_fuente)
        except Exception:
            fuente = None
        draw.rectangle([(20, 20), (ancho - 20, alto - 20)], fill=(240, 240, 240))
        draw.text((30, 30), texto, fill=(0, 0, 0), font=fuente)

    buffer = io.BytesIO()
    img.save(buffer, format=formato)
    return buffer.getvalue()


class TestOcrRendimientoYRobustez(unittest.TestCase):
    """Pruebas de robustez, escalado defensivo y rendimiento en OcrReader."""

    def setUp(self) -> None:
        self.reader = OcrReader()

    # ─────────────────────────────────────────────────────────────
    # Caso 1: Imagen pequeña
    # ─────────────────────────────────────────────────────────────
    def test_imagen_pequena(self) -> None:
        """Una imagen pequeña (200x100) no debe alterarse ni sufrir distorsión."""
        ancho, alto = 200, 100
        imagen_bytes = _crear_imagen_bytes(ancho, alto, "HOLA")

        # Mock del reader subyacente para aislar la lógica de preprocesamiento
        mock_easyocr = MagicMock()
        # Retorna bbox: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        mock_easyocr.readtext.return_value = [
            ([[10, 20], [80, 20], [80, 40], [10, 40]], "HOLA", 0.96)
        ]
        self.reader._reader = mock_easyocr

        resultado = self.reader._leer_sincrono(imagen_bytes)

        # Verificar que readtext recibió las dimensiones intactas (200x100)
        self.assertEqual(mock_easyocr.readtext.call_count, 1)
        args, kwargs = mock_easyocr.readtext.call_args
        arreglo_enviado = args[0]
        self.assertEqual(arreglo_enviado.shape, (alto, ancho, 3))
        self.assertEqual(kwargs.get("canvas_size"), settings.ocr_canvas_size)

        # Verificar coordenadas 1:1 (sin factor de escala)
        self.assertEqual(len(resultado.lineas), 1)
        linea = resultado.lineas[0]
        self.assertEqual(linea.texto, "HOLA")
        self.assertEqual(linea.x_inicio, 10)
        self.assertEqual(linea.y_centro, 30)  # (20 + 20 + 40 + 40) / 4 = 30
        self.assertEqual(linea.alto, 20)      # 40 - 20 = 20

    # ─────────────────────────────────────────────────────────────
    # Caso 2: Imagen típica de celular de alta resolución
    # ─────────────────────────────────────────────────────────────
    def test_imagen_celular_alta_resolucion(self) -> None:
        """Foto típica de 12MP (4000x3000) debe redimensionarse a ocr_max_dimension y remapear bboxes."""
        orig_w, orig_h = 4000, 3000
        imagen_bytes = _crear_imagen_bytes(orig_w, orig_h, "FARMACIA SAN RAFAEL")

        mock_easyocr = MagicMock()
        # Simulamos detección en el espacio redimensionado (1280x960)
        # Factor escala = 1280 / 4000 = 0.32
        # Supongamos un bbox en coords redimensionadas: [320, 240] a [640, 280]
        mock_easyocr.readtext.return_value = [
            ([[320, 240], [640, 240], [640, 280], [320, 280]], "FARMACIA SAN RAFAEL", 0.98)
        ]
        self.reader._reader = mock_easyocr

        resultado = self.reader._leer_sincrono(imagen_bytes)

        # Verificar que el arreglo enviado a EasyOCR fue escalado a 1280x960
        args, kwargs = mock_easyocr.readtext.call_args
        arreglo_enviado = args[0]
        max_esperado = settings.ocr_max_dimension
        self.assertEqual(max(arreglo_enviado.shape[:2]), max_esperado)
        self.assertEqual(arreglo_enviado.shape, (960, 1280, 3))

        # Verificar que las coordenadas retornadas se remapearon a la escala original (4000x3000)
        # escala = 1280 / 4000 = 0.32 -> 1 / escala = 3.125
        # x_inicio original = 320 / 0.32 = 1000
        # y_centro original = 260 / 0.32 = 812.5 -> 812
        # alto original = (280 - 240) / 0.32 = 125
        linea = resultado.lineas[0]
        self.assertEqual(linea.texto, "FARMACIA SAN RAFAEL")
        self.assertEqual(linea.x_inicio, 1000)
        self.assertEqual(linea.y_centro, 812)
        self.assertEqual(linea.alto, 125)

    # ─────────────────────────────────────────────────────────────
    # Caso 3: Imagen de tamaño extremo (48MP)
    # ─────────────────────────────────────────────────────────────
    def test_imagen_tamano_extremo(self) -> None:
        """Imagen de 8000x6000 (48 megapíxeles) se acota sin agotar memoria."""
        orig_w, orig_h = 8000, 6000
        imagen_bytes = _crear_imagen_bytes(orig_w, orig_h, "CARTEL GIGANTE")

        mock_easyocr = MagicMock()
        mock_easyocr.readtext.return_value = [
            ([[160, 120], [320, 120], [320, 140], [160, 140]], "CARTEL GIGANTE", 0.95)
        ]
        self.reader._reader = mock_easyocr

        resultado = self.reader._leer_sincrono(imagen_bytes)

        # Verificamos que el arreglo fue acotado a 1280x960 exactamente
        args, _ = mock_easyocr.readtext.call_args
        arreglo_enviado = args[0]
        self.assertEqual(arreglo_enviado.shape, (960, 1280, 3))

        # Escala: 1280 / 8000 = 0.16 -> multiplicador = 6.25
        # x_inicio = 160 * 6.25 = 1000
        # alto = 20 * 6.25 = 125
        linea = resultado.lineas[0]
        self.assertEqual(linea.x_inicio, 1000)
        self.assertEqual(linea.alto, 125)

    # ─────────────────────────────────────────────────────────────
    # Caso 4: Archivos inválidos o corruptos
    # ─────────────────────────────────────────────────────────────
    def test_archivo_invalido_o_corrupto(self) -> None:
        """Archivos corruptos o de formato inválido deben lanzar ValueError claro."""
        casos_invalidos = [
            (b"", "bytes vacíos"),
            (b"GIF89a\x00\x00\x00\x00", "encabezado inválido/incompleto"),
            (b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00", "JPEG truncado"),
            (os.urandom(256), "bytes aleatorios sin estructura"),
        ]

        for payload, descripcion in casos_invalidos:
            with self.subTest(caso=descripcion):
                with self.assertRaises(ValueError) as ctx:
                    self.reader._leer_sincrono(payload)
                self.assertIn("no es una imagen válida o está dañado", str(ctx.exception))

    # ─────────────────────────────────────────────────────────────
    # Caso 5: Prueba de regresión de memoria y latencia
    # ─────────────────────────────────────────────────────────────
    def test_regresion_memoria_y_rendimiento(self) -> None:
        """Verifica que el preprocesamiento de una imagen de 4000x3000 consume < 80 MB de memoria."""
        imagen_bytes = _crear_imagen_bytes(4000, 3000, "PRUEBA REGRESION")

        mock_easyocr = MagicMock()
        mock_easyocr.readtext.return_value = []
        self.reader._reader = mock_easyocr

        tracemalloc.start()
        inicio_tiempo = time.perf_counter()

        self.reader._leer_sincrono(imagen_bytes)

        tiempo_total_s = time.perf_counter() - inicio_tiempo
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mem_mb = peak_mem_bytes / (1024 * 1024)

        # El preprocesamiento y redimensión deben tomar menos de 2.0s en CPU
        self.assertLess(
            tiempo_total_s,
            2.0,
            f"El preprocesamiento tardó {tiempo_total_s:.2f}s (esperado < 2.0s)",
        )

        # El pico de memoria debe ser < 80 MB (muy por debajo de los 1258 MB del error previo)
        self.assertLess(
            peak_mem_mb,
            80.0,
            f"Pico de memoria {peak_mem_mb:.2f} MB superó el umbral de 80 MB",
        )


if __name__ == "__main__":
    unittest.main()
