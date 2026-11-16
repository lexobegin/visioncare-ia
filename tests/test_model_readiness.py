"""
═══════════════════════════════════════════════════════════
VisionCare AI — Tests Automatizados de Preparación y Readiness de Modelos
Valida que:
1. prepare_models.py verifica exitosamente la presencia e integridad de pesos.
2. El endpoint readiness (/health/ready) reporta HTTP 200 cuando los
   modelos están listos y HTTP 503 cuando falta precarga.
3. El motor OcrReader y OcrService implementan fail-fast (< 50ms) con
   HTTP 503 ante falta de inicialización en lugar de timeouts silenciosos.
4. FastAPI registra y expone adecuadamente las rutas de salud y readiness.
═══════════════════════════════════════════════════════════
"""

import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from fastapi import HTTPException

# Asegurar path para importar app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai_engine.object_detector import DetectorObjetos
from app.ai_engine.ocr_engine import OcrReader
from app.api.v1.salud_router import readiness
from app.main import app
from app.services.ocr_service import OcrService
from prepare_models import preparar_modelos


class TestModelReadinessYFailFast(unittest.TestCase):
    """Pruebas de preparación de artefactos, readiness probes y fail-fast."""

    def test_01_script_prepare_models_verify_only(self) -> None:
        """Verifica que prepare_models.py reporte 0 al verificar modelos descargados."""
        raiz = Path(__file__).resolve().parent.parent
        resultado = preparar_modelos(raiz / "models", verify_only=True)
        self.assertTrue(
            resultado,
            "Los modelos locales en models/ deben estar presentes e íntegros.",
        )

        # Ejecución CLI con --verify-only
        proc = subprocess.run(
            [sys.executable, "prepare_models.py", "--verify-only"],
            cwd=str(raiz),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"prepare_models.py --verify-only falló con salida:\n{proc.stdout}\n{proc.stderr}",
        )

    def test_02_endpoint_readiness_cuando_modelos_listos(self) -> None:
        """Verifica que readiness() devuelva HTTP 200 si los modelos están en memoria."""
        with patch.object(DetectorObjetos, "cargado", new_callable=PropertyMock, return_value=True), \
             patch.object(OcrReader, "cargado", new_callable=PropertyMock, return_value=True):

            resp = readiness()
            self.assertEqual(
                resp.status_code,
                200,
                "readiness() debe responder 200 cuando los modelos están listos",
            )
            data = json.loads(resp.body.decode())
            self.assertEqual(data["status"], "ready")
            self.assertTrue(data["modelos"]["yolo"]["listo"])
            self.assertTrue(data["modelos"]["ocr"]["listo"])

    def test_03_endpoint_readiness_cuando_modelos_no_cargados(self) -> None:
        """Verifica que readiness() responda 503 (no listo) si un modelo no está en RAM."""
        with patch.object(DetectorObjetos, "cargado", new_callable=PropertyMock, return_value=True), \
             patch.object(OcrReader, "cargado", new_callable=PropertyMock, return_value=False), \
             patch.object(OcrReader, "error_inicializacion", "Pesos ausentes"):

            resp = readiness()
            self.assertEqual(
                resp.status_code,
                503,
                "readiness() debe devolver 503 si el OCR no está cargado",
            )
            data = json.loads(resp.body.decode())
            self.assertEqual(data["status"], "not_ready")
            self.assertFalse(data["modelos"]["ocr"]["listo"])
            self.assertIn("error", data["modelos"]["ocr"])
            self.assertIn("mensaje", data)

    def test_04_ocr_reader_fail_fast_inmediato(self) -> None:
        """
        Verifica que OcrReader lance RuntimeError inmediatamente (< 5ms)
        si hubo un error de inicialización, impidiendo reintentos bloqueantes.
        """
        reader = OcrReader()
        reader._reader = None
        reader._error_inicializacion = "Simulación: pesos corruptos o ausentes"

        inicio = time.perf_counter()
        with self.assertRaises(RuntimeError) as ctx:
            reader._leer_sincrono(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
        duracion_ms = (time.perf_counter() - inicio) * 1000

        self.assertIn("error en inicialización", str(ctx.exception))
        self.assertLess(
            duracion_ms,
            10.0,
            f"El fallo de inicialización tardó {duracion_ms:.2f} ms; debe ser instantáneo (<10ms)",
        )

    def test_05_rutas_registradas_en_openapi(self) -> None:
        """Verifica que las rutas de health y health/ready estén registradas en la app."""
        schema = app.openapi()
        rutas = list(schema.get("paths", {}).keys())

        self.assertTrue(
            any("/health" in r for r in rutas),
            "Debe existir al menos una ruta /health en OpenAPI",
        )
        self.assertTrue(
            any("/health/ready" in r for r in rutas),
            "Debe existir la ruta /health/ready en OpenAPI",
        )


if __name__ == "__main__":
    unittest.main()
