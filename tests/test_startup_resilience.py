"""
═══════════════════════════════════════════════════════════
VisionCare AI — Suite de Pruebas de Arranque y Resiliencia OpenBLAS
Verifica la inicialización reproducible del entorno, la configuración
automática de hilos y la estabilidad ante concurrencia en Windows/CPU.
═══════════════════════════════════════════════════════════
"""

import concurrent.futures
import os
import sys
import unittest

# Asegurar path para importar app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core import bootstrap
from app.config import settings
import numpy as np


class TestStartupYConcurrenciaOpenBLAS(unittest.TestCase):
    """Pruebas de verificación de entorno y resiliencia de hilos OpenBLAS/PyTorch."""

    def test_bootstrap_exporta_variables_de_entorno(self) -> None:
        """Verifica que el módulo bootstrap configura las variables críticas en os.environ."""
        variables_esperadas = [
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ]
        for var in variables_esperadas:
            self.assertIn(
                var,
                os.environ,
                f"La variable {var} debe existir en os.environ para prevenir fallos en Windows",
            )
            self.assertEqual(os.environ[var], "1")

    def test_config_settings_exporta_valores_de_concurrencia(self) -> None:
        """Verifica que Pydantic Settings contiene y propaga los límites de hilos."""
        self.assertEqual(settings.openblas_num_threads, "1")
        self.assertEqual(settings.omp_num_threads, "1")
        self.assertEqual(settings.mkl_num_threads, "1")
        self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), "1")

    def test_concurrencia_hilos_sin_fallo_openblas(self) -> None:
        """
        Simula múltiples hilos ejecutando operaciones BLAS concurrentes (como ocurre con
        asyncio.to_thread en FastAPI) y confirma que no hay errores de reserva en OpenBLAS.
        """
        def _tarea_algebra_lineal(semilla: int) -> float:
            rng = np.random.default_rng(semilla)
            mat_a = rng.standard_normal((300, 300), dtype=np.float32)
            mat_b = rng.standard_normal((300, 300), dtype=np.float32)
            producto = np.matmul(mat_a, mat_b)
            return float(np.trace(producto))

        # Lanzar 16 tareas distribuidas en 8 hilos worker
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ejecutor:
            futuros = [ejecutor.submit(_tarea_algebra_lineal, i) for i in range(16)]
            resultados = [f.result() for f in concurrent.futures.as_completed(futuros)]

        self.assertEqual(len(resultados), 16)
        for val in resultados:
            self.assertIsInstance(val, float)

    def test_fastapi_app_instanciada_correctamente(self) -> None:
        """Verifica que la app FastAPI se importa limpiamente y monta las rutas esperadas."""
        from app.main import app

        self.assertEqual(app.title, settings.app_name)
        schema = app.openapi()
        rutas = list(schema.get("paths", {}).keys())

        # Verificar presencia de endpoints clave
        self.assertTrue(any("/health" in r for r in rutas))
        self.assertTrue(any("/ocr/leer" in r for r in rutas))
        self.assertTrue(any("/detecciones/objetos" in r for r in rutas))
        self.assertTrue(any("/productos/reconocer" in r for r in rutas))


if __name__ == "__main__":
    unittest.main()
