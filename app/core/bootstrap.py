"""
═══════════════════════════════════════════════════════════
VisionCare AI — Bootstrap de Entorno y Configuración de Hilos
Garantiza configuración determinista de hilos para BLAS/OpenMP/MKL
antes de que las librerías C/C++ (NumPy, PyTorch, SciPy) carguen.
═══════════════════════════════════════════════════════════
"""

import os
import sys

# Variables críticas de concurrencia para evitar el fallo de OpenBLAS
# "Memory allocation still failed after 10 retries, giving up" en Windows.
DEFAULTS_HILOS = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def aplicar_configuracion_hilos() -> dict[str, str]:
    """
    Aplica las variables de entorno de limitación de hilos si no existen.
    Usa setdefault para permitir sobrescrituras explícitas del operador.
    """
    configuracion_activa = {}
    for variable, valor_por_defecto in DEFAULTS_HILOS.items():
        if variable not in os.environ:
            os.environ[variable] = valor_por_defecto
        configuracion_activa[variable] = os.environ[variable]
    return configuracion_activa


# Se ejecuta automáticamente en cuanto este módulo es importado
ESTADO_HILOS = aplicar_configuracion_hilos()
