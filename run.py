"""
═══════════════════════════════════════════════════════════
VisionCare AI — Script de Arranque Estandarizado (DevOps)
Uso:
    python run.py
    python run.py --port 8000 --no-reload
═══════════════════════════════════════════════════════════
"""

import argparse
import os
import sys

# ─── 1. Blindaje de variables de entorno para BLAS/OpenMP/MKL ───
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# Asegurar path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from app.config import settings


def main():
    parser = argparse.ArgumentParser(description="Arranque de VisionCare AI Servicio de IA")
    parser.add_argument("--host", default="0.0.0.0", help="Dirección IP de escucha (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=settings.port, help=f"Puerto de escucha (default: {settings.port})")
    parser.add_argument("--reload", action="store_true", default=not settings.es_produccion, help="Activar recarga automática en cambios")
    parser.add_argument("--no-reload", action="store_false", dest="reload", help="Desactivar recarga automática")

    args = parser.parse_args()

    print(
        f"Iniciando {settings.app_name} en http://{args.host}:{args.port} "
        f"(entorno: {settings.node_env}, reload: {args.reload})"
    )
    print(
        f"Límites de hilos CPU: OPENBLAS={os.environ['OPENBLAS_NUM_THREADS']}, "
        f"OMP={os.environ['OMP_NUM_THREADS']}, MKL={os.environ['MKL_NUM_THREADS']}"
    )

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
