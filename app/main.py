"""
═══════════════════════════════════════════════════════════
VisionCare AI — Punto de entrada del servicio IA (FastAPI)

Arranque:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Lifespan:
    1. init_db()        → crea tablas propias si faltan
    2. precargar YOLO   → warmup (primera petición = rápida)
    3. shutdown         → cierra el pool de conexiones

Tolerancia a fallos en startup: si la BD o la descarga del
modelo fallan, el servicio IGUAL levanta (la detección es la
función crítica; el monitoreo es best-effort y la carga del
modelo puede reintentarse en la primera petición).

Docs:
    - Swagger UI:  http://localhost:8000/docs
    - ReDoc:       http://localhost:8000/redoc
═══════════════════════════════════════════════════════════
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core.exceptions import registrar_manejadores
from app.database import cerrar_conexiones
from app.init_db import init_db
from app.services import deteccion_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("visioncare-ia.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida: startup → atender peticiones → shutdown."""

    # ═══════════════ STARTUP ═══════════════
    logger.info("=== Iniciando %s (entorno: %s) ===", settings.app_name, settings.node_env)

    # 1. Tablas propias (best-effort: si la BD cae, el servicio
    #    levanta igual — el monitoreo ya es best-effort por diseño)
    try:
        await init_db()
    except Exception as e:
        logger.error(
            "No se pudo inicializar la BD (el monitoreo quedará "
            "degradado hasta reconectar): %s",
            e,
        )

    # 2. Warmup de YOLO (best-effort: sin internet falla la
    #    descarga inicial, pero se reintentará en la 1ª petición)
    try:
        await deteccion_service.precargar()
    except Exception as e:
        logger.warning(
            "Warmup de YOLO falló — se reintentará en la primera "
            "petición: %s",
            e,
        )

    logger.info("=== Servicio IA listo para peticiones ===")

    yield  # ← aquí la app atiende peticiones

    # ═══════════════ SHUTDOWN ═══════════════
    logger.info("=== Cerrando servicio IA ===")
    await cerrar_conexiones()
    logger.info("=== Servicio IA detenido correctamente ===")


app = FastAPI(
    title=settings.app_name,
    description=(
        "Servicio especializado de visión artificial e inteligencia "
        "artificial de VisionCare AI: detección de objetos (YOLOv8), "
        "y análisis de imágenes para la autonomía de personas con "
        "discapacidad visual.\n\n"
        "Consumido por el backend NestJS (server-to-server) con el "
        "token JWT del usuario final reenviado."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS (para probar el Swagger directamente desde el navegador) ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Manejadores globales de error (formato estándar) ───
registrar_manejadores(app)

# ─── Routers (bajo el prefijo global /api/v1) ───
from app.api.v1.deteccion_router import router as deteccion_router
from app.api.v1.salud_router import router as salud_router

app.include_router(salud_router, prefix=settings.api_prefix_path)
app.include_router(deteccion_router, prefix=settings.api_prefix_path)


@app.get("/", include_in_schema=False)
async def raiz() -> dict:
    """Redirección informativa a la documentación."""
    return {
        "servicio": settings.app_name,
        "documentacion": "/docs",
        "health": f"/{settings.api_prefix_path}/health",
    }