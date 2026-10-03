"""
═══════════════════════════════════════════════════════════
VisionCare AI — Health-check del servicio IA

GET /api/v1/health

Sin autenticación: lo usan el backend NestJS, monitoreo y
balanceadores. Reporta estado de la BD y del modelo YOLO
(cargado en memoria o pendiente de carga perezosa).
═══════════════════════════════════════════════════════════
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.config import settings
from app.database import verificar_conexion
from app.services import deteccion_service

router = APIRouter(tags=["Salud"])


@router.get(
    "/health",
    summary="Estado del servicio IA (BD + modelo YOLO)",
    description=(
        "Verifica la conectividad con PostgreSQL y si el modelo "
        "YOLO está cargado en memoria. status='degradado' si la "
        "BD no responde (la detección seguiría funcionando, "
        "el monitoreo no)."
    ),
)
async def health(db_ok: bool = Depends(verificar_conexion)) -> dict:
    info = deteccion_service.info_modelo

    return {
        "status": "ok" if db_ok else "degradado",
        "servicio": settings.app_name,
        "entorno": settings.node_env,
        "base_datos": "conectada" if db_ok else "sin conexión",
        "modelo": {
            "nombre": info["nombre"],
            "version": info["version"],
            "cargado": deteccion_service.detector.cargado,
            "dispositivo": settings.yolo_device,
            "umbral_confianza": settings.yolo_confidence,
        },
        "autenticacion": {
            "api_key_activa": settings.api_key_activa,
            "jwt_validacion": True,
        },
        "version_api": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }