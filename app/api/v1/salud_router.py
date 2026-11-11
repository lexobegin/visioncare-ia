"""
═══════════════════════════════════════════════════════════
VisionCare AI — Health-check del servicio IA (Sprint 2)

GET /api/v1/health — inventario completo:
  YOLO (detección) + OCR + análisis S2 sin modelos (color,
  productos, billetes, señales, escena) reutilizando motores.
═══════════════════════════════════════════════════════════
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.config import settings
from app.database import verificar_conexion
from app.services import (
    billete_service,
    color_service,
    deteccion_service,
    escena_service,
    ocr_service,
    producto_service,
    senal_service,
)

router = APIRouter(tags=["Salud"])


@router.get(
    "/health",
    summary="Estado del servicio IA (BD + inventario de motores)",
    description=(
        "Verifica la conectividad con PostgreSQL y el estado de "
        "los motores: YOLO y EasyOCR cargados en memoria (lazy), "
        "analizadores compuestos listos (reutilizan los modelos). "
        "status='degradado' si la BD no responde (la detección "
        "seguiría funcionando, el monitoreo no)."
    ),
)
async def health(db_ok: bool = Depends(verificar_conexion)) -> dict:
    info_yolo = deteccion_service.info_modelo

    return {
        "status": "ok" if db_ok else "degradado",
        "servicio": settings.app_name,
        "entorno": settings.node_env,
        "base_datos": "conectada" if db_ok else "sin conexión",
        "modelos": {
            "yolo": {
                "nombre": info_yolo["nombre"],
                "version": info_yolo["version"],
                "cargado": deteccion_service.detector.cargado,
                "dispositivo": settings.yolo_device,
                "umbral_confianza": settings.yolo_confidence,
            },
            "ocr": {
                "nombre": ocr_service.info_modelo["nombre"],
                "idiomas": ocr_service.info_modelo["idiomas"],
                "cargado": ocr_service.reader.cargado,
            },
        },
        "analizadores_compuestos": {
            "colores": color_service.__class__.__name__,
            "productos": producto_service.info_modelo,
            "billetes": billete_service.__class__.__name__,
            "senales": senal_service.info_modelo,
            "escena": escena_service.info_modelo,
        },
        "autenticacion": {
            "api_key_activa": settings.api_key_activa,
            "jwt_validacion": True,
        },
        "version_api": "0.2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }