"""
═══════════════════════════════════════════════════════════
VisionCare AI — Health-check del servicio IA (Sprint 2)

GET /api/v1/health — inventario completo:
  YOLO (detección) + OCR + análisis S2 sin modelos (color,
  productos, billetes, señales, escena) reutilizando motores.

Sprint 3 (verificación): esta es la ruta REAL que consume el
backend NestJS para el health-check del servicio IA
("Pendientes vivos" del Sprint 2). RUTA EXACTA:
  GET {IA_URL}/api/v1/health   (pública, sin auth)
Interpretación para NestJS: 'ok' = todo bien; 'degradado' =
BD sin conexión pero el servicio IGUAL analiza imágenes
(tratar 'degradado' como UP para la detección, no como caída).

🛠️ FIX Sprint 3: el inventario usaba `info_modelo` en los 5
analizadores compuestos, pero SOLO producto_service define esa
property (bug latente del Sprint 2 que rompía /health con
AttributeError: 'SenalService' object has no attribute
'info_modelo'). Ahora la metadata se lee de las constantes de
módulo de cada service (MODELO_NOMBRE/VERSION/TIPO_OPERACION)
con getattr — /health es el liveness probe y NUNCA debe 500.

🆕 Sprint 3: agrega tiempo_activo_segundos (uptime del
proceso, útil para el panel de monitoreo).
═══════════════════════════════════════════════════════════
"""

import sys
import time
from datetime import datetime, timezone
from typing import Any

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

# Arranque del proceso (uptime del health-check)
_inicio_servicio = time.monotonic()


def _info_analizador(servicio: Any) -> dict[str, str | None]:
    """
    Metadata mínima de un analizador compuesto, SIN asumir que
    expone `info_modelo` (solo producto_service la tiene; el
    resto declara MODELO_NOMBRE / MODELO_VERSION / TIPO_OPERACION
    como constantes de su módulo).

    Si nada está disponible, igual devuelve el nombre de la
    clase: /health es el liveness probe y jamás debe fallar.
    """
    modulo = sys.modules.get(type(servicio).__module__)
    return {
        "clase": type(servicio).__name__,
        "modelo": getattr(modulo, "MODELO_NOMBRE", None) if modulo else None,
        "version": getattr(modulo, "MODELO_VERSION", None) if modulo else None,
        "tipo_operacion": (
            getattr(modulo, "TIPO_OPERACION", None) if modulo else None
        ),
    }


@router.get(
    "/health",
    summary="Estado del servicio IA (BD + inventario de motores)",
    description=(
        "Verifica la conectividad con PostgreSQL y el estado de "
        "los motores: YOLO y EasyOCR cargados en memoria (lazy), "
        "analizadores compuestos listos (reutilizan los modelos). "
        "status='degradado' si la BD no responde (la detección "
        "seguiría funcionando, el monitoreo no). Esta es la ruta "
        "que consulta el backend NestJS para saber si la IA viva."
    ),
)
async def health(db_ok: bool = Depends(verificar_conexion)) -> dict:
    info_yolo = deteccion_service.info_modelo

    return {
        "status": "ok" if db_ok else "degradado",
        "servicio": settings.app_name,
        "entorno": settings.node_env,
        "tiempo_activo_segundos": int(time.monotonic() - _inicio_servicio),
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
            "colores": _info_analizador(color_service),
            "productos": _info_analizador(producto_service),
            "billetes": _info_analizador(billete_service),
            "senales": _info_analizador(senal_service),
            "escena": _info_analizador(escena_service),
        },
        "autenticacion": {
            "api_key_activa": settings.api_key_activa,
            "jwt_validacion": True,
        },
        "version_api": "0.2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }