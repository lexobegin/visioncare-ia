"""
═══════════════════════════════════════════════════════════
VisionCare AI — Router de detección de objetos (CU08)

POST /api/v1/detecciones/objetos

Contrato EXACTO que envía vision-ia.client.ts (NestJS):
  multipart/form-data:
    - imagen:      archivo (JPEG/PNG/WEBP)     [obligatorio]
    - id_captura:  int opcional (trazabilidad en registros_ia)

Seguridad: doble capa (X-API-Key si configurada + JWT usuario).
═══════════════════════════════════════════════════════════
"""

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from app.config import settings
from app.core.security import seguridad_reconocimiento
from app.schemas.deteccion import RespuestaDeteccion
from app.services import deteccion_service

# Tipos MIME aceptados (idéntico a la validación del backend)
MIME_PERMITIDOS = {"image/jpeg", "image/png", "image/webp"}

router = APIRouter(prefix="/detecciones", tags=["Detecciones"])


@router.post(
    "/objetos",
    response_model=RespuestaDeteccion,
    status_code=status.HTTP_200_OK,
    summary="CU08 — Detectar objetos en una imagen (YOLOv8)",
    description=(
        "Analiza una imagen con el modelo YOLO y devuelve los objetos "
        "detectados con su confianza y posición (bounding box), "
        "ordenados por confianza descendente. Requiere el token JWT "
        "del usuario (reenviado por el backend NestJS)."
    ),
    responses={
        400: {"description": "Archivo faltante, vacío o imagen inválida"},
        401: {"description": "API Key o token JWT ausente/inválido"},
        413: {"description": "La imagen supera el tamaño máximo permitido"},
        500: {"description": "Error interno durante el análisis"},
    },
)
async def detectar_objetos(
    imagen: UploadFile = File(
        ..., description="Imagen a analizar (JPEG, PNG o WEBP)"
    ),
    id_captura: int | None = Form(
        default=None,
        ge=1,
        description=(
            "ID de la captura en el backend NestJS "
            "(trazabilidad en registros_ia para el monitoreo CU40)"
        ),
    ),
    usuario: dict = Depends(seguridad_reconocimiento),
) -> RespuestaDeteccion:
    """
    Endpoint principal de detección.

    Flujo: validación de tipo/tamaño → lectura de bytes →
    deteccion_service (YOLO + monitoreo) → contrato NestJS.
    """

    # ─── 1. Validar tipo MIME ───
    if imagen.content_type not in MIME_PERMITIDOS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Formato no soportado ({imagen.content_type}). "
                "Aceptados: JPEG, PNG, WEBP."
            ),
        )

    # ─── 2. Leer y validar contenido ───
    datos = await imagen.read()

    if not datos:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo está vacío.",
        )

    if len(datos) > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"La imagen supera el tamaño máximo de "
                f"{settings.max_image_size_mb} MB."
            ),
        )

    # ─── 3. Ejecutar detección con monitoreo ───
    # ValueError (imagen inválida) → 400 y otros errores → 500,
    # ya gestionados dentro del service (con registro en registros_ia).
    return await deteccion_service.ejecutar_deteccion(
        imagen_bytes=datos,
        id_usuario=usuario["id_usuario"],
        id_captura=id_captura,
    )