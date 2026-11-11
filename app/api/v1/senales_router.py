"""
═══════════════════════════════════════════════════════════
VisionCare AI — Router de señales (CU12)
POST /api/v1/senales/reconocer — contrato NestJS (T3 S2).
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
from app.schemas.analisis import ResultadoSenales
from app.services.senal_service import senal_service

MIME_PERMITIDOS = {"image/jpeg", "image/png", "image/webp"}

router = APIRouter(prefix="/senales", tags=["Señales"])


@router.post(
    "/reconocer",
    response_model=ResultadoSenales,
    status_code=status.HTTP_200_OK,
    summary="CU12 — Reconocer señales y lugares (YOLO + OCR de palabras clave)",
    description=(
        "Detecta señales físicas (stop, semáforo) con YOLO y señales "
        "escritas (SALIDA, BAÑOS, PELIGRO…) mediante OCR de palabras "
        "clave. Si el texto refuerza una señal física, se fusionan."
    ),
    responses={
        400: {"description": "Archivo faltante, vacío o imagen inválida"},
        401: {"description": "API Key o token JWT ausente/inválido"},
        413: {"description": "La imagen supera el tamaño máximo permitido"},
    },
)
async def reconocer_senales(
    imagen: UploadFile = File(..., description="Imagen a analizar"),
    id_captura: int | None = Form(
        default=None, ge=1,
        description="ID de captura en NestJS (trazabilidad)",
    ),
    usuario: dict = Depends(seguridad_reconocimiento),
) -> ResultadoSenales:
    if imagen.content_type not in MIME_PERMITIDOS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Formato no soportado ({imagen.content_type}). "
                "Aceptados: JPEG, PNG, WEBP."
            ),
        )

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

    return await senal_service.ejecutar_analisis(
        imagen_bytes=datos,
        id_usuario=usuario["id_usuario"],
        id_captura=id_captura,
    )