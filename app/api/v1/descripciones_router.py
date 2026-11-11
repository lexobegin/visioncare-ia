"""
═══════════════════════════════════════════════════════════
VisionCare AI — Router de descripción de escena (CU10)
POST /api/v1/descripciones/generar — contrato NestJS (T3 S2).
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
from app.schemas.analisis import ResultadoDescripcion
from app.services.escena_service import escena_service

MIME_PERMITIDOS = {"image/jpeg", "image/png", "image/webp"}

router = APIRouter(prefix="/descripciones", tags=["Descripciones"])


@router.post(
    "/generar",
    response_model=ResultadoDescripcion,
    status_code=status.HTTP_200_OK,
    summary="CU10 — Describir la escena (composición espacial desde YOLO)",
    description=(
        "Genera una descripción de la escena con la posición de "
        "cada objeto (al frente / izquierda / derecha), priorizando "
        "lo relevante. Devuelve también un resumen corto listo para "
        "TTS (usado por el Modo Continuo)."
    ),
    responses={
        400: {"description": "Archivo faltante, vacío o imagen inválida"},
        401: {"description": "API Key o token JWT ausente/inválido"},
        413: {"description": "La imagen supera el tamaño máximo permitido"},
    },
)
async def generar_descripcion(
    imagen: UploadFile = File(..., description="Imagen de la escena"),
    id_captura: int | None = Form(
        default=None, ge=1,
        description="ID de captura en NestJS (trazabilidad)",
    ),
    usuario: dict = Depends(seguridad_reconocimiento),
) -> ResultadoDescripcion:
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

    return await escena_service.ejecutar_analisis(
        imagen_bytes=datos,
        id_usuario=usuario["id_usuario"],
        id_captura=id_captura,
    )