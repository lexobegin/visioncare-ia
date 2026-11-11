"""
═══════════════════════════════════════════════════════════
VisionCare AI — Router OCR (CU11)

POST /api/v1/ocr/leer

Contrato EXACTO que envía vision-ia.client.ts (NestJS):
  multipart/form-data:
    - imagen:     archivo (JPEG/PNG/WEBP)  [obligatorio]
    - id_captura: int opcional (trazabilidad en registros_ia)

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
from app.schemas.analisis import ResultadoOcr
from app.services.ocr_service import ocr_service

MIME_PERMITIDOS = {"image/jpeg", "image/png", "image/webp"}

router = APIRouter(prefix="/ocr", tags=["OCR"])


@router.post(
    "/leer",
    response_model=ResultadoOcr,
    status_code=status.HTTP_200_OK,
    summary="CU11 — Leer textos de una imagen (EasyOCR)",
    description=(
        "Extrae el texto de la imagen con su orden de lectura "
        "(arriba→abajo) y una heurística de tipo de documento "
        "(cartel, etiqueta, precio, menú…). Requiere el token JWT "
        "del usuario (reenviado por el backend NestJS)."
    ),
    responses={
        400: {"description": "Archivo faltante, vacío o imagen inválida"},
        401: {"description": "API Key o token JWT ausente/inválido"},
        413: {"description": "La imagen supera el tamaño máximo permitido"},
        500: {"description": "Error interno durante la lectura"},
    },
)
async def leer_texto(
    imagen: UploadFile = File(
        ..., description="Imagen con texto a leer (JPEG, PNG o WEBP)"
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
) -> ResultadoOcr:
    """Endpoint principal de lectura OCR."""

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

    # ─── 3. Ejecutar OCR con monitoreo ───
    return await ocr_service.ejecutar_ocr(
        imagen_bytes=datos,
        id_usuario=usuario["id_usuario"],
        id_captura=id_captura,
    )