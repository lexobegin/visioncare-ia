"""
═══════════════════════════════════════════════════════════
VisionCare AI — Servicio de lectura OCR (CU11)

Orquesta: motor EasyOCR → heurísticas (idioma, tipo de
documento) → monitoreo (best-effort) → contrato ResultadoOcr.

Excepciones:
  - ValueError (imagen inválida)   → HTTPException 400
  - Otros errores del motor        → HTTPException 500
Ambos intentan registrar el error en registros_ia.
═══════════════════════════════════════════════════════════
"""

import logging
import re
import time

from fastapi import HTTPException, status

from app.ai_engine.ocr_engine import OcrReader
from app.config import settings
from app.schemas.analisis import ResultadoOcr
from app.services.monitoreo_service import monitoreo

# Identificación del modelo para modelos_ia
MODELO_NOMBRE = "easyocr"
MODELO_VERSION = "1.0"
TIPO_OPERACION = "LECTURA_OCR"


class OcrService:
    """Orquestador del análisis OCR (inyectable para tests)."""

    def __init__(self, reader: OcrReader | None = None) -> None:
        self.reader = reader or OcrReader()
        self.logger = logging.getLogger("visioncare-ia.ocr-service")

    # ═══════════════ CICLO DE VIDA ═══════════════

    async def precargar(self) -> None:
        """Warmup del reader (se llama desde el lifespan)."""
        await self.reader.precargar()

    @property
    def info_modelo(self) -> dict[str, str]:
        return self.reader.info_modelo

    # ═══════════════ FLUJO PRINCIPAL ═══════════════

    async def ejecutar_ocr(
        self,
        imagen_bytes: bytes,
        id_usuario: int | None = None,
        id_captura: int | None = None,
    ) -> ResultadoOcr:
        """Lee el texto de la imagen y devuelve el contrato del backend."""
        inicio = time.perf_counter()

        # ─── 1. Inferencia ───
        try:
            resultado = await self.reader.leer(imagen_bytes)
        except ValueError as e:
            duracion = self._ms_desde(inicio)
            await monitoreo.registrar_error(
                tipo_modelo="OCR",
                tipo_operacion=TIPO_OPERACION,
                error=e,
                duracion_ms=duracion,
                id_usuario=id_usuario,
                id_captura=id_captura,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as e:
            duracion = self._ms_desde(inicio)
            self.logger.exception("Fallo inesperado del motor OCR.")
            await monitoreo.registrar_error(
                tipo_modelo="OCR",
                tipo_operacion=TIPO_OPERACION,
                error=e,
                duracion_ms=duracion,
                id_usuario=id_usuario,
                id_captura=id_captura,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error interno durante la lectura del texto.",
            )

        # ─── 2. Heurísticas + monitoreo ───
        duracion_total = self._ms_desde(inicio)

        idioma = self._detectar_idioma(resultado.texto)
        tipo_documento = self._clasificar_documento(resultado.texto)

        await monitoreo.registrar_exito(
            modelo={"nombre": MODELO_NOMBRE, "version": MODELO_VERSION},
            tipo_modelo="OCR",
            ruta_modelo=f"easyocr:{','.join(settings.ocr_idiomas_list)}",
            tipo_operacion=TIPO_OPERACION,
            duracion_ms=duracion_total,
            confianza_promedio=resultado.confianza,
            id_usuario=id_usuario,
            id_captura=id_captura,
            metadatos={
                "total_lineas": len(resultado.lineas),
                "total_caracteres": len(resultado.texto),
                "idiomas": settings.ocr_idiomas_list,
                "tamano_bytes": len(imagen_bytes),
            },
        )

        # ─── 3. Contrato (espejo ResultadoOcrDto del backend) ───
        return ResultadoOcr(
            texto_extraido=resultado.texto,
            idioma=idioma,
            confianza=resultado.confianza,
            tipo_documento=tipo_documento,
        )

    # ═══════════════ HEURÍSTICAS ═══════════════

    @staticmethod
    def _detectar_idioma(texto: str) -> str | None:
        """
        Detección ligera: EasyOCR lee es+en sin distinguir.
        Marcadores españoles (áéíóúñ¿¡) → 'es'; palabras inglesas
        frecuentes → 'en'; sin señal clara → None.
        """
        if not texto:
            return None

        texto_lower = texto.lower()

        if re.search(r"[áéíóúñ¿¡]", texto_lower):
            return "es"

        marcadores_en = re.search(
            r"\b(the|and|exit|open|push|out|stop|welcome)\b", texto_lower
        )
        if marcadores_en:
            return "en"

        # Sin marcadores: el español es el idioma principal del contexto
        return "es" if settings.ocr_idiomas_list[0] == "es" else None

    @staticmethod
    def _clasificar_documento(texto: str) -> str:
        """
        Heurística de clasificación (documentada — ampliable):
          precio    → patrones de moneda (Bs, $, USD, €) o total
          menu      → palabras 'menú'/'carta'
          etiqueta  → unidades de producto (g, kg, ml, L)
          documento → denso (8+ líneas, 300+ caracteres)
          cartel    → pocas líneas cortas (típico de letreros)
          otro      → default
        """
        texto_lower = texto.lower()
        lineas = [l for l in texto.split("\n") if l.strip()]

        # Precio: moneda + dígitos (en cualquier orden)
        if re.search(
            r"(b[sz]\.?|\$|usd|€)\s*\d|\d+[.,]\d{2}\s*(b[sz]|\$)|\btotal\b",
            texto_lower,
        ):
            return "precio"

        # Menú / carta
        if re.search(r"\bmenu\b|\bmen[uú]\b|\bcarta\b", texto_lower):
            return "menu"

        # Etiqueta de producto: unidades de peso/volumen
        if re.search(r"\b\d+\s*(g|gr|gramos|kg|ml|l|lt|litros)\b", texto_lower):
            return "etiqueta"

        # Documento denso
        if len(lineas) >= 8 and len(texto) > 300:
            return "documento"

        # Cartel: poco texto concentrado
        if len(lineas) <= 3 and len(texto) <= 60:
            return "cartel"

        return "otro"

    # ═══════════════ HELPERS ═══════════════

    @staticmethod
    def _ms_desde(inicio: float) -> int:
        return int((time.perf_counter() - inicio) * 1000)


# ═══════════════════ SINGLETON ═══════════════════
ocr_service = OcrService()