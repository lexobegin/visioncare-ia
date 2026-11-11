"""
═══════════════════════════════════════════════════════════
VisionCare AI — Servicio de billetes (CU15/CU16)

Orquesta BilleteEngine (OCR + color) + monitoreo.
Modelo registrado: 'billete-analyzer' v1.0 (CLASIFICACION).

Comparte los SINGLETONS de ocr_service.reader y
color_service.analyzer → los modelos se cargan UNA vez
por proceso (warmups existentes del lifespan).
═══════════════════════════════════════════════════════════
"""

import logging
import time

from fastapi import HTTPException, status

from app.ai_engine.billete_engine import BilleteEngine
from app.schemas.analisis import ResultadoBillete
from app.services.monitoreo_service import monitoreo

MODELO_NOMBRE = "billete-analyzer"
MODELO_VERSION = "1.0"
TIPO_OPERACION = "RECONOCER_BILLETE"


class BilleteService:
    def __init__(self, engine: BilleteEngine | None = None) -> None:
        if engine is not None:
            self.engine = engine
        else:
            # Comparte singletons (memoria única de modelos)
            from app.services.color_service import color_service
            from app.services.ocr_service import ocr_service

            self.engine = BilleteEngine(
                reader_ocr=ocr_service.reader,
                analizador_color=color_service.analyzer,
            )
        self.logger = logging.getLogger("visioncare-ia.billete-service")

    async def ejecutar_analisis(
        self,
        imagen_bytes: bytes,
        id_usuario: int | None = None,
        id_captura: int | None = None,
    ) -> ResultadoBillete:
        inicio = time.perf_counter()

        try:
            resultado = await self.engine.analizar(imagen_bytes)
        except ValueError as e:
            await monitoreo.registrar_error(
                tipo_modelo="CLASIFICACION",
                tipo_operacion=TIPO_OPERACION,
                error=e,
                duracion_ms=self._ms_desde(inicio),
                id_usuario=id_usuario,
                id_captura=id_captura,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
            )
        except Exception as e:
            self.logger.exception("Fallo inesperado del motor de billetes.")
            await monitoreo.registrar_error(
                tipo_modelo="CLASIFICACION",
                tipo_operacion=TIPO_OPERACION,
                error=e,
                duracion_ms=self._ms_desde(inicio),
                id_usuario=id_usuario,
                id_captura=id_captura,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error interno durante el análisis del billete.",
            )

        duracion = self._ms_desde(inicio)

        await monitoreo.registrar_exito(
            modelo={"nombre": MODELO_NOMBRE, "version": MODELO_VERSION},
            tipo_modelo="CLASIFICACION",
            ruta_modelo="easyocr + color-analyzer",
            tipo_operacion=TIPO_OPERACION,
            duracion_ms=duracion,
            confianza_promedio=resultado.confianza,
            id_usuario=id_usuario,
            id_captura=id_captura,
            metadatos={
                "moneda": resultado.moneda,
                "denominacion": resultado.denominacion,
                "autenticidad": resultado.autenticidad_estimacion,
                "observaciones": resultado.observaciones,
            },
        )

        return ResultadoBillete(
            moneda=resultado.moneda,
            denominacion=resultado.denominacion,
            confianza=resultado.confianza,
            autenticidad_estimacion=resultado.autenticidad_estimacion,
            confianza_autenticidad=resultado.confianza_autenticidad,
            observaciones=" ".join(resultado.observaciones) or None,
        )

    @staticmethod
    def _ms_desde(inicio: float) -> int:
        return int((time.perf_counter() - inicio) * 1000)


billete_service = BilleteService()