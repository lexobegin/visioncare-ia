"""
═══════════════════════════════════════════════════════════
VisionCare AI — Servicio de señales (CU12)
Comparte singletons de ocr_service/deteccion_service.
Modelo: 'senal-analyzer' v1.0 (DETECCION).
═══════════════════════════════════════════════════════════
"""

import logging
import time

from fastapi import HTTPException, status

from app.ai_engine.senal_engine import SenalEngine
from app.schemas.analisis import ItemSenal, ResultadoSenales
from app.services.monitoreo_service import monitoreo

MODELO_NOMBRE = "senal-analyzer"
MODELO_VERSION = "1.0"
TIPO_OPERACION = "RECONOCER_SENAL"


class SenalService:
    def __init__(self, engine: SenalEngine | None = None) -> None:
        if engine is not None:
            self.engine = engine
        else:
            from app.services.deteccion_service import deteccion_service
            from app.services.ocr_service import ocr_service

            self.engine = SenalEngine(
                detector=deteccion_service.detector,
                reader_ocr=ocr_service.reader,
            )
        self.logger = logging.getLogger("visioncare-ia.senal-service")

    async def ejecutar_analisis(
        self,
        imagen_bytes: bytes,
        id_usuario: int | None = None,
        id_captura: int | None = None,
    ) -> ResultadoSenales:
        inicio = time.perf_counter()

        try:
            resultado = await self.engine.analizar(imagen_bytes)
        except ValueError as e:
            await monitoreo.registrar_error(
                tipo_modelo="DETECCION",
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
            self.logger.exception("Fallo inesperado del motor de señales.")
            await monitoreo.registrar_error(
                tipo_modelo="DETECCION",
                tipo_operacion=TIPO_OPERACION,
                error=e,
                duracion_ms=self._ms_desde(inicio),
                id_usuario=id_usuario,
                id_captura=id_captura,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error interno durante el reconocimiento de señales.",
            )

        duracion = self._ms_desde(inicio)

        confianza_prom = (
            round(
                sum(s.confianza for s in resultado.senales)
                / len(resultado.senales),
                4,
            )
            if resultado.senales
            else None
        )

        await monitoreo.registrar_exito(
            modelo={"nombre": MODELO_NOMBRE, "version": MODELO_VERSION},
            tipo_modelo="DETECCION",
            ruta_modelo="yolov8n.pt + easyocr",
            tipo_operacion=TIPO_OPERACION,
            duracion_ms=duracion,
            confianza_promedio=confianza_prom,
            id_usuario=id_usuario,
            id_captura=id_captura,
            metadatos={
                "total_senales": len(resultado.senales),
                "tipos": [s.tipo_senal for s in resultado.senales],
            },
        )

        return ResultadoSenales(
            senales=[
                ItemSenal(
                    tipo_senal=s.tipo_senal,
                    nombre_reconocido=s.nombre_reconocido,
                    descripcion=s.descripcion,
                    confianza=s.confianza,
                )
                for s in resultado.senales
            ]
        )

    @staticmethod
    def _ms_desde(inicio: float) -> int:
        return int((time.perf_counter() - inicio) * 1000)


senal_service = SenalService()