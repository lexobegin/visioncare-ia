"""
═══════════════════════════════════════════════════════════
VisionCare AI — Servicio de descripción de escena (CU10)
Comparte el singleton del detector. Modelo: 'escena-describer'.
═══════════════════════════════════════════════════════════
"""

import logging
import time

from fastapi import HTTPException, status

from app.ai_engine.escena_engine import EscenaEngine
from app.schemas.analisis import ResultadoDescripcion
from app.services.monitoreo_service import monitoreo

MODELO_NOMBRE = "escena-describer"
MODELO_VERSION = "1.0"
TIPO_OPERACION = "DESCRIBIR_ESCENA"


class EscenaService:
    def __init__(self, engine: EscenaEngine | None = None) -> None:
        if engine is not None:
            self.engine = engine
        else:
            from app.services.deteccion_service import deteccion_service

            self.engine = EscenaEngine(
                detector=deteccion_service.detector,
            )
        self.logger = logging.getLogger("visioncare-ia.escena-service")

    async def ejecutar_analisis(
        self,
        imagen_bytes: bytes,
        id_usuario: int | None = None,
        id_captura: int | None = None,
    ) -> ResultadoDescripcion:
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
            self.logger.exception("Fallo inesperado del motor de escena.")
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
                detail="Error interno durante la descripción de la escena.",
            )

        duracion = self._ms_desde(inicio)

        await monitoreo.registrar_exito(
            modelo={"nombre": MODELO_NOMBRE, "version": MODELO_VERSION},
            tipo_modelo="DETECCION",
            ruta_modelo="yolov8n.pt",
            tipo_operacion=TIPO_OPERACION,
            duracion_ms=duracion,
            confianza_promedio=resultado.confianza,
            id_usuario=id_usuario,
            id_captura=id_captura,
            metadatos={
                "total_menciones": len(resultado.menciones),
            },
        )

        return ResultadoDescripcion(
            descripcion=resultado.descripcion,
            resumen=resultado.resumen,
            confianza=resultado.confianza,
        )

    @staticmethod
    def _ms_desde(inicio: float) -> int:
        return int((time.perf_counter() - inicio) * 1000)


escena_service = EscenaService()