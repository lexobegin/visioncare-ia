"""
═══════════════════════════════════════════════════════════
VisionCare AI — Servicio de colores (CU14)

Motor numérico (sin modelos) + monitoreo best-effort.
Modelo registrado: 'color-analyzer' v1.0 (CLASIFICACION).
═══════════════════════════════════════════════════════════
"""

import logging
import time

from fastapi import HTTPException, status

from app.ai_engine.color_engine import ColorAnalyzer
from app.schemas.analisis import ResultadoColores
from app.services.monitoreo_service import monitoreo

MODELO_NOMBRE = "color-analyzer"
MODELO_VERSION = "1.0"
TIPO_OPERACION = "RECONOCER_COLOR"


class ColorService:
    def __init__(self, analyzer: ColorAnalyzer | None = None) -> None:
        self.analyzer = analyzer or ColorAnalyzer()
        self.logger = logging.getLogger("visioncare-ia.color-service")

    async def ejecutar_analisis(
        self,
        imagen_bytes: bytes,
        id_usuario: int | None = None,
        id_captura: int | None = None,
    ) -> ResultadoColores:
        inicio = time.perf_counter()

        try:
            resultado = await self.analyzer.analizar(imagen_bytes)
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
            self.logger.exception("Fallo inesperado del motor de colores.")
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
                detail="Error interno durante el análisis de color.",
            )

        duracion = self._ms_desde(inicio)

        await monitoreo.registrar_exito(
            modelo={"nombre": MODELO_NOMBRE, "version": MODELO_VERSION},
            tipo_modelo="CLASIFICACION",
            ruta_modelo=None,
            tipo_operacion=TIPO_OPERACION,
            duracion_ms=duracion,
            confianza_promedio=resultado.confianza,
            id_usuario=id_usuario,
            id_captura=id_captura,
            metadatos={
                "color_principal": resultado.color_principal,
                "secundarios": resultado.colores_secundarios,
            },
        )

        return ResultadoColores(
            color_principal=resultado.color_principal,
            colores_secundarios=resultado.colores_secundarios,
            confianza=resultado.confianza,
        )

    @staticmethod
    def _ms_desde(inicio: float) -> int:
        return int((time.perf_counter() - inicio) * 1000)


color_service = ColorService()