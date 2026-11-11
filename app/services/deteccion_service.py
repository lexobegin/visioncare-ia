"""
═══════════════════════════════════════════════════════════
VisionCare AI — Servicio de detección de objetos (CU08)

REFACTOR S2: usa el MonitoreoService GENÉRICO (T10) — la
lógica privada de modelos_ia/registros_ia del Sprint 1 se
elimina (estaba duplicada; ahora todos los servicios comparten
el mismo patrón best-effort).

Orquesta: motor YOLO (lazy + thread-safe) → monitoreo →
contrato RespuestaDeteccion (idéntico a S1 — sin cambios de
contrato, solo limpieza interna).

Excepciones:
  - ValueError (imagen inválida)  → HTTPException 400
  - Cualquier otro error del motor → HTTPException 500
  Ambos casos intentan registrar el error en registros_ia.
═══════════════════════════════════════════════════════════
"""

import logging
import time

from fastapi import HTTPException, status

from app.ai_engine.object_detector import (
    DetectorObjetos,
    ResultadoDeteccion,
)
from app.config import settings
from app.schemas.deteccion import (
    DeteccionItem,
    ModeloInfo,
    RespuestaDeteccion,
)
from app.services.monitoreo_service import monitoreo

# Código de operación persistido en registros_ia
OP_DETECCION_OBJETOS = "DETECCION_OBJETOS"


class DeteccionService:
    """Orquestador del análisis de objetos (inyectable para tests)."""

    def __init__(self, detector: DetectorObjetos | None = None) -> None:
        self.detector = detector or DetectorObjetos()
        self.logger = logging.getLogger("visioncare-ia.deteccion")

    # ═══════════════ CICLO DE VIDA ═══════════════

    async def precargar(self) -> None:
        """Warmup del modelo (se llama desde el lifespan de main.py)."""
        await self.detector.precargar()

    @property
    def info_modelo(self) -> dict[str, str]:
        """Info del modelo activo (usado por /health)."""
        return self.detector.info_modelo

    # ═══════════════ FLUJO PRINCIPAL ═══════════════

    async def ejecutar_deteccion(
        self,
        imagen_bytes: bytes,
        id_usuario: int | None = None,
        id_captura: int | None = None,
    ) -> RespuestaDeteccion:
        """
        Analiza una imagen y devuelve el contrato NestJS,
        registrando la ejecución en el monitoreo genérico.
        """
        inicio = time.perf_counter()

        # ─── 1. Inferencia (ValueError = imagen inválida) ───
        try:
            resultado = await self.detector.detectar(imagen_bytes)
        except ValueError as e:
            duracion = self._ms_desde(inicio)
            await monitoreo.registrar_error(
                tipo_modelo="DETECCION",
                tipo_operacion=OP_DETECCION_OBJETOS,
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
            self.logger.exception("Fallo inesperado del motor YOLO.")
            await monitoreo.registrar_error(
                tipo_modelo="DETECCION",
                tipo_operacion=OP_DETECCION_OBJETOS,
                error=e,
                duracion_ms=duracion,
                id_usuario=id_usuario,
                id_captura=id_captura,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error interno durante el análisis de la imagen.",
            )

        # ─── 2. Monitoreo (genérico, best-effort) ───
        duracion_total = self._ms_desde(inicio)

        confianza_prom: float | None = None
        if resultado.detecciones:
            confianza_prom = round(
                sum(d.confianza for d in resultado.detecciones)
                / len(resultado.detecciones),
                4,
            )

        await monitoreo.registrar_exito(
            modelo={
                "nombre": resultado.modelo_nombre,
                "version": resultado.modelo_version,
            },
            tipo_modelo="DETECCION",
            ruta_modelo=settings.yolo_model,
            tipo_operacion=OP_DETECCION_OBJETOS,
            duracion_ms=duracion_total,
            confianza_promedio=confianza_prom,
            id_usuario=id_usuario,
            id_captura=id_captura,
            metadatos={
                "total_detecciones": len(resultado.detecciones),
                "umbral": settings.yolo_confidence,
                "dispositivo": settings.yolo_device,
                "tamano_bytes": len(imagen_bytes),
            },
        )

        # ─── 3. Contrato hacia NestJS (sin cambios de S1) ───
        return self._construir_respuesta(resultado, duracion_total)

    # ═══════════════ HELPERS ═══════════════

    @staticmethod
    def _ms_desde(inicio: float) -> int:
        return int((time.perf_counter() - inicio) * 1000)

    @staticmethod
    def _construir_respuesta(
        resultado: ResultadoDeteccion, duracion_ms: int
    ) -> RespuestaDeteccion:
        return RespuestaDeteccion(
            modelo=ModeloInfo(
                nombre=resultado.modelo_nombre,
                version=resultado.modelo_version,
            ),
            duracion_ms=duracion_ms,
            detecciones=[
                DeteccionItem(
                    etiqueta=d.etiqueta,
                    categoria=d.categoria,
                    confianza=d.confianza,
                    x_min=d.x_min,
                    y_min=d.y_min,
                    x_max=d.x_max,
                    y_max=d.y_max,
                )
                for d in resultado.detecciones
            ],
        )


# ═══════════════════ SINGLETON ═══════════════════
deteccion_service = DeteccionService()