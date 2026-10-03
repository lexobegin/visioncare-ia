"""
═══════════════════════════════════════════════════════════
VisionCare AI — Servicio de detección de objetos (CU08)

Orquesta el flujo completo:

  1. Ejecuta el motor YOLO (DetectorObjetos: lazy + thread-safe)
  2. AUTO-REGISTRO del modelo activo en modelos_ia
     (get-or-create por nombre+versión; reactiva si estaba
     INACTIVO o en PRUEBA)
  3. Persiste CADA ejecución en registros_ia (monitoreo CU40):
     duración, estado, confianza promedio, usuario, metadatos
  4. Devuelve el contrato que espera vision-ia.client.ts

El monitoreo es BEST-EFFORT: si la BD no responde, se
loguea y la respuesta IA igual se entrega.

Excepciones:
  - ValueError (imagen inválida)  → HTTPException 400
  - Cualquier otro error del motor → HTTPException 500
  Ambos casos intentan registrar el error en registros_ia.
═══════════════════════════════════════════════════════════
"""

import logging
import time
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy import select

from app.ai_engine.object_detector import (
    DetectorObjetos,
    ResultadoDeteccion,
)
from app.config import settings
from app.database import session_factory
from app.models import ModeloIA, RegistroIA
from app.schemas.deteccion import (
    DeteccionItem,
    ModeloInfo,
    RespuestaDeteccion,
)

# Código de operación persistido en registros_ia.tipo_operacion
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
        registrando la ejecución en el monitoreo.

        Args:
            imagen_bytes: contenido de la imagen (JPEG/PNG/WEBP).
            id_usuario: del JWT validado (atribución CU40).
            id_captura: si el backend la envía (trazabilidad).
        """
        inicio = time.perf_counter()

        # ─── 1. Inferencia (ValueError = imagen inválida) ───
        try:
            resultado = await self.detector.detectar(imagen_bytes)
        except ValueError as e:
            duracion = self._ms_desde(inicio)
            await self._registrar_error(
                e, duracion, id_usuario, id_captura
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as e:
            duracion = self._ms_desde(inicio)
            self.logger.exception("Fallo inesperado del motor YOLO.")
            await self._registrar_error(
                e, duracion, id_usuario, id_captura
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error interno durante el análisis de la imagen.",
            )

        # ─── 2. Monitoreo (best-effort) ───
        duracion_total = self._ms_desde(inicio)
        await self._registrar_exito(
            resultado, duracion_total, len(imagen_bytes),
            id_usuario, id_captura,
        )

        # ─── 3. Contrato hacia NestJS ───
        return self._construir_respuesta(resultado, duracion_total)

    # ═══════════════ MONITOREO ═══════════════

    async def _obtener_o_registrar_modelo(
        self, session, resultado: ResultadoDeteccion
    ) -> ModeloIA:
        """
        Get-or-create del modelo en modelos_ia (clave: nombre+versión).
        Si existía INACTIVO/PRUEBA, lo reactiva (el servicio lo
        está usando ahora mismo).
        """
        stmt = select(ModeloIA).where(
            ModeloIA.nombre == resultado.modelo_nombre,
            ModeloIA.version == resultado.modelo_version,
        )
        modelo = (await session.execute(stmt)).scalar_one_or_none()

        if modelo is None:
            modelo = ModeloIA(
                nombre=resultado.modelo_nombre,
                tipo_modelo="DETECCION",
                version=resultado.modelo_version,
                proveedor="Ultralytics",
                ruta_modelo=settings.yolo_model,
                estado="ACTIVO",
                fecha_publicacion=datetime.now(timezone.utc),
            )
            session.add(modelo)
            await session.flush()  # obtiene id_modelo
            self.logger.info(
                "Modelo auto-registrado en modelos_ia: %s v%s (id=%s)",
                resultado.modelo_nombre,
                resultado.modelo_version,
                modelo.id_modelo,
            )
        elif modelo.estado != "ACTIVO":
            modelo.estado = "ACTIVO"
            await session.flush()
            self.logger.info(
                "Modelo %s v%s reactivado (estado ACTIVO).",
                resultado.modelo_nombre,
                resultado.modelo_version,
            )

        return modelo

    async def _registrar_exito(
        self,
        resultado: ResultadoDeteccion,
        duracion_ms: int,
        tamano_bytes: int,
        id_usuario: int | None,
        id_captura: int | None,
    ) -> None:
        """Persiste la ejecución EXITOSA (best-effort)."""
        try:
            async with session_factory() as session:
                modelo = await self._obtener_o_registrar_modelo(
                    session, resultado
                )

                # Confianza promedio (None si no hubo detecciones)
                confianza_prom: float | None = None
                if resultado.detecciones:
                    confianza_prom = round(
                        sum(d.confianza for d in resultado.detecciones)
                        / len(resultado.detecciones),
                        4,
                    )

                registro = RegistroIA(
                    id_modelo=modelo.id_modelo,
                    id_usuario=id_usuario,
                    id_captura=id_captura,
                    tipo_operacion=OP_DETECCION_OBJETOS,
                    duracion_ms=duracion_ms,
                    resultado_estado="EXITO",
                    confianza=confianza_prom,
                    metadatos={
                        "total_detecciones": len(resultado.detecciones),
                        "umbral": settings.yolo_confidence,
                        "dispositivo": settings.yolo_device,
                        "tamano_bytes": tamano_bytes,
                    },
                )
                session.add(registro)
                await session.commit()

                self.logger.info(
                    "Monitoreo EXITO registrado (registro=%s, modelo=%s, "
                    "%d detecciones, %d ms).",
                    registro.id_registro,
                    modelo.id_modelo,
                    len(resultado.detecciones),
                    duracion_ms,
                )
        except Exception as e:
            self.logger.warning(
                "No se pudo registrar la ejecución en registros_ia "
                "(best-effort): %s",
                e,
            )

    async def _registrar_error(
        self,
        error: Exception,
        duracion_ms: int,
        id_usuario: int | None,
        id_captura: int | None,
    ) -> None:
        """Persiste la ejecución FALLIDA (best-effort, sin id_modelo)."""
        try:
            async with session_factory() as session:
                registro = RegistroIA(
                    id_modelo=None,  # el error pudo ocurrir antes de cargar
                    id_usuario=id_usuario,
                    id_captura=id_captura,
                    tipo_operacion=OP_DETECCION_OBJETOS,
                    duracion_ms=duracion_ms,
                    resultado_estado="ERROR",
                    mensaje_error=str(error)[:1000],
                    metadatos={
                        "tipo_error": error.__class__.__name__,
                        "dispositivo": settings.yolo_device,
                    },
                )
                session.add(registro)
                await session.commit()
        except Exception as e:
            self.logger.warning(
                "No se pudo registrar el ERROR en registros_ia "
                "(best-effort): %s",
                e,
            )

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
# Instancia única usada por los routers (main.py la precarga
# en el lifespan). Para tests: DeteccionService(detector=fake).
deteccion_service = DeteccionService()