"""
═══════════════════════════════════════════════════════════
VisionCare AI — Servicio de productos (CU13)

Orquesta ProductoEngine (YOLO+OCR composición) + monitoreo.
Modelo registrado: 'producto-analyzer' v1.0 (CLASIFICACION).
═══════════════════════════════════════════════════════════
"""

import logging
import time

from fastapi import HTTPException, status

from app.ai_engine.producto_engine import ProductoEngine
from app.schemas.analisis import ItemProducto, ResultadoProductos
from app.services.monitoreo_service import monitoreo

MODELO_NOMBRE = "producto-analyzer"
MODELO_VERSION = "1.0"
TIPO_OPERACION = "RECONOCER_PRODUCTO"


class ProductoService:
    def __init__(
        self,
        engine: ProductoEngine | None = None,
        *,
        detector=None,
        reader_ocr=None,
    ) -> None:
        if engine is not None:
            self.engine = engine
        else:
            # Comparte las instancias de los singletons de servicio
            # (los modelos se cargan UNA vez por proceso, en los
            # warmups de deteccion_service y ocr_service)
            from app.services.deteccion_service import deteccion_service
            from app.services.ocr_service import ocr_service

            self.engine = ProductoEngine(
                detector=detector or deteccion_service.detector,
                reader_ocr=reader_ocr or ocr_service.reader,
            )
        self.logger = logging.getLogger("visioncare-ia.producto-service")

    @property
    def info_modelo(self) -> dict[str, str]:
        return self.engine.info_modelo

    async def ejecutar_analisis(
        self,
        imagen_bytes: bytes,
        id_usuario: int | None = None,
        id_captura: int | None = None,
    ) -> ResultadoProductos:
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
            self.logger.exception("Fallo inesperado del motor de productos.")
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
                detail="Error interno durante el reconocimiento de productos.",
            )

        duracion = self._ms_desde(inicio)

        confianza_prom = (
            round(
                sum(i.confianza for i in resultado.items) / len(resultado.items),
                4,
            )
            if resultado.items
            else None
        )

        await monitoreo.registrar_exito(
            modelo={"nombre": MODELO_NOMBRE, "version": MODELO_VERSION},
            tipo_modelo="CLASIFICACION",
            ruta_modelo="yolov8n.pt + easyocr",
            tipo_operacion=TIPO_OPERACION,
            duracion_ms=duracion,
            confianza_promedio=confianza_prom,
            id_usuario=id_usuario,
            id_captura=id_captura,
            metadatos={
                "total_productos": len(resultado.items),
                "marcas": [i.marca for i in resultado.items if i.marca],
            },
        )

        return ResultadoProductos(
            productos=[
                ItemProducto(
                    nombre_producto=i.nombre_producto,
                    marca=i.marca,
                    categoria=i.categoria,
                    descripcion=i.descripcion,
                    confianza=i.confianza,
                )
                for i in resultado.items
            ]
        )

    @staticmethod
    def _ms_desde(inicio: float) -> int:
        return int((time.perf_counter() - inicio) * 1000)


producto_service = ProductoService()