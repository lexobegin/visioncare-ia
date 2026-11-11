"""
═══════════════════════════════════════════════════════════
VisionCare AI — Monitoreo GENÉRICO (modelos_ia + registros_ia)

Refactor del patrón del Sprint 1 (deteccion_service) en un
servicio reutilizable: los 6 motores del Sprint 2 registran
sus ejecuciones con su propio tipo_operacion:

    MONITOREO.registrar_exito(
        resultado_motor=..., tipo_operacion="LECTURA_OCR", ...
    )

Principios (idénticos a S1):
  - Auto-registro del modelo activo (get-or-create)
  - BEST-EFFORT: si la BD falla, el análisis se entrega igual
  - Errores también se registran (sin id_modelo si falló antes)
═══════════════════════════════════════════════════════════
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.config import settings
from app.database import session_factory
from app.models import ModeloIA, RegistroIA

logger = logging.getLogger("visioncare-ia.monitoreo")


class MonitoreoService:
    """Registro de ejecuciones IA (inyectable para tests)."""

    # ═══════════════ MODELO ═══════════════

    async def obtener_o_registrar_modelo(
        self,
        session,
        *,
        nombre: str,
        version: str,
        tipo_modelo: str,
        ruta: str | None,
        proveedor: str = "VisionCare",
    ) -> ModeloIA:
        """Get-or-create por nombre+versión (reactiva si no activo)."""
        stmt = select(ModeloIA).where(
            ModeloIA.nombre == nombre,
            ModeloIA.version == version,
        )
        modelo = (await session.execute(stmt)).scalar_one_or_none()

        if modelo is None:
            modelo = ModeloIA(
                nombre=nombre,
                tipo_modelo=tipo_modelo,
                version=version,
                proveedor=proveedor,
                ruta_modelo=ruta,
                estado="ACTIVO",
                fecha_publicacion=datetime.now(timezone.utc),
            )
            session.add(modelo)
            await session.flush()
            logger.info(
                "Modelo auto-registrado: %s v%s (id=%s)",
                nombre, version, modelo.id_modelo,
            )
        elif modelo.estado != "ACTIVO":
            modelo.estado = "ACTIVO"
            await session.flush()
            logger.info("Modelo %s v%s reactivado.", nombre, version)

        return modelo

    # ═══════════════ EJECUCIONES ═══════════════

    async def registrar_exito(
        self,
        *,
        modelo: dict[str, str],      # {nombre, version}
        tipo_modelo: str,
        ruta_modelo: str | None,
        tipo_operacion: str,
        duracion_ms: int,
        confianza_promedio: float | None,
        id_usuario: int | None = None,
        id_captura: int | None = None,
        metadatos: dict[str, Any] | None = None,
    ) -> None:
        """Persiste una ejecución EXITOSA (best-effort)."""
        try:
            async with session_factory() as session:
                modelo_orm = await self.obtener_o_registrar_modelo(
                    session,
                    nombre=modelo["nombre"],
                    version=modelo["version"],
                    tipo_modelo=tipo_modelo,
                    ruta=ruta_modelo,
                )

                registro = RegistroIA(
                    id_modelo=modelo_orm.id_modelo,
                    id_usuario=id_usuario,
                    id_captura=id_captura,
                    tipo_operacion=tipo_operacion,
                    duracion_ms=duracion_ms,
                    resultado_estado="EXITO",
                    confianza=confianza_promedio,
                    metadatos=metadatos or {},
                )
                session.add(registro)
                await session.commit()

                logger.info(
                    "Monitoreo EXITO: %s (%s, %d ms).",
                    tipo_operacion, modelo["nombre"], duracion_ms,
                )
        except Exception as e:
            logger.warning(
                "No se pudo registrar la ejecución (best-effort): %s", e
            )

    async def registrar_error(
        self,
        *,
        tipo_modelo: str,
        tipo_operacion: str,
        error: Exception,
        duracion_ms: int,
        id_usuario: int | None = None,
        id_captura: int | None = None,
    ) -> None:
        """Persiste una ejecución FALLIDA (best-effort)."""
        try:
            async with session_factory() as session:
                registro = RegistroIA(
                    id_modelo=None,  # pudo fallar antes de cargar
                    id_usuario=id_usuario,
                    id_captura=id_captura,
                    tipo_operacion=tipo_operacion,
                    duracion_ms=duracion_ms,
                    resultado_estado="ERROR",
                    mensaje_error=str(error)[:1000],
                    metadatos={"tipo_error": error.__class__.__name__},
                )
                session.add(registro)
                await session.commit()
        except Exception as e:
            logger.warning(
                "No se pudo registrar el ERROR (best-effort): %s", e
            )


# Instancia única (los services la importan)
monitoreo = MonitoreoService()