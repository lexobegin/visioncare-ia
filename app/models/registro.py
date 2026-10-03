"""
═══════════════════════════════════════════════════════════
VisionCare AI — Modelo ORM: tabla registros_ia

Registro técnico de CADA ejecución de un modelo:
duración, estado (éxito/error), confianza y metadatos.
Alimenta el monitoreo IA (CU40) desde la web admin.

Nota de diseño: id_usuario e id_captura van SIN ForeignKey
en este ORM a propósito — son tablas gestionadas por el
backend NestJS. Este servicio solo las referencia por id.
═══════════════════════════════════════════════════════════
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Identity,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class RegistroIA(Base):
    __tablename__ = "registros_ia"

    __table_args__ = (
        Index("idx_registros_ia_fecha", "fecha_ejecucion"),
        Index("idx_registros_ia_modelo_fecha", "id_modelo", "fecha_ejecucion"),
        Index("idx_registros_ia_estado", "resultado_estado"),
        CheckConstraint(
            "resultado_estado IN ('EXITO','ERROR','ADVERTENCIA')",
            name="ck_registro_estado",
        ),
        CheckConstraint(
            "duracion_ms IS NULL OR duracion_ms >= 0",
            name="ck_registro_duracion",
        ),
        CheckConstraint(
            "confianza IS NULL OR confianza BETWEEN 0 AND 1",
            name="ck_registro_confianza",
        ),
        {
            "comment": (
                "Registros técnicos de ejecución de los "
                "servicios de inteligencia artificial."
            )
        },
    )

    id_registro: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )

    # FK hacia tabla propia del servicio
    id_modelo: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("modelos_ia.id_modelo", ondelete="SET NULL"),
        nullable=True,
    )

    # Referencias a tablas del backend (sin FK en este ORM — ver docstring)
    id_usuario: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, index=True
    )
    id_captura: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, index=True
    )

    # Ej: DETECCION_OBJETOS | LECTURA_OCR | RECONOCER_BILLETE ...
    tipo_operacion: Mapped[str] = mapped_column(String(50), nullable=False)

    duracion_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    resultado_estado: Mapped[str] = mapped_column(String(20), nullable=False)

    confianza: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )

    mensaje_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    metadatos: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB, nullable=True
    )

    fecha_ejecucion: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # ─── Relaciones ───
    modelo: Mapped[Optional["ModeloIA"]] = relationship(
        back_populates="registros"
    )

    def __repr__(self) -> str:
        return (
            f"<RegistroIA op={self.tipo_operacion} "
            f"estado={self.resultado_estado}>"
        )