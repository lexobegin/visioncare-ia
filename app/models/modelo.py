"""
═══════════════════════════════════════════════════════════
VisionCare AI — Modelo ORM: tabla modelos_ia

Catálogo de modelos de IA desplegados por este servicio.
Al arrancar, el servicio auto-registra el modelo YOLO activo
(nombre, versión, ruta de pesos, precisión estimada).

Restricciones replicadas del SQL:
  - uq_modelo_version: UNIQUE (nombre, version)
  - ck_modelo_estado:  estado IN ('ACTIVO','INACTIVO','PRUEBA')
  - ck_modelo_precision: precision_aproximada 0–1 o NULL
═══════════════════════════════════════════════════════════
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Identity,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ModeloIA(Base):
    __tablename__ = "modelos_ia"

    __table_args__ = (
        UniqueConstraint("nombre", "version", name="uq_modelo_version"),
        CheckConstraint(
            "estado IN ('ACTIVO','INACTIVO','PRUEBA')",
            name="ck_modelo_estado",
        ),
        CheckConstraint(
            "precision_aproximada IS NULL OR precision_aproximada BETWEEN 0 AND 1",
            name="ck_modelo_precision",
        ),
        {
            "comment": (
                "Catálogo de modelos de IA desplegados "
                "mediante el servicio FastAPI."
            )
        },
    )

    id_modelo: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )

    nombre: Mapped[str] = mapped_column(String(150), nullable=False)

    # Tipo: DETECCION | OCR | CLASIFICACION | DESCRIPCION
    tipo_modelo: Mapped[str] = mapped_column(String(50), nullable=False)

    version: Mapped[str] = mapped_column(String(50), nullable=False)

    proveedor: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )

    # Ruta de los pesos (ej: modelos/yolov8n.pt)
    ruta_modelo: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    precision_aproximada: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )

    estado: Mapped[str] = mapped_column(
        String(20), nullable=False, default="ACTIVO", server_default="ACTIVO"
    )

    fecha_publicacion: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),  # replica el trigger actualizar_updated_at()
    )

    # ─── Relaciones ───
    registros: Mapped[list["RegistroIA"]] = relationship(
        back_populates="modelo"
    )

    def __repr__(self) -> str:
        return f"<ModeloIA {self.nombre} v{self.version} ({self.estado})>"