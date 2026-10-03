"""
═══════════════════════════════════════════════════════════
VisionCare AI — Esquemas de detección de objetos (CU08)

Contrato de salida = EXACTAMENTE lo que consume
vision-ia-client.ts del backend NestJS (interface
RespuestaDeteccionIA):

{
  "modelo": { "nombre": "yolov8n", "version": "1.0" },
  "duracion_ms": 230,
  "detecciones": [
    { "etiqueta": "person", "categoria": "persona",
      "confianza": 0.94, "x_min": 10.5, "y_min": 20.1,
      "x_max": 150.2, "y_max": 300.9 }
  ]
}
═══════════════════════════════════════════════════════════
"""

from pydantic import BaseModel, Field


class ModeloInfo(BaseModel):
    """Identificación del modelo que ejecutó el análisis."""

    nombre: str = Field(
        example="yolov8n", description="Nombre corto del modelo"
    )
    version: str = Field(
        example="1.0", description="Versión de nuestro despliegue"
    )


class DeteccionItem(BaseModel):
    """Un objeto detectado en la imagen."""

    etiqueta: str = Field(
        example="person",
        description="Etiqueta original del modelo (inglés, dataset COCO)",
    )
    categoria: str | None = Field(
        example="persona",
        description="Traducción al español (None si no está en el diccionario)",
    )
    confianza: float = Field(
        ge=0, le=1, example=0.94,
        description="Confianza de la detección (0–1)",
    )
    x_min: float = Field(ge=0, example=10.5, description="Bounding box — esquina sup. izq. X")
    y_min: float = Field(ge=0, example=20.1, description="Bounding box — esquina sup. izq. Y")
    x_max: float = Field(ge=0, example=150.2, description="Bounding box — esquina inf. der. X")
    y_max: float = Field(ge=0, example=300.9, description="Bounding box — esquina inf. der. Y")


class RespuestaDeteccion(BaseModel):
    """Respuesta completa del endpoint POST /detecciones/objetos."""

    modelo: ModeloInfo = Field(description="Modelo que ejecutó el análisis")
    duracion_ms: int = Field(
        ge=0, example=230,
        description="Duración TOTAL del análisis (inferencia + overhead)",
    )
    detecciones: list[DeteccionItem] = Field(
        default_factory=list,
        description="Objetos detectados, ordenados por confianza DESC",
    )