"""
═══════════════════════════════════════════════════════════
VisionCare AI — Esquemas Pydantic SPRINT 2 (6 análisis nuevos)

Contratos = ESPEJO EXACTO de los DTOs del backend NestJS
(Tanda 3 del S2: resultados-analisis.dto.ts). Si cambian aquí,
deben cambiar allí — TypeScript y Pydantic validan el mismo
contrato por ambos lados.

Endpoints que los usarán (T11–15):
  POST /ocr/leer            → ResultadoOcr
  POST /senales/reconocer   → ResultadoSenales
  POST /productos/reconocer → ResultadoProductos
  POST /colores/reconocer   → ResultadoColores
  POST /billetes/reconocer  → ResultadoBillete
  POST /descripciones/generar → ResultadoDescripcion
═══════════════════════════════════════════════════════════
"""

from pydantic import BaseModel, Field


# ═══════════════ OCR (CU11) ═══════════════

class ResultadoOcr(BaseModel):
    texto_extraido: str = Field(
        example="SALIDA DE EMERGENCIA\nEmpuje la barra",
        description="Texto reconocido completo (puede ser multilínea)",
    )
    idioma: str | None = Field(default=None, example="es")
    confianza: float | None = Field(
        default=None, ge=0, le=1, example=0.92
    )
    tipo_documento: str | None = Field(
        default=None,
        example="cartel",
        description=(
            "Heurística: cartel, etiqueta, precio, menu, documento, otro"
        ),
    )


# ═══════════════ SEÑALES (CU12) ═══════════════

class ItemSenal(BaseModel):
    tipo_senal: str = Field(
        example="salida",
        description=(
            "Normalizado: senal_stop, semaforo, salida, bano, "
            "escaleras, peligro, cruce…"
        ),
    )
    nombre_reconocido: str | None = Field(
        default=None, example="SALIDA DE EMERGENCIA"
    )
    descripcion: str | None = Field(
        default=None, example="Salida de emergencia o de recinto"
    )
    confianza: float | None = Field(default=None, ge=0, le=1, example=0.88)


class ResultadoSenales(BaseModel):
    senales: list[ItemSenal] = Field(default_factory=list)


# ═══════════════ PRODUCTOS (CU13) ═══════════════

class ItemProducto(BaseModel):
    nombre_producto: str | None = Field(
        default=None, example="Coca-Cola 500ml"
    )
    marca: str | None = Field(default=None, example="Coca-Cola")
    categoria: str | None = Field(
        default=None,
        example="bebida",
        description="bebida, fruta, snack, limpieza, higiene, otro",
    )
    descripcion: str | None = Field(default=None)
    confianza: float | None = Field(default=None, ge=0, le=1, example=0.85)


class ResultadoProductos(BaseModel):
    productos: list[ItemProducto] = Field(default_factory=list)


# ═══════════════ COLORES (CU14) ═══════════════

class ResultadoColores(BaseModel):
    color_principal: str = Field(
        example="azul", description="Color dominante en español"
    )
    colores_secundarios: list[str] | None = Field(
        default=None, example=["blanco", "gris"]
    )
    confianza: float | None = Field(default=None, ge=0, le=1, example=0.9)


# ═══════════════ BILLETES (CU15/16) ═══════════════

class ResultadoBillete(BaseModel):
    moneda: str | None = Field(default=None, example="BOB")
    denominacion: str | None = Field(default=None, example="100")
    confianza: float | None = Field(default=None, ge=0, le=1, example=0.87)
    autenticidad_estimacion: str | None = Field(
        default=None,
        description=(
            "PROBABLEMENTE_AUTENTICO | PROBABLEMENTE_SOSPECHOSO | "
            "INDETERMINADO — estimación VISUAL, no certificación"
        ),
    )
    confianza_autenticidad: float | None = Field(
        default=None, ge=0, le=1, example=0.6
    )
    observaciones: str | None = Field(
        default=None,
        example="Denominación legible; patrones visuales consistentes",
    )


# ═══════════════ DESCRIPCIÓN (CU10) ═══════════════

class ResultadoDescripcion(BaseModel):
    descripcion: str = Field(
        example=(
            "Escena con 2 personas al frente, una a la izquierda; "
            "una señal a la derecha."
        )
    )
    resumen: str = Field(
        example="Dos personas al frente y una señal a la derecha.",
        description="Versión corta lista para TTS",
    )
    confianza: float | None = Field(default=None, ge=0, le=1, example=0.8)