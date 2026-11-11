"""
═══════════════════════════════════════════════════════════
VisionCare AI — Motor de productos (CU13)

COMPOSICIÓN de los dos motores existentes:

  1. YOLO (DetectorObjetos) → objetos del encuadre
  2. Filtro a clases "comerciales" (PRODUCTOS_COCO: bottle,
     banana, cup, cell phone…) — máx. 3 por confianza
  3. Por cada bbox: RECORTE de la región → OCR (OcrReader)
     → categorizar_producto(etiqueta, texto) fusiona
     clase COCO + marca/texto del empaque
  4. FALLBACK: sin clase comercial pero OCR encuentra una
     MARCA_CONOCIDA → producto solo-por-texto
  5. Vacío si nada → el backend responde "no identificado"

Notas:
  - Reutiliza los locks de YOLO y OCR (ambos thread-safe)
  - El recorte se codifica a PNG en memoria (rápido y sin
    pérdida para el OCR)
═══════════════════════════════════════════════════════════
"""

import asyncio
import io
import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.ai_engine.object_detector import DetectorObjetos
from app.ai_engine.ocr_engine import OcrReader
from app.ai_engine.traducciones import (
    MARCAS_CONOCIDAS,
    PRODUCTOS_COCO,
    categorizar_producto,
)

logger = logging.getLogger("visioncare-ia.productos")

MAX_PRODUCTOS = 3
CONFIANZA_MINIMA = 0.45


@dataclass
class ItemProductoInterno:
    nombre_producto: str | None
    marca: str | None
    categoria: str | None
    descripcion: str | None
    confianza: float


@dataclass
class ResultadoProductosInterno:
    items: list[ItemProductoInterno] = field(default_factory=list)


class ProductoEngine:
    """Detector de productos por composición YOLO + OCR."""

    def __init__(
        self,
        detector: DetectorObjetos | None = None,
        reader_ocr: OcrReader | None = None,
    ) -> None:
        self.detector = detector or DetectorObjetos()
        self.reader_ocr = reader_ocr or OcrReader()

    @property
    def info_modelo(self) -> dict[str, str]:
        return {"nombre": "producto-analyzer", "version": "1.0"}

    # ═══════════════════ ANÁLISIS ═══════════════════

    async def analizar(self, imagen_bytes: bytes) -> ResultadoProductosInterno:
        return await asyncio.to_thread(self._analizar_sincrono, imagen_bytes)

    def _analizar_sincrono(self, imagen_bytes: bytes) -> ResultadoProductosInterno:
        # ─── 0. Decodificar una sola vez (para recortes) ───
        try:
            imagen = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise ValueError(
                "El archivo recibido no es una imagen válida "
                "(formatos aceptados: JPEG, PNG, WEBP)."
            )

        items: list[ItemProductoInterno] = []

        # ─── 1. YOLO (usa el lock interno del detector) ───
        resultado_yolo = None

        # _inferir_sincrono es privado del detector; usamos su API async
        # desde el hilo worker con el lock ya gestionado internamente:
        resultado_yolo = self.detector._inferir_sincrono(
            imagen_bytes, CONFIANZA_MINIMA
        )

        # ─── 2. Filtrar clases comerciales (por confianza DESC) ───
        comerciales = [
            d
            for d in resultado_yolo.detecciones
            if d.etiqueta.lower() in PRODUCTOS_COCO
        ]
        comerciales.sort(key=lambda d: d.confianza, reverse=True)

        # ─── 3. OCR del empaque por cada bbox (máx. 3) ───
        for det in comerciales[:MAX_PRODUCTOS]:
            x_min = int(max(0, det.x_min))
            y_min = int(max(0, det.y_min))
            x_max = int(min(imagen.width, det.x_max))
            y_max = int(min(imagen.height, det.y_max))
            if x_max <= x_min or y_max <= y_min:
                continue

            recorte = imagen.crop((x_min, y_min, x_max, y_max))

            # OCR del recorte (lock interno del reader)
            buffer = io.BytesIO()
            recorte.save(buffer, format="PNG")
            texto_ocr: str | None = None
            try:
                lectura = self.reader_ocr._leer_sincrono(buffer.getvalue())
                texto_ocr = lectura.texto or None
            except Exception:
                texto_ocr = None  # OCR del recorte es best-effort

            nombre, marca, categoria = categorizar_producto(
                det.etiqueta, texto_ocr
            )

            descripcion = f"Objeto '{det.etiqueta}' detectado"
            if texto_ocr:
                descripcion += " con texto legible en el empaque"
            descripcion += "."

            # Confianza combinada: detección + bonus de texto
            confianza = det.confianza
            if texto_ocr and (nombre or marca):
                confianza = min(0.97, det.confianza + 0.10)

            items.append(
                ItemProductoInterno(
                    nombre_producto=nombre,
                    marca=marca,
                    categoria=categoria,
                    descripcion=descripcion,
                    confianza=round(confianza, 4),
                )
            )

        # ─── 4. Fallback: solo OCR con marca conocida ───
        if not items:
            try:
                lectura = self.reader_ocr._leer_sincrono(imagen_bytes)
                texto = lectura.texto.lower()
            except Exception:
                texto = ""

            marca_encontrada = next(
                (m for m in MARCAS_CONOCIDAS if m in texto), None
            )
            if marca_encontrada:
                nombre, marca, categoria = categorizar_producto(
                    None, lectura.texto
                )
                items.append(
                    ItemProductoInterno(
                        nombre_producto=nombre,
                        marca=marca,
                        categoria=categoria or "otro",
                        descripcion="Identificado por el texto del empaque.",
                        confianza=round(lectura.confianza or 0.6, 4),
                    )
                )

        logger.info("Productos: %d identificado(s).", len(items))

        return ResultadoProductosInterno(items=items)