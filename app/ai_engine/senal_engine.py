"""
═══════════════════════════════════════════════════════════
VisionCare AI — Motor de señales y lugares (CU12)

DOBLE VÍA fusionada:

  VÍA A — YOLO (señales FÍSICAS):
    'stop sign'    → senal_stop
    'traffic light'→ semaforo
    Confianza = la de YOLO directamente.

  VÍA B — OCR (señales ESCRITAS):
    detectar_palabra_senal() (T10) sobre el texto del
    encuadre → salida, bano, peligro, prohibicion,
    escaleras, ascensor, cajas, emergencia…
    Confianza = del OCR (peso menor: texto puede ser de
    contexto, no de la señal misma).

  FUSIÓN: si el texto refuerza a una señal YOLO (p. ej.
  "STOP" escrito + stop sign detectada) → un solo ítem
  con +0.05 de confianza (máx. 0.97).
═══════════════════════════════════════════════════════════
"""

import asyncio
import io
import logging
import re
from dataclasses import dataclass, field

from PIL import Image, UnidentifiedImageError

from app.ai_engine.object_detector import DetectorObjetos
from app.ai_engine.ocr_engine import OcrReader
from app.ai_engine.traducciones import detectar_palabra_senal

logger = logging.getLogger("visioncare-ia.senales")

# Etiqueta COCO → tipo_senal normalizado (BD)
SENALES_COCO: dict[str, str] = {
    "stop sign": "senal_stop",
    "traffic light": "semaforo",
}

# Texto YOLO que refuerza cada tipo (para la fusión)
REFUERZO_TEXTO: dict[str, re.Pattern] = {
    "senal_stop": re.compile(r"\bstop\b|\balto\b", re.IGNORECASE),
    "semaforo": re.compile(r"\bsemaforo\b|\btraffic\b", re.IGNORECASE),
}

CONFIANZA_MINIMA_YOLO = 0.40
BONUS_REFUERZO = 0.05


@dataclass
class ItemSenalInterno:
    """Una señal o lugar detectado."""

    tipo_senal: str
    nombre_reconocido: str | None
    descripcion: str | None
    confianza: float
    fuente: str  # 'yolo' | 'ocr' | 'fusion' (para logs/depuración)


@dataclass
class ResultadoSenalesInterno:
    senales: list[ItemSenalInterno] = field(default_factory=list)


class SenalEngine:
    """Detector de señales por composición YOLO + OCR."""

    def __init__(
        self,
        detector: DetectorObjetos | None = None,
        reader_ocr: OcrReader | None = None,
    ) -> None:
        # Singletons se inyectan desde el service (T13 patrón)
        self.detector = detector or DetectorObjetos()
        self.reader_ocr = reader_ocr or OcrReader()

    @property
    def info_modelo(self) -> dict[str, str]:
        return {"nombre": "senal-analyzer", "version": "1.0"}

    # ═══════════════════ ANÁLISIS ═══════════════════

    async def analizar(self, imagen_bytes: bytes) -> ResultadoSenalesInterno:
        return await asyncio.to_thread(self._analizar_sincrono, imagen_bytes)

    def _analizar_sincrono(self, imagen_bytes: bytes) -> ResultadoSenalesInterno:
        # ─── 0. Decodificar (para el OCR del encuadre completo) ───
        try:
            Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise ValueError(
                "El archivo recibido no es una imagen válida "
                "(formatos aceptados: JPEG, PNG, WEBP)."
            )

        senales: list[ItemSenalInterno] = []

        # ─── VÍA A: YOLO (lock interno del detector) ───
        resultado_yolo = self.detector._inferir_sincrono(
            imagen_bytes, CONFIANZA_MINIMA_YOLO
        )

        for det in resultado_yolo.detecciones:
            tipo = SENALES_COCO.get(det.etiqueta.lower())
            if tipo is None:
                continue
            senales.append(
                ItemSenalInterno(
                    tipo_senal=tipo,
                    nombre_reconocido=None,
                    descripcion=(
                        f"{'Señal de stop' if tipo == 'senal_stop' else 'Semáforo'} "
                        "detectada físicamente en la escena."
                    ),
                    confianza=round(det.confianza, 4),
                    fuente="yolo",
                )
            )

        # ─── VÍA B: OCR del encuadre completo (lock interno) ───
        texto_encontrado: str | None = None
        conf_ocr: float | None = None
        try:
            lectura = self.reader_ocr._leer_sincrono(imagen_bytes)
            texto_encontrado = lectura.texto or None
            conf_ocr = lectura.confianza
        except Exception:
            texto_encontrado = None  # OCR es best-effort en señales

        if texto_encontrado:
            resultado_texto = detectar_palabra_senal(texto_encontrado)
            if resultado_texto:
                tipo, descripcion = resultado_texto

                # ─── FUSIÓN: ¿el texto refuerza una señal YOLO? ───
                fusionada = False
                for senal in senales:
                    patron = REFUERZO_TEXTO.get(senal.tipo_senal)
                    if patron and patron.search(texto_encontrado):
                        senal.confianza = round(
                            min(0.97, senal.confianza + BONUS_REFUERZO), 4
                        )
                        senal.fuente = "fusion"
                        if not senal.nombre_reconocido:
                            # primera línea en MAYÚSCULAS como nombre
                            senal.nombre_reconocido = self._linea_mayusculas(
                                texto_encontrado
                            )
                        fusionada = True
                        break

                if not fusionada:
                    conf = (conf_ocr or 0.6) * 0.85  # peso menor del texto
                    senales.append(
                        ItemSenalInterno(
                            tipo_senal=tipo,
                            nombre_reconocido=self._linea_mayusculas(
                                texto_encontrado
                            ),
                            descripcion=descripcion,
                            confianza=round(conf, 4),
                            fuente="ocr",
                        )
                    )

        logger.info(
            "Señales: %d detectada(s) [%s].",
            len(senales),
            ", ".join(s.fuente for s in senales) or "-",
        )

        return ResultadoSenalesInterno(senales=senales)

    @staticmethod
    def _linea_mayusculas(texto: str) -> str | None:
        """Primera línea mayormente mayúscula (típico de letreros)."""
        for linea in texto.split("\n"):
            linea = linea.strip()
            letras = [c for c in linea if c.isalpha()]
            if len(linea) >= 3 and letras and sum(
                1 for c in letras if c.isupper()
            ) / len(letras) >= 0.6:
                return linea[:150]
        return None