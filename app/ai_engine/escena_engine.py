"""
═══════════════════════════════════════════════════════════
VisionCare AI — Motor de descripción de escena (CU10)

Narración ESPACIAL desde los bboxes de YOLO (sin modelos de
lenguaje — determinista y explicable):

  POSICIÓN (por tercios del centro X):
    < 0.33 → "a tu izquierda" | > 0.66 → "a tu derecha"
    centro → "al frente"

  PRIORIZACIÓN de mención (semáforo del PriorizacionService):
    PELIGRO (persona, vehículos…) primero, luego ATENCION
    (semáforo, señal stop), luego el resto por confianza.

  DESCRIPCIÓN completa: todas las detecciones con posición.
  RESUMEN (TTS): máx. 3 menciones + recuento si hay más.

  CONFIANZA: promedio de las detecciones mencionadas.
═══════════════════════════════════════════════════════════
"""

import asyncio
import logging
from dataclasses import dataclass, field

from app.ai_engine.object_detector import DetectorObjetos
from app.ai_engine.traducciones import traducir_etiqueta

logger = logging.getLogger("visioncare-ia.escena")

CONFIANZA_MINIMA = 0.45
MAX_EN_RESUMEN = 3

# Categorías para priorización de mención (consistente con HU17)
PELIGRO = {"person", "car", "motorcycle", "bicycle", "bus", "truck", "train", "dog"}
ATENCION = {"traffic light", "stop sign", "fire hydrant", "parking meter"}


@dataclass
class MencionEscena:
    """Un objeto con su posición narrable."""

    nombre: str          # en español
    posicion: str        # "al frente" | "a tu izquierda" | "a tu derecha"
    confianza: float
    prioridad: int       # 1=peligro, 2=atención, 3=resto


@dataclass
class ResultadoEscenaInterno:
    descripcion: str
    resumen: str
    confianza: float | None
    menciones: list[MencionEscena] = field(default_factory=list)


class EscenaEngine:
    """Generador de descripciones espaciales (stateless)."""

    def __init__(self, detector: DetectorObjetos | None = None) -> None:
        # Singleton del detector (inyectado desde el service)
        self.detector = detector or DetectorObjetos()

    @property
    def info_modelo(self) -> dict[str, str]:
        return {"nombre": "escena-describer", "version": "1.0"}

    # ═══════════════════ ANÁLISIS ═══════════════════

    async def analizar(self, imagen_bytes: bytes) -> ResultadoEscenaInterno:
        return await asyncio.to_thread(self._analizar_sincrono, imagen_bytes)

    def _analizar_sincrono(self, imagen_bytes: bytes) -> ResultadoEscenaInterno:
        # ─── 1. YOLO (lock interno del detector compartido) ───
        resultado_yolo = self.detector._inferir_sincrono(
            imagen_bytes, CONFIANZA_MINIMA
        )

        if not resultado_yolo.detecciones:
            return ResultadoEscenaInterno(
                descripcion=(
                    "No identifiqué objetos claros en la escena. "
                    "Puede que el entorno esté vacío o la cámara "
                    "esté cubierta."
                ),
                resumen="No identifiqué objetos claros al frente.",
                confianza=None,
            )

        # ─── 2. Posición + prioridad de cada detección ───
        # ancho = 1.0  # los bboxes vienen en px del frame real;
        # el centro X relativo no depende del ancho absoluto:
        # usamos el bbox tal cual (0–ancho) normalizado por el máximo

        ancho_imagen = max(
            (d.x_max or 0) for d in resultado_yolo.detecciones
        ) or 1.0

        menciones: list[MencionEscena] = []
        for det in resultado_yolo.detecciones:
            nombre = (
                traducir_etiqueta(det.etiqueta)
                or det.categoria
                or det.etiqueta
            )

            centro_x = ((det.x_min or 0) + (det.x_max or 0)) / 2 / ancho_imagen
            if centro_x < 0.33:
                posicion = "a tu izquierda"
            elif centro_x > 0.66:
                posicion = "a tu derecha"
            else:
                posicion = "al frente"

            etiqueta_lower = det.etiqueta.lower()
            if etiqueta_lower in PELIGRO:
                prioridad = 1
            elif etiqueta_lower in ATENCION:
                prioridad = 2
            else:
                prioridad = 3

            menciones.append(
                MencionEscena(
                    nombre=nombre,
                    posicion=posicion,
                    confianza=det.confianza,
                    prioridad=prioridad,
                )
            )

        # ─── 3. Orden: prioridad → confianza ───
        menciones.sort(key=lambda m: (m.prioridad, -m.confianza))

        # ─── 4. Descripción completa ───
        partes_completas = [
            f"{m.nombre} {m.posicion}" for m in menciones
        ]
        descripcion = "Escena con: " + ", ".join(partes_completas) + "."

        # ─── 5. Resumen TTS (máx. 3 + recuento) ───
        mencionadas = menciones[:MAX_EN_RESUMEN]
        partes_resumen = [
            f"{m.nombre} {m.posicion}" for m in mencionadas
        ]
        resumen = " ".join(p.capitalize() + "." for p in partes_resumen)

        if len(menciones) > MAX_EN_RESUMEN:
            restantes = len(menciones) - MAX_EN_RESUMEN
            resumen += f" Y {restantes} objeto(s) más."

        # ─── 6. Confianza promedio de lo mencionado ───
        confianza = round(
            sum(m.confianza for m in mencionadas) / len(mencionadas), 4
        )

        logger.info(
            "Escena: %d objetos, resumen con %d menciones.",
            len(menciones), len(mencionadas),
        )

        return ResultadoEscenaInterno(
            descripcion=descripcion,
            resumen=resumen,
            confianza=confianza,
            menciones=menciones,
        )