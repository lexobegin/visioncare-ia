"""
═══════════════════════════════════════════════════════════
VisionCare AI — Motor de colores (CU14)

Análisis NUMÉRICO (sin modelos — respuesta instantánea):

  1. RECORTE CENTRAL (60%): al apuntar un objeto, queda al
     centro del encuadre — evitamos el fondo (pared, cielo).
  2. HSV: PIL entrega H 0–255 → normalizamos a escala OpenCV
     0–179 (la de RANGOS_COLORES en traducciones.py).
  3. NEUTROS primero (por mediana): negro (v<0.12), blanco
     (v>0.85 y s<0.15), gris (s<0.15) — decision manual que
     supera a un hue cuando no hay saturación real.
  4. DOMINANTE: histograma de hue (18 bins × 10°) sobre
     píxeles cromáticos (s≥0.15, v≥0.12) → bin mayoritario →
     mediana del hue DENTRO del bin → color_desde_hsv().
  5. SECUNDARIOS: bins con ≥12% de los píxeles cromáticos,
     nombres distintos del dominante, máx. 2.
  6. CONFIANZA = pureza (proporción de píxeles que concuerdan
     con el color reportado), mapeada a 0.55–0.95.
═══════════════════════════════════════════════════════════
"""

import io
import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.ai_engine.traducciones import color_desde_hsv

logger = logging.getLogger("visioncare-ia.colores")

# Recorte central (fracción de cada dimensión)
FRACCION_CENTRAL = 0.60
# Downsample máximo para el análisis (velocidad)
LADO_MAXIMO = 200
# Umbrales de cromaticidad (s/v en 0–1)
S_MINIMA = 0.15
V_MINIMA = 0.12
# Bin secundario mínimo (% de píxeles cromáticos)
PCT_SECUNDARIO = 12.0


@dataclass
class ResultadoColorInterno:
    """Resultado del análisis de color."""

    color_principal: str
    colores_secundarios: list[str] = field(default_factory=list)
    confianza: float | None = None


class ColorAnalyzer:
    """Analizador de color dominante (stateless — thread-safe)."""

    async def analizar(self, imagen_bytes: bytes) -> ResultadoColorInterno:
        """CPU-light → ejecuta directo (el event loop no sufre)."""
        return self._analizar(imagen_bytes)

    def _analizar(self, imagen_bytes: bytes) -> ResultadoColorInterno:
        # ─── 1. Decodificar ───
        try:
            imagen = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise ValueError(
                "El archivo recibido no es una imagen válida "
                "(formatos aceptados: JPEG, PNG, WEBP)."
            )

        # ─── 2. Recorte central + downsample ───
        ancho, alto = imagen.size
        cx, cy = int(ancho * FRACCION_CENTRAL), int(alto * FRACCION_CENTRAL)
        x0, y0 = (ancho - cx) // 2, (alto - cy) // 2
        region = imagen.crop((x0, y0, x0 + cx, y0 + cy))

        if max(region.size) > LADO_MAXIMO:
            region.thumbnail((LADO_MAXIMO, LADO_MAXIMO))

        # PIL HSV: H 0–255, S 0–255, V 0–255
        hsv = np.asarray(region.convert("HSV"), dtype=np.float32)
        h_pil = hsv[:, :, 0]
        s = hsv[:, :, 1] / 255.0
        v = hsv[:, :, 2] / 255.0

        total = h_pil.size
        if total == 0:
            return ResultadoColorInterno("gris", [], 0.55)

        # ─── 3. Neutros por mediana global ───
        mediana_v = float(np.median(v))
        mediana_s = float(np.median(s))

        if mediana_v < V_MINIMA:
            return self._resultado("negro", [], total, total)

        if mediana_v > 0.85 and mediana_s < S_MINIMA:
            return self._resultado("blanco", [], total, total)

        if mediana_s < S_MINIMA:
            return self._resultado("gris", [], total, total)

        # ─── 4. Píxeles cromáticos → histograma de hue ───
        mascara = (s >= S_MINIMA) & (v >= V_MINIMA)
        n_cromaticos = int(mascara.sum())

        if n_cromaticos < total * 0.10:
            # Menos del 10% es cromático → dominan neutros
            if mediana_v < 0.5:
                return self._resultado("gris oscuro", [], total, total)
            return self._resultado("gris", [], total, total)

        # H a escala OpenCV 0–179
        h_opencv = (h_pil / 255.0) * 179.0
        hues = h_opencv[mascara]

        # 18 bins de 10°
        bins = np.histogram(hues, bins=18, range=(0, 180))[0]
        bin_dominante = int(bins.argmax())

        # Mediana del hue DENTRO del bin dominante (robusto)
        limite_inf = bin_dominante * 10
        limite_sup = limite_inf + 10
        en_bin = hues[(hues >= limite_inf) & (hues < limite_sup)]
        hue_mediana = float(np.median(en_bin))

        color_principal = color_desde_hsv(
            hue_mediana,
            float(np.median(s[mascara])),
            float(np.median(v[mascara])),
        )

        # ─── 5. Secundarios ───
        secundarios: list[str] = []
        orden_bins = np.argsort(bins)[::-1]  # mayor → menor
        for idx in orden_bins:
            if idx == bin_dominante:
                continue
            pct_bin = (bins[idx] / n_cromaticos) * 100
            if pct_bin < PCT_SECUNDARIO:
                break  # bins ordenados: los siguientes son menores

            inf, sup = int(idx) * 10, int(idx) * 10 + 10
            hue_bin = hues[(hues >= inf) & (hues < sup)]
            if hue_bin.size == 0:
                continue

            nombre = color_desde_hsv(
                float(np.median(hue_bin)),
                mediana_s,
                mediana_v,
            )
            if nombre != color_principal and nombre not in secundarios:
                secundarios.append(nombre)
            if len(secundarios) >= 2:
                break

        # ─── 6. Confianza = pureza del dominante ───
        pureza = bins[bin_dominante] / n_cromaticos
        # 0.40 pureza → 0.55 conf | 0.90 pureza → 0.95 conf
        confianza = round(min(0.95, 0.55 + pureza * 0.55), 2)

        logger.info(
            "Color: %s (sec: %s) — pureza %.0f%%.",
            color_principal, secundarios, pureza * 100,
        )

        return ResultadoColorInterno(
            color_principal=color_principal,
            colores_secundarios=secundarios or None,
            confianza=confianza,
        )

    @staticmethod
    def _resultado(
        nombre: str,
        secundarios: list[str],
        coinciden: int,
        total: int,
    ) -> ResultadoColorInterno:
        """Arma resultado neutro con confianza = proporción del neutro."""
        pureza = coinciden / total if total else 0.0
        return ResultadoColorInterno(
            color_principal=nombre,
            colores_secundarios=secundarios or None,
            confianza=round(min(0.95, 0.55 + pureza * 0.55), 2),
        )