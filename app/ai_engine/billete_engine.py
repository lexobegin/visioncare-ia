"""
═══════════════════════════════════════════════════════════
VisionCare AI — Motor de billetes (CU15 reconocimiento,
CU16 autenticidad estimada)

COMPOSICIÓN de motores existentes (sin modelos nuevos):

  1. OCR (OcrReader singleton) → texto del billete con
     posición y ALTO de cada línea (la denominación impresa
     es GRANDE → bbox alto)
  2. DENOMINACIÓN: candidatos numéricos ponderados por alto
     de texto + números en palabra (CIEN, TWENTY…), validados
     contra los sets reales por moneda
  3. MONEDA: marcadores textuales (BOLIVIANOS, Bs, DOLLARS,
     EUROS…); sin marcador → None (la voz del backend lo maneja)
  4. COLOR (ColorAnalyzer) → consistencia color↔denominación
     (tablas aproximadas por moneda — heurística documentada)
  5. AUTENTICIDAD = puntaje de 4 señales (NUNCA certificación):
       +1 denominación legible
       +1 denominación aparece 2+ veces (grande + pequeñas)
       +1 color consistente con la denominación esperada
       +1 confianza del OCR ≥ 0.70
     → PROBABLEMENTE_AUTENTICO (3–4 pts)
       INDETERMINADO (2 pts o datos insuficientes)
       PROBABLEMENTE_SOSPECHOSO (0–1 pt CON denominación legible)

⚠️ El resultado es ORIENTATIVO — el mensaje al usuario SIEMPRE
   lo comunica como estimación (lo hace el backend en voz).
═══════════════════════════════════════════════════════════
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field

from app.ai_engine.color_engine import ColorAnalyzer
from app.ai_engine.ocr_engine import OcrReader

logger = logging.getLogger("visioncare-ia.billetes")

# ═══════════════ CATÁLOGOS ═══════════════

# Marcadores de moneda (regex sobre texto normalizado)
MONEDA_PATRONES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bbolivianos?\b|\bb[sz]\.?\b", re.IGNORECASE), "BOB"),
    (re.compile(r"\bdollars?\b|\bd[oó]lares?\b|\busd\b|\$\b", re.IGNORECASE), "USD"),
    (re.compile(r"\beuros?\b|\beur\b|\b€\b", re.IGNORECASE), "EUR"),
]

# Denominaciones válidas por moneda (billetes en circulación)
DENOMINACIONES_VALIDAS: dict[str, set[int]] = {
    "BOB": {10, 20, 50, 100, 200},
    "USD": {1, 2, 5, 10, 20, 50, 100},
    "EUR": {5, 10, 20, 50, 100, 200, 500},
}

# Números en palabra (billetes es/en) — normalizados sin tildes
NUMEROS_PALABRA: dict[str, int] = {
    "diez": 10, "veinte": 20, "cincuenta": 50,
    "cien": 100, "doscientos": 200,
    "ten": 10, "twenty": 20, "fifty": 50,
    "hundred": 100, "two hundred": 200,
}

# Color esperado por denominación (HEURÍSTICA aproximada —
# diseños vigentes; se evalúa como "compatible" con familias)
COLOR_ESPERADO: dict[str, dict[int, set[str]]] = {
    "BOB": {
        10: {"rojo", "marrón"},
        20: {"naranja", "rojo"},
        50: {"violeta", "rosa", "marrón"},
        100: {"azul", "celeste"},
        200: {"verde", "verde claro"},
    },
    "USD": {
        1: {"verde", "verde claro"},
        2: {"verde", "verde claro"},
        5: {"verde", "verde claro"},
        10: {"naranja", "verde"},
        20: {"verde", "verde claro"},
        50: {"rosa", "verde"},
        100: {"azul", "verde"},
    },
    "EUR": {
        5: {"gris", "gris oscuro"},
        10: {"rojo", "marrón"},
        20: {"azul", "celeste"},
        50: {"naranja", "marrón"},
        100: {"verde", "verde claro"},
        200: {"amarillo", "verde claro"},
        500: {"violeta", "rosa"},
    },
}

ESTADO_AUTENTICO = "PROBABLEMENTE_AUTENTICO"
ESTADO_SOSPECHOSO = "PROBABLEMENTE_SOSPECHOSO"
ESTADO_INDETERMINADO = "INDETERMINADO"


@dataclass
class ResultadoBilleteInterno:
    """Resultado del análisis de billete."""

    moneda: str | None
    denominacion: str | None
    confianza: float | None
    autenticidad_estimacion: str | None
    confianza_autenticidad: float | None
    observaciones: list[str] = field(default_factory=list)


class BilleteEngine:
    """Analizador de billetes por composición OCR + color."""

    def __init__(
        self,
        reader_ocr: OcrReader | None = None,
        analizador_color: ColorAnalyzer | None = None,
    ) -> None:
        # Por defecto: instancias compartidas se inyectan desde
        # el service (singletons de ocr_service / color_service)
        self.reader_ocr = reader_ocr or OcrReader()
        self.color_analyzer = analizador_color or ColorAnalyzer()

    # ═══════════════════ ANÁLISIS ═══════════════════

    async def analizar(self, imagen_bytes: bytes) -> ResultadoBilleteInterno:
        return await asyncio.to_thread(self._analizar_sincrono, imagen_bytes)

    def _analizar_sincrono(self, imagen_bytes: bytes) -> ResultadoBilleteInterno:
        observaciones: list[str] = []

        # ─── 1. OCR del billete (lock interno del reader) ───
        try:
            lectura = self.reader_ocr._leer_sincrono(imagen_bytes)
        except ValueError:
            raise  # imagen inválida → 400 en el service
        except Exception as e:
            raise RuntimeError(f"OCR falló durante el análisis: {e}")

        texto = lectura.texto
        texto_normalizado = re.sub(r"\s+", " ", texto.lower())

        # ─── 2. Moneda por marcadores ───
        moneda: str | None = None
        for patron, codigo in MONEDA_PATRONES:
            if patron.search(texto_normalizado):
                moneda = codigo
                break

        # ─── 3. Denominación (número grande + palabra) ───
        denominacion, fuente_denom, conf_denom = self._extraer_denominacion(
            lectura, moneda
        )

        if denominacion is None:
            observaciones.append(
                "No se leyó una denominación clara; "
                "coloca el billete plano y bien iluminado."
            )
            return ResultadoBilleteInterno(
                moneda=moneda,
                denominacion=None,
                confianza=lectura.confianza,
                autenticidad_estimacion=None,
                confianza_autenticidad=None,
                observaciones=observaciones,
            )

        observaciones.append(
            f"Denominación {denominacion} leída por {fuente_denom}."
        )

        # ─── 4. Color del billete ───
        color_bill: str | None = None
        color_consistente: bool | None = None
        try:
            resultado_color = self.color_analyzer._analizar(imagen_bytes)
            color_bill = resultado_color.color_principal

            if moneda and moneda in COLOR_ESPERADO:
                esperados = COLOR_ESPERADO[moneda].get(denominacion)
                if esperados:
                    color_consistente = color_bill in esperados
                    if color_consistente:
                        observaciones.append(
                            f"Color {color_bill} consistente con "
                            f"{denominacion} {moneda}."
                        )
                    else:
                        observaciones.append(
                            f"Color {color_bill} no coincide con el "
                            f"habitual de {denominacion} {moneda}."
                        )
        except Exception:
            color_bill = None  # color es best-effort

        # ─── 5. Denominación repetida (señal de autenticidad) ───
        repeticiones = len(
            re.findall(rf"\b{denominacion}\b", texto_normalizado)
        )
        if repeticiones >= 2:
            observaciones.append(
                f"Denominación visible en {repeticiones} posiciones."
            )

        # ─── 6. Puntaje de autenticidad (4 señales) ───
        puntos = 0
        puntos += 1  # denominación legible (llegamos hasta aquí)
        if repeticiones >= 2:
            puntos += 1
        if color_consistente is True:
            puntos += 1
        elif color_consistente is False:
            puntos -= 1  # inconsistencia es señal negativa activa
        if (lectura.confianza or 0) >= 0.70:
            puntos += 1

        if puntos >= 3:
            estado = ESTADO_AUTENTICO
            conf_autenticidad = min(0.80, 0.55 + puntos * 0.08)
        elif puntos == 2:
            estado = ESTADO_INDETERMINADO
            conf_autenticidad = 0.45
        else:
            estado = ESTADO_SOSPECHOSO
            conf_autenticidad = 0.50

        observaciones.append(
            "Estimación visual orientativa — NO constituye "
            "certificación de autenticidad."
        )

        # Confianza de la denominación: OCR ponderada por alto
        # ya reflejada; ajustamos con la validación del set
        conf_final = round(min(0.97, (conf_denom or 0.6) + 0.05), 4)

        logger.info(
            "Billete: %s %s (%s) — autenticidad %s (%d pts).",
            denominacion, moneda or "?", fuente_denom, estado, puntos,
        )

        return ResultadoBilleteInterno(
            moneda=moneda,
            denominacion=str(denominacion),
            confianza=conf_final,
            autenticidad_estimacion=estado,
            confianza_autenticidad=round(conf_autenticidad, 2),
            observaciones=observaciones,
        )

    # ═══════════════ DENOMINACIÓN ═══════════════

    def _extraer_denominacion(
        self,
        lectura,          # ResultadoOCR (con lineas[].alto)
        moneda: str | None,
    ) -> tuple[int | None, str, float | None]:
        """
        Extrae la denominación más probable.

        Estrategia:
          a) Candidatos numéricos de las líneas con MAYOR alto
             de texto (la denominación se imprime grande).
          b) Números en palabra del texto completo.
          c) Validación contra el set de la moneda (si hay);
             sin moneda, se acepta cualquier valor 1–500 y la
             moneda queda None.
        Devuelve (valor, fuente, confianza) o (None, '', None).
        """
        candidatos: list[tuple[int, float, str]] = []  # (valor, score, fuente)

        # ─── a) Números escritos, ponderados por alto ───
        if lectura.lineas:
            alto_max = max(l.alto for l in lectura.lineas) or 1

            for linea in lectura.lineas:
                numeros = re.findall(r"\b(\d{1,4})\b", linea.texto)
                for num in numeros:
                    valor = int(num)
                    if valor < 1 or valor > 500:
                        continue
                    # score: confianza OCR × tamaño relativo
                    peso_tamano = 0.5 + 0.5 * (linea.alto / alto_max)
                    score = round(linea.confianza * peso_tamano, 4)
                    candidatos.append((valor, score, "texto grande"))

        # ─── b) Números en palabra ───
        texto_norm = re.sub(r"\s+", " ", lectura.texto.lower())
        for palabra, valor in NUMEROS_PALABRA.items():
            if re.search(rf"\b{palabra}\b", texto_norm):
                candidatos.append((valor, 0.70, "palabra"))

        if not candidatos:
            return (None, "", None)

        # ─── c) Validación por moneda ───
        set_valido = (
            DENOMINACIONES_VALIDAS.get(moneda) if moneda else None
        )

        if set_valido:
            validos = [c for c in candidatos if c[0] in set_valido]
            if validos:
                # mejor score entre los válidos
                mejor = max(validos, key=lambda c: c[1])
                return (mejor[0], mejor[2], mejor[1])
            # nada coincide con la moneda detectada → sospechoso de
            # mala lectura; devolvemos el mejor crudo igualmente
            mejor = max(candidatos, key=lambda c: c[1])
            return (mejor[0], mejor[2], round((mejor[1] or 0.5) * 0.8, 4))

        # Sin moneda: aceptar el mejor candidato plausible
        mejor = max(candidatos, key=lambda c: c[1])
        return (mejor[0], mejor[2], mejor[1])