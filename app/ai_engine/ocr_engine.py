"""
═══════════════════════════════════════════════════════════
VisionCare AI — Motor OCR (EasyOCR)

Encapsula easyocr.Reader con:

  - CARGA PEREZOSA: el reader se instancia en el primer uso
    (o en el warmup del lifespan) — la 1ª vez descarga los
    modelos (~80 MB, idiomas es+en).
  - THREAD-SAFETY: easyocr no soporta inferencia concurrente
    sobre la misma instancia → threading.Lock serializa.
  - NO-BLOQUEO: la inferencia corre en asyncio.to_thread().
  - ORDEN DE LECTURA: las líneas se ordenan arriba→abajo y
    izquierda→derecha usando el bbox que devuelve EasyOCR.
  - CONFIANZA PONDERADA: por longitud de cada línea detectada.

Salida (ResultadoOCR): texto fusionado + confianza + líneas.
═══════════════════════════════════════════════════════════
"""

import asyncio
import io
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import settings

logger = logging.getLogger("visioncare-ia.ocr")


@dataclass
class LineaTexto:
    """Una línea de texto detectada (con posición para el orden)."""

    texto: str
    confianza: float
    y_centro: int   # posición vertical (orden de lectura)
    x_inicio: int   # posición horizontal (desempate)
    alto: int       # altura del bbox en px (tamaño relativo del texto)


@dataclass
class ResultadoOCR:
    """Resultado completo de la lectura."""

    texto: str                       # líneas fusionadas con \n
    confianza: float | None          # ponderada por longitud
    lineas: list[LineaTexto] = field(default_factory=list)


class OcrReader:
    """Wrapper thread-safe de EasyOCR (gestionado por OcrService)."""

    def __init__(self) -> None:
        self._reader = None
        self._lock = threading.Lock()
        self._error_inicializacion: str | None = None

    # ═══════════════════ CARGA ═══════════════════

    def _obtener_storage_dir(self) -> str:
        """Determina la ruta de modelos, priorizando el directorio local models/."""
        ruta_dir = Path(getattr(settings, "ocr_model_dir", "models"))
        if not ruta_dir.is_absolute():
            raiz = Path(__file__).resolve().parent.parent.parent
            ruta_dir = raiz / ruta_dir
        ruta_dir.mkdir(parents=True, exist_ok=True)

        # Si el directorio local no tiene los modelos, intentar copiar desde ~/.EasyOCR/model
        craft_file = ruta_dir / "craft_mlt_25k.pth"
        latin_file = ruta_dir / "latin_g2.pth"
        if not craft_file.exists() or not latin_file.exists():
            try:
                home_dir = Path.home() / ".EasyOCR" / "model"
                if home_dir.exists():
                    for nombre in ["craft_mlt_25k.pth", "latin_g2.pth"]:
                        origen = home_dir / nombre
                        destino = ruta_dir / nombre
                        if origen.exists() and not destino.exists():
                            shutil.copy2(origen, destino)
                            logger.info("Modelo %s sincronizado desde %s a %s", nombre, origen, destino)
            except (PermissionError, OSError) as e:
                logger.debug("No se pudo acceder a ~/.EasyOCR/model: %s", e)

        return str(ruta_dir)

    def _obtener_reader(self):
        """Instancia el reader si aún no está cargado (thread-safe)."""
        if self._reader is None:
            with self._lock:
                if self._reader is None:  # doble chequeo
                    storage_dir = self._obtener_storage_dir()
                    logger.info(
                        "Cargando EasyOCR (idiomas: %s, gpu: %s, dir: %s)...",
                        settings.ocr_idiomas_list,
                        settings.ocr_gpu,
                        storage_dir,
                    )
                    inicio = time.perf_counter()

                    import easyocr  # import perezoso

                    try:
                        self._reader = easyocr.Reader(
                            settings.ocr_idiomas_list,
                            gpu=settings.ocr_gpu,
                            model_storage_directory=storage_dir,
                            verbose=False,
                        )
                        self._error_inicializacion = None
                    except Exception as e:
                        self._error_inicializacion = str(e)
                        logger.error("Error al instanciar EasyOCR Reader: %s", e)
                        raise

                    duracion = (time.perf_counter() - inicio) * 1000
                    logger.info("EasyOCR cargado en %.0f ms.", duracion)
        return self._reader

    async def precargar(self) -> None:
        """
        Warmup: carga el reader (descarga modelos si falta) y
        ejecuta una inferencia mínima (se llama desde lifespan).
        """
        def _warmup() -> None:
            try:
                reader = self._obtener_reader()
                imagen_dummy = np.zeros((64, 320, 3), dtype=np.uint8)
                reader.readtext(imagen_dummy, detail=0, paragraph=False)
            except Exception as e:
                self._error_inicializacion = str(e)
                logger.error("Fallo durante el warmup de EasyOCR: %s", e)
                raise

        await asyncio.to_thread(_warmup)
        logger.info("Warmup de OCR completado — listo para peticiones.")

    @property
    def cargado(self) -> bool:
        return self._reader is not None

    @property
    def error_inicializacion(self) -> str | None:
        return self._error_inicializacion

    @property
    def model_dir(self) -> str:
        return self._obtener_storage_dir()

    @property
    def info_modelo(self) -> dict[str, str]:
        return {
            "nombre": "easyocr",
            "version": "1.0",
            "idiomas": ",".join(settings.ocr_idiomas_list),
            "storage_dir": self._obtener_storage_dir(),
        }

    # ═══════════════════ INFERENCIA ═══════════════════

    async def leer(self, imagen_bytes: bytes) -> ResultadoOCR:
        """
        Lee todo el texto de la imagen.

        Raises:
            ValueError: si los bytes no son una imagen válida.
        """
        return await asyncio.to_thread(self._leer_sincrono, imagen_bytes)

    def _leer_sincrono(self, imagen_bytes: bytes) -> ResultadoOCR:
        """Inferencia real (sincrónica, protegida por lock)."""

        # Validación fail-fast: si el modelo falló su inicialización, abortar de inmediato (< 1ms)
        if self._reader is None and self._error_inicializacion:
            raise RuntimeError(
                f"El motor OCR no está disponible (error en inicialización: {self._error_inicializacion}). "
                f"Verifique los archivos en '{settings.ocr_model_dir}' o ejecute 'python prepare_models.py'."
            )

        # ─── 1. Validar y decodificar (Pillow) ───
        try:
            imagen = Image.open(io.BytesIO(imagen_bytes))
            imagen.load()  # Forzar decodificación para detectar imágenes corruptas/truncadas
            imagen = imagen.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            raise ValueError(
                "El archivo recibido no es una imagen válida o está dañado "
                "(formatos aceptados: JPEG, PNG, WEBP)."
            )

        ancho_orig, alto_orig = imagen.size
        if ancho_orig <= 0 or alto_orig <= 0:
            raise ValueError("Las dimensiones de la imagen son inválidas.")

        # ─── 2. Redimensionamiento adaptativo (prevención de OOM) ───
        # Limita la dimensión máxima para evitar que el detector CRAFT de EasyOCR asigne
        # tensores excesivos (ej: >1.2 GB en imágenes de celular >= 12MP en CPU).
        max_dim = getattr(settings, "ocr_max_dimension", 1280)
        canvas_size = getattr(settings, "ocr_canvas_size", max_dim)

        dim_mayor = max(ancho_orig, alto_orig)
        if dim_mayor > max_dim:
            factor_escala = max_dim / dim_mayor
            nuevo_ancho = max(1, int(round(ancho_orig * factor_escala)))
            nuevo_alto = max(1, int(round(alto_orig * factor_escala)))
            imagen_proc = imagen.resize(
                (nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS
            )
        else:
            factor_escala = 1.0
            imagen_proc = imagen

        arreglo = np.array(imagen_proc)

        # ─── 3. Inferencia EasyOCR (sección crítica) ───
        with self._lock:
            inicio = time.perf_counter()

            resultados = self._obtener_reader().readtext(
                arreglo,
                detail=1,        # [bbox, texto, confianza]
                paragraph=False, # líneas individuales (control del orden)
                canvas_size=canvas_size,
            )

            duracion_ms = int((time.perf_counter() - inicio) * 1000)

        # ─── 4. Normalizar líneas + orden de lectura (escalando a coords originales) ───
        lineas: list[LineaTexto] = []

        for bbox, texto, conf in resultados:
            texto_limpio = str(texto).strip()
            if not texto_limpio:
                continue

            if factor_escala != 1.0:
                xs = [p[0] / factor_escala for p in bbox]
                ys = [p[1] / factor_escala for p in bbox]
            else:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]

            lineas.append(
                LineaTexto(
                    texto=texto_limpio,
                    confianza=float(conf),
                    y_centro=int(sum(ys) / len(ys)),
                    x_inicio=int(min(xs)),
                    alto=int(max(ys) - min(ys)),
                )
            )

        # Orden natural de lectura: arriba→abajo, luego izquierda→derecha
        lineas.sort(key=lambda l: (l.y_centro, l.x_inicio))

        # ─── 4. Texto fusionado + confianza ponderada ───
        texto = "\n".join(l.texto for l in lineas)

        confianza: float | None = None
        if lineas:
            total_caracteres = sum(len(l.texto) for l in lineas)
            if total_caracteres > 0:
                confianza = round(
                    sum(
                        l.confianza * len(l.texto) for l in lineas
                    ) / total_caracteres,
                    4,
                )

        logger.info(
            "OCR: %d líneas en %d ms.", len(lineas), duracion_ms
        )

        return ResultadoOCR(
            texto=texto,
            confianza=confianza,
            lineas=lineas,
        )