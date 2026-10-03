"""
═══════════════════════════════════════════════════════════
VisionCare AI — Detector de objetos (YOLOv8)

Encapsula ultralytics.YOLO con:

  - CARGA PEREZOSA: el modelo se instancia en el primer uso
    (o en el warmup del lifespan) — no al importar el módulo.
  - THREAD-SAFETY: YOLO no soporta inferencia concurrente
    sobre la misma instancia → threading.Lock serializa.
  - NO-BLOQUEO: la inferencia corre en asyncio.to_thread()
    para liberar el event loop de FastAPI.
  - VALIDACIÓN DE ENTRADA: decodifica con Pillow antes de
    pasar el array a YOLO (mensajes claros ante imagen
    corrupta o formato inválido).

Salida = contrato de vision-ia.client.ts (NestJS):
    { modelo, duracion_ms, detecciones[] }
═══════════════════════════════════════════════════════════
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import asyncio

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.ai_engine.traducciones import traducir_etiqueta
from app.config import settings

logger = logging.getLogger("visioncare-ia.detector")


@dataclass
class Deteccion:
    """Una detección individual (objeto encontrado en la imagen)."""

    etiqueta: str              # original del modelo (inglés, ej: 'person')
    categoria: str | None      # traducción al español (ej: 'persona')
    confianza: float           # 0–1
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class ResultadoDeteccion:
    """Resultado completo del análisis (contrato hacia NestJS)."""

    modelo_nombre: str
    modelo_version: str
    duracion_ms: int
    detecciones: list[Deteccion] = field(default_factory=list)


class DetectorObjetos:
    """
    Wrapper thread-safe de YOLOv8 (singleton por proceso,
    gestionado como dependencia en main.py — Tanda 7).
    """

    def __init__(self) -> None:
        self._modelo = None                       # carga perezosa
        self._lock = threading.Lock()             # serializa carga + inferencia
        self._nombre = Path(settings.yolo_model).stem   # ej: 'yolov8n'
        self._version = "1.0"                     # versión de nuestro despliegue

    # ═══════════════════ CARGA ═══════════════════

    def _obtener_modelo(self):
        """Instancia el modelo si aún no está cargado (thread-safe)."""
        if self._modelo is None:
            with self._lock:
                if self._modelo is None:          # doble chequeo
                    logger.info(
                        "Cargando modelo YOLO '%s' en dispositivo '%s'...",
                        settings.yolo_model,
                        settings.yolo_device,
                    )
                    inicio = time.perf_counter()

                    from ultralytics import YOLO  # import perezoso

                    self._modelo = YOLO(settings.yolo_model)

                    duracion = (time.perf_counter() - inicio) * 1000
                    logger.info(
                        "Modelo YOLO cargado en %.0f ms.", duracion
                    )
        return self._modelo

    async def precargar(self) -> None:
        """
        Warmup: carga el modelo ANTES de atender peticiones
        (se llama desde el lifespan de main.py). Ejecuta una
        inferencia mínima para inicializar el backend de torch.
        """
        def _warmup() -> None:
            modelo = self._obtener_modelo()
            imagen_dummy = np.zeros(
                (settings.yolo_imgsz, settings.yolo_imgsz, 3), dtype=np.uint8
            )
            modelo.predict(
                imagen_dummy,
                verbose=False,
                device=settings.yolo_device,
            )

        await asyncio.to_thread(_warmup)
        logger.info("Warmup de YOLO completado — listo para peticiones.")

    @property
    def cargado(self) -> bool:
        return self._modelo is not None

    @property
    def info_modelo(self) -> dict[str, str]:
        return {"nombre": self._nombre, "version": self._version}

    # ═══════════════════ INFERENCIA ═══════════════════

    async def detectar(
        self,
        imagen_bytes: bytes,
        umbral_confianza: float | None = None,
    ) -> ResultadoDeteccion:
        """
        Analiza una imagen (bytes) y devuelve las detecciones.

        Args:
            imagen_bytes: contenido de la imagen (JPEG/PNG/WEBP).
            umbral_confianza: override opcional de
                settings.yolo_confidence (0–1).

        Raises:
            ValueError: si los bytes no son una imagen válida.
        """
        umbral = umbral_confianza or settings.yolo_confidence

        # La parte pesada (CPU/GPU) fuera del event loop
        return await asyncio.to_thread(
            self._inferir_sincrono, imagen_bytes, umbral
        )

    def _inferir_sincrono(
        self, imagen_bytes: bytes, umbral: float
    ) -> ResultadoDeteccion:
        """Inferencia real (sincrónica, protegida por lock)."""

        # ─── 1. Validar y decodificar la imagen (Pillow) ───
        try:
            imagen = Image.open(
                __import__("io").BytesIO(imagen_bytes)
            ).convert("RGB")
        except UnidentifiedImageError:
            raise ValueError(
                "El archivo recibido no es una imagen válida "
                "(formatos aceptados: JPEG, PNG, WEBP)."
            )
        arreglo = np.array(imagen)

        # ─── 2. Inferencia YOLO (sección crítica) ───
        with self._lock:
            inicio = time.perf_counter()

            modelo = self._obtener_modelo()
            resultados = modelo.predict(
                arreglo,
                conf=umbral,
                imgsz=settings.yolo_imgsz,
                device=settings.yolo_device,
                verbose=False,
            )

            duracion_ms = int((time.perf_counter() - inicio) * 1000)

        # ─── 3. Extraer detecciones del resultado ───
        detecciones: list[Deteccion] = []

        for resultado in resultados:
            cajas = resultado.boxes
            if cajas is None:
                continue

            # xyxy → numpy en píxeles | conf y cls → tensores
            coordenadas = cajas.xyxy.cpu().numpy()
            confianzas = cajas.conf.cpu().numpy()
            clases = cajas.cls.cpu().numpy()
            nombres = resultado.names  # {id_clase: etiqueta}

            for i in range(len(coordenadas)):
                etiqueta = str(nombres.get(int(clases[i]), "desconocido"))
                x_min, y_min, x_max, y_max = (
                    float(v) for v in coordenadas[i]
                )

                # Coherencia con CHECK de detecciones_ia:
                # x_max > x_min, y_max > y_min, valores >= 0
                if x_max <= x_min or y_max <= y_min:
                    continue

                detecciones.append(
                    Deteccion(
                        etiqueta=etiqueta,
                        categoria=traducir_etiqueta(etiqueta),
                        confianza=round(float(confianzas[i]), 4),
                        x_min=round(max(0.0, x_min), 2),
                        y_min=round(max(0.0, y_min), 2),
                        x_max=round(max(0.0, x_max), 2),
                        y_max=round(max(0.0, y_max), 2),
                    )
                )

        # Ordenar por confianza DESC (como espera el detalle de captura)
        detecciones.sort(key=lambda d: d.confianza, reverse=True)

        logger.info(
            "YOLO: %d detecciones en %d ms (umbral %.2f).",
            len(detecciones),
            duracion_ms,
            umbral,
        )

        return ResultadoDeteccion(
            modelo_nombre=self._nombre,
            modelo_version=self._version,
            duracion_ms=duracion_ms,
            detecciones=detecciones,
        )