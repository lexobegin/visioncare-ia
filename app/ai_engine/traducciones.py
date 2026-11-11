"""
═══════════════════════════════════════════════════════════
VisionCare AI — Diccionarios canónico de traducción/clasificación

SPRINT 1: TRADUCCION_COCO_ES (80 clases COCO → español)
SPRINT 2 (nuevos):
  - PALABRAS_SENAL:    OCR de señales/lugares (SALIDA, BAÑO…)
  - CATEGORIAS_PRODUCTO: clasificación de productos por clase/
                         palabras del empaque
  - NOMBRES_COLORES:   mapeo HSV → nombre de color en español
  - helpers: traducir_etiqueta, detectar_palabra_senal,
             categorizar_producto, color_desde_hsv
═══════════════════════════════════════════════════════════
"""

# ═══════════════════ SPRINT 1 ═══════════════════

TRADUCCION_COCO_ES: dict[str, str] = {
    # Personas y vehículos
    "person": "persona",
    "bicycle": "bicicleta",
    "car": "auto",
    "motorcycle": "motocicleta",
    "airplane": "avión",
    "bus": "bus",
    "train": "tren",
    "truck": "camión",
    "boat": "bote",
    # Señalización urbana
    "traffic light": "semáforo",
    "fire hydrant": "hidrante",
    "stop sign": "señal de stop",
    "parking meter": "parquímetro",
    "bench": "banca",
    # Animales
    "bird": "pájaro",
    "cat": "gato",
    "dog": "perro",
    "horse": "caballo",
    "sheep": "oveja",
    "cow": "vaca",
    "elephant": "elefante",
    "bear": "oso",
    "zebra": "cebra",
    "giraffe": "jirafa",
    # Objetos personales
    "backpack": "mochila",
    "umbrella": "paraguas",
    "handbag": "bolso",
    "tie": "corbata",
    "suitcase": "maleta",
    # Deportes
    "frisbee": "frisbee",
    "skis": "esquíes",
    "snowboard": "snowboard",
    "sports ball": "pelota",
    "kite": "cometa",
    "baseball bat": "bate de béisbol",
    "baseball glove": "guante de béisbol",
    "skateboard": "patineta",
    "surfboard": "tabla de surf",
    "tennis racket": "raqueta de tenis",
    # Cocina y comida
    "bottle": "botella",
    "wine glass": "copa de vino",
    "cup": "taza",
    "fork": "tenedor",
    "knife": "cuchillo",
    "spoon": "cuchara",
    "bowl": "tazón",
    "banana": "banana",
    "apple": "manzana",
    "sandwich": "sándwich",
    "orange": "naranja",
    "broccoli": "brócoli",
    "carrot": "zanahoria",
    "hot dog": "perrito caliente",
    "pizza": "pizza",
    "donut": "dona",
    "cake": "pastel",
    # Mobiliario
    "chair": "silla",
    "couch": "sofá",
    "potted plant": "planta",
    "bed": "cama",
    "dining table": "mesa",
    "toilet": "inodoro",
    # Electrónica
    "tv": "pantalla",
    "laptop": "computadora portátil",
    "mouse": "ratón",
    "remote": "control remoto",
    "keyboard": "teclado",
    "cell phone": "teléfono celular",
    "microwave": "microondas",
    "oven": "horno",
    "toaster": "tostadora",
    "sink": "lavabo",
    "refrigerator": "refrigerador",
    # Hogar y varios
    "book": "libro",
    "clock": "reloj",
    "vase": "jarrón",
    "scissors": "tijeras",
    "teddy bear": "oso de peluche",
    "hair drier": "secadora de pelo",
    "toothbrush": "cepillo de dientes",
}


def traducir_etiqueta(etiqueta: str) -> str | None:
    """
    Traduce una etiqueta COCO al español.
    Acepta variantes con guiones bajos ('traffic_light').
    """
    normalizada = etiqueta.strip().lower().replace("_", " ")
    return TRADUCCION_COCO_ES.get(normalizada)


# ═══════════════════ SPRINT 2 ═══════════════════

# ─── CU12: señales y lugares vía OCR de palabras clave ───
# (complementa las COCO 'stop sign'/'traffic light')
# Formato: (regex_sobre_texto_normalizado, tipo_senal, descripción)

import re

PALABRAS_SENAL: list[tuple[re.Pattern, str, str]] = [
    (
        re.compile(r"\bsalidas?\b|\bsalida de emergencia\b", re.IGNORECASE),
        "salida",
        "Salida de emergencia o de recinto",
    ),
    (
        re.compile(r"\bban[oa]s?\b|\bwc\b|\bsanitarios?\b|\bservicios?\b", re.IGNORECASE),
        "bano",
        "Baños / sanitarios",
    ),
    (
        re.compile(r"\bpeligro\b|\bcuidado\b|\briesgo\b", re.IGNORECASE),
        "peligro",
        "Señal de peligro",
    ),
    (
        re.compile(r"\bprohibido\b|\bno pasar\b|\bprohibida la entrada\b", re.IGNORECASE),
        "prohibicion",
        "Señal de prohibición",
    ),
    (
        re.compile(r"\bescaleras?\b", re.IGNORECASE),
        "escaleras",
        "Escaleras",
    ),
    (
        re.compile(r"\bascensor\b|\belevador\b", re.IGNORECASE),
        "ascensor",
        "Ascensor",
    ),
    (
        re.compile(r"\bcaja[s]?\b|\bcajas?\b", re.IGNORECASE),
        "cajas",
        "Cajas / atención al cliente",
    ),
    (
        re.compile(r"\bemergencia\b|\bemergencias\b", re.IGNORECASE),
        "emergencia",
        "Punto de emergencia",
    ),
    (
        re.compile(r"\bprimero[s]? auxilio[s]?\b", re.IGNORECASE),
        "primeros_auxilios",
        "Botiquín de primeros auxilios",
    ),
    (
        re.compile(r"\bentrada\b", re.IGNORECASE),
        "entrada",
        "Entrada",
    ),
]


def detectar_palabra_senal(texto: str) -> tuple[str, str] | None:
    """
    Busca palabras clave de señales en un texto OCR.
    Devuelve (tipo_senal, descripción) o None.
    """
    for patron, tipo, descripcion in PALABRAS_SENAL:
        if patron.search(texto):
            return (tipo, descripcion)
    return None


# ─── CU13: categorías de producto ───
# Clases COCO "comerciales" → categoría + nombre base

PRODUCTOS_COCO: dict[str, tuple[str, str]] = {
    # etiqueta → (categoría, nombre genérico)
    "bottle": ("bebida", "botella"),
    "wine glass": ("bebida", "copa"),
    "cup": ("bebida", "taza/vaso"),
    "banana": ("fruta", "banana"),
    "apple": ("fruta", "manzana"),
    "orange": ("fruta", "naranja"),
    "sandwich": ("comida", "sándwich"),
    "hot dog": ("comida", "perrito caliente"),
    "pizza": ("comida", "pizza"),
    "donut": ("panadería", "dona"),
    "cake": ("panadería", "pastel"),
    "broccoli": ("verdura", "brócoli"),
    "carrot": ("verdura", "zanahoria"),
    "book": ("librería", "libro"),
    "toothbrush": ("higiene", "cepillo de dientes"),
    "hair drier": ("higiene", "secadora"),
    "backpack": ("accesorios", "mochila"),
    "handbag": ("accesorios", "bolso"),
    "remote": ("electrónica", "control remoto"),
    "cell phone": ("electrónica", "teléfono celular"),
    "keyboard": ("electrónica", "teclado"),
    "clock": ("hogar", "reloj"),
    "vase": ("hogar", "jarrón"),
}

# Marcas conocidas para OCR de empaques (heurística de demo,
# ampliable con un catálogo real en S3)
MARCAS_CONOCIDAS: list[str] = [
    "coca cola", "cocacola", "pepsi", "sprite", "fanta",
    "lays", "doritos", "oreo", "nestle", "bimbo",
    "pillow", "pillow's", "kyk",
]


def categorizar_producto(
    etiqueta_coco: str | None,
    texto_ocr: str | None,
) -> tuple[str | None, str | None, str | None]:
    """
    Combina clase COCO + OCR del empaque para derivar
    (nombre_producto, marca, categoria).
    """
    nombre: str | None = None
    marca: str | None = None
    categoria: str | None = None

    # 1) Clase COCO
    if etiqueta_coco:
        info = PRODUCTOS_COCO.get(etiqueta_coco.lower())
        if info:
            categoria, nombre = info

    # 2) OCR del empaque
    if texto_ocr:
        texto = texto_ocr.lower()

        # Marca conocida
        for m in MARCAS_CONOCIDAS:
            if m in texto:
                marca = m.title()
                break

        # Producto en mayúsculas del empaque (heurística:
        # los empaques ponen el nombre en caps)
        lineas = [
            l.strip()
            for l in (texto_ocr or "").split("\n")
            if l.strip()
        ]
        for linea in lineas:
            # Línea "mayúscula" (≥3 chars, ≥60% mayúsculas)
            letras = [c for c in linea if c.isalpha()]
            if len(linea) >= 3 and letras and sum(
                1 for c in letras if c.isupper()
            ) / len(letras) >= 0.6:
                nombre = nombre or linea[:150]
                break

    return (nombre, marca, categoria)


# ─── CU14: colores — rangos HSV → nombre español ───
# (H en grados 0–179 según convención OpenCV)

RANGOS_COLORES: list[tuple[str, int, int]] = [
    # (nombre, hue_min, hue_max)
    ("rojo", 0, 10),
    ("naranja", 11, 25),
    ("amarillo", 26, 34),
    ("verde claro", 35, 49),
    ("verde", 50, 78),
    ("celeste", 79, 96),
    ("azul", 97, 130),
    ("violeta", 131, 155),
    ("rosa", 156, 169),
    ("rojo", 170, 179),  # el rojo envuelve el círculo cromático
]

# Neutros definidos por saturación/valor (fuera del hue)
NOMBRE_NEGRO = "negro"
NOMBRE_BLANCO = "blanco"
NOMBRE_GRIS = "gris"
NOMBRE_MARRON = "marrón"


def color_desde_hsv(
    h: float, s: float, v: float
) -> str:
    """
    Convierte (hue 0–179, sat 0–1, val 0–1) a nombre en español.
    Prioriza neutros por saturación/valor.
    """
    # Muy oscuro → negro
    if v < 0.12:
        return NOMBRE_NEGRO
    # Muy claro y poco saturado → blanco
    if v > 0.85 and s < 0.15:
        return NOMBRE_BLANCO
    # Poco saturado → gris (o marrón si hue cálido con s media)
    if s < 0.15:
        return NOMBRE_GRIS

    # Marrón: naranja oscuro/desaturado
    if 10 <= h <= 25 and v < 0.55:
        return NOMBRE_MARRON

    for nombre, h_min, h_max in RANGOS_COLORES:
        if h_min <= h <= h_max:
            return nombre

    return NOMBRE_GRIS