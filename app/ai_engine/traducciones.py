"""
═══════════════════════════════════════════════════════════
VisionCare AI — Diccionario canónico COCO → Español

Las 80 clases del dataset COCO (con el que fue entrenado
YOLOv8) traducidas para el feedback por voz (TTS).

El diccionario `traducir_etiqueta` es tolerante: si la
etiqueta no está en el diccionario, devuelve None y el
llamante decide (usar original, filtrar, etc.).

Sprint 2: este módulo crecerá con diccionarios de señales,
productos y billetes (separados por función).
═══════════════════════════════════════════════════════════
"""

TRADUCCION_COCO_ES: dict[str, str] = {
    # ─── Personas y vehículos ───
    "person": "persona",
    "bicycle": "bicicleta",
    "car": "auto",
    "motorcycle": "motocicleta",
    "airplane": "avión",
    "bus": "bus",
    "train": "tren",
    "truck": "camión",
    "boat": "bote",
    # ─── Señalización urbana (clave para orientación) ───
    "traffic light": "semáforo",
    "fire hydrant": "hidrante",
    "stop sign": "señal de stop",
    "parking meter": "parquímetro",
    "bench": "banca",
    # ─── Animales ───
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
    # ─── Objetos personales ───
    "backpack": "mochila",
    "umbrella": "paraguas",
    "handbag": "bolso",
    "tie": "corbata",
    "suitcase": "maleta",
    # ─── Deportes ───
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
    # ─── Cocina y comida ───
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
    # ─── Mobiliario ───
    "chair": "silla",
    "couch": "sofá",
    "potted plant": "planta",
    "bed": "cama",
    "dining table": "mesa",
    "toilet": "inodoro",
    # ─── Electrónica ───
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
    # ─── Hogar y varios ───
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
    Devuelve None si no está en el diccionario.
    """
    normalizada = etiqueta.strip().lower().replace("_", " ")
    return TRADUCCION_COCO_ES.get(normalizada)