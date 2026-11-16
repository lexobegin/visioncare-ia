"""
═══════════════════════════════════════════════════════════
VisionCare AI — Gestor de Artefactos y Modelos de Inferencia
Sprint 2/3 (DevOps & MLOps): Aprovisionamiento determinista
de pesos para YOLOv8 y EasyOCR (CRAFT + Latin).

Uso:
    python prepare_models.py
    python prepare_models.py --verify-only
═══════════════════════════════════════════════════════════
"""

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# URLs oficiales de los modelos de EasyOCR (JaidedAI)
EASYOCR_URLS = {
    "craft_mlt_25k.pth": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/craft_mlt_25k.zip",
        "craft_mlt_25k.zip",
    ),
    "latin_g2.pth": (
        "https://github.com/JaidedAI/EasyOCR/releases/download/v1.6.1/latin_g2.zip",
        "latin_g2.zip",
    ),
}

# Tamaños aproximados mínimos en bytes para validación de integridad
TAMANOS_MINIMOS = {
    "craft_mlt_25k.pth": 75 * 1024 * 1024,  # ~83.1 MB real
    "latin_g2.pth": 14 * 1024 * 1024,       # ~15.4 MB real
    "yolov8n.pt": 5 * 1024 * 1024,          # ~6.2 MB real
}


def descargar_con_progreso(url: str, destino_zip: Path) -> None:
    """Descarga un archivo mostrando el progreso en consola."""
    print(f"  Descargando desde: {url}")
    with urllib.request.urlopen(url) as respuesta, open(destino_zip, "wb") as salida:
        total = int(respuesta.info().get("Content-Length", -1))
        descargado = 0
        bloque_tam = 1024 * 1024  # 1 MB

        while True:
            bloque = respuesta.read(bloque_tam)
            if not bloque:
                break
            salida.write(bloque)
            descargado += len(bloque)
            if total > 0:
                porcentaje = (descargado / total) * 100
                mb_desc = descargado / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                print(f"\r  Progreso: {mb_desc:.1f}/{mb_total:.1f} MB ({porcentaje:.1f}%)", end="", flush=True)
            else:
                mb_desc = descargado / (1024 * 1024)
                print(f"\r  Descargado: {mb_desc:.1f} MB", end="", flush=True)
        print()


def preparar_modelos(destino_dir: Path, verify_only: bool = False) -> bool:
    """Aprovisiona y valida los modelos requeridos para el servicio."""
    destino_dir.mkdir(parents=True, exist_ok=True)
    home_easyocr = Path.home() / ".EasyOCR" / "model"
    todos_ok = True

    print(f"Directorio de modelos: {destino_dir.resolve()}")

    # ─── 1. Modelos EasyOCR ───
    for modelo_nombre, (url, zip_nombre) in EASYOCR_URLS.items():
        destino_archivo = destino_dir / modelo_nombre
        min_bytes = TAMANOS_MINIMOS[modelo_nombre]

        if destino_archivo.exists() and destino_archivo.stat().st_size >= min_bytes:
            print(f"  [OK] {modelo_nombre} ({destino_archivo.stat().st_size / (1024*1024):.1f} MB)")
            continue

        # Intentar copiar desde ~/.EasyOCR/model si ya existe localmente
        try:
            origen_home = home_easyocr / modelo_nombre
            if origen_home.exists() and origen_home.stat().st_size >= min_bytes:
                print(f"  Copiando {modelo_nombre} desde caché del usuario ({origen_home})...")
                shutil.copy2(origen_home, destino_archivo)
                print(f"  [OK] {modelo_nombre} sincronizado.")
                continue
        except (PermissionError, OSError):
            pass

        if verify_only:
            print(f"  [FALTA] {modelo_nombre} no está presente en {destino_dir}")
            todos_ok = False
            continue

        # Descarga desde GitHub Releases
        print(f"  Aprovisionando {modelo_nombre}...")
        destino_zip = destino_dir / zip_nombre
        try:
            descargar_con_progreso(url, destino_zip)
            print(f"  Descomprimiendo {zip_nombre}...")
            with zipfile.ZipFile(destino_zip, "r") as zf:
                zf.extractall(destino_dir)
            if destino_zip.exists():
                destino_zip.unlink()
            print(f"  [OK] {modelo_nombre} instalado correctamente.")
        except Exception as e:
            print(f"  [ERROR] Falló la descarga de {modelo_nombre}: {e}")
            todos_ok = False

    # ─── 2. Modelo YOLO ───
    yolo_path = destino_dir.parent / "yolov8n.pt"
    if yolo_path.exists() and yolo_path.stat().st_size >= TAMANOS_MINIMOS["yolov8n.pt"]:
        print(f"  [OK] yolov8n.pt ({yolo_path.stat().st_size / (1024*1024):.1f} MB)")
    else:
        if verify_only:
            print(f"  [FALTA] yolov8n.pt no está presente en {yolo_path}")
            todos_ok = False
        else:
            print("  Aprovisionando YOLOv8n...")
            try:
                from ultralytics import YOLO
                YOLO("yolov8n.pt")
                print("  [OK] yolov8n.pt descargado y verificado.")
            except Exception as e:
                print(f"  [ERROR] Falló la preparación de YOLO: {e}")
                todos_ok = False

    return todos_ok


def main():
    parser = argparse.ArgumentParser(description="Aprovisionador de modelos de IA para VisionCare AI")
    parser.add_argument("--verify-only", action="store_true", help="Solo verificar presencia sin descargar")
    parser.add_argument("--dir", default="models", help="Directorio de destino para modelos OCR (default: models)")
    args = parser.parse_args()

    raiz = Path(__file__).resolve().parent
    destino = Path(args.dir)
    if not destino.is_absolute():
        destino = raiz / destino

    print("═══════════════════════════════════════════════════════════")
    print(" VisionCare AI — Verificación y Aprovisionamiento de Modelos")
    print("═══════════════════════════════════════════════════════════")

    exito = preparar_modelos(destino, verify_only=args.verify_only)

    if exito:
        print("\nTodos los modelos están listos para inferencia offline inmediata.")
        sys.exit(0)
    else:
        print("\nADVERTENCIA: Uno o más modelos no están listos.")
        sys.exit(1)


if __name__ == "__main__":
    main()
