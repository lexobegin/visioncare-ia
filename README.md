# VisionCare AI — Servicio de Inteligencia Artificial

Microservicio de visión artificial e inferencia multimodal basado en **FastAPI**, **YOLOv8** (Ultralytics) y **EasyOCR** (PyTorch / CRAFT), diseñado para asistir a personas con discapacidad visual.

---

## 🚀 Inicio Rápido (Entorno Reproducible)

El proyecto incluye un punto de entrada estandarizado (`run.py`) y scripts de arranque cross-platform que configuran automáticamente el entorno de ejecución, evitando errores de concurrencia y reserva de memoria en Windows y Linux.

### 1. Requisitos Previos
- **Python**: 3.10 o 3.12 (64-bit recomendado).
- **Entorno Virtual**: `.venv` configurado con las dependencias instaladas.
- **Base de Datos**: PostgreSQL (`visioncare_db`) compartida con el backend NestJS.

### 2. Instalación de Dependencias
```bash
# Dentro del directorio visioncare-ia
python -m venv .venv

# Activar en Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Activar en Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Variables de Entorno
Copiar `.env.example` como `.env` y ajustar credenciales:
```bash
cp .env.example .env
```

### 4. Arranque del Servicio

#### Opción A: Script Estandarizado (Recomendada en cualquier OS)
```bash
python run.py
```
> Configura automáticamente los hilos de CPU, carga `.env`, activa autoreload en desarrollo y levanta el servidor en `http://0.0.0.0:8000`.

#### Opción B: Scripts de un Clic
- **Windows (PowerShell)**:
  ```powershell
  .\start.ps1
  ```
- **Windows (CMD)**:
  ```cmd
  start.bat
  ```
- **Linux / macOS (Bash)**:
  ```bash
  chmod +x start.sh
  ./start.sh
  ```

#### Opción C: Uvicorn Directo
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
> `app/__init__.py` y `app/core/bootstrap.py` garantizan que las variables de entorno críticas se inicialicen antes de que NumPy o PyTorch carguen sus librerías dinámicas C/C++.

---

## ⚙️ Concurrencia y Optimización de Memoria (DevOps)

El servicio procesa inferencia en CPU mediante hilos (`asyncio.to_thread` / `ThreadPoolExecutor`). Para evitar contención y errores de reserva de memoria en Windows (`OpenBLAS error: Memory allocation still failed after 10 retries, giving up`), el sistema controla de forma determinista las variables de álgebra lineal de bajo nivel:

| Variable | Valor por Defecto | Propósito Operativo |
| :--- | :---: | :--- |
| `OPENBLAS_NUM_THREADS` | `1` | Previene colisiones de reserva `VirtualAlloc` en `libopenblas*.dll` (NumPy). |
| `OMP_NUM_THREADS` | `1` | Limita la contención de OpenMP intra-operación en CPU. |
| `MKL_NUM_THREADS` | `1` | Controla los hilos de Intel MKL (PyTorch CPU). |
| `OCR_MAX_DIMENSION` | `1280` | Acota imágenes de smartphone (12–48 MP) para evitar picos de asignación de >1.2 GB en tensores CRAFT. |
| `OCR_CANVAS_SIZE` | `1280` | Limita el canvas interno de EasyOCR al tamaño óptimo. |

---

## 🧪 Pruebas Automatizadas

El proyecto utiliza `unittest` nativo de Python para pruebas de regresión, rendimiento, robustez de imagen y verificación de arranque limpio:

```bash
# Ejecutar todas las pruebas del servicio de IA
python -m unittest discover -s tests -p "test_*.py" -v
```

### Cobertura de Pruebas:
- `test_startup_resilience.py`: Verificación de bootstrap de entorno, propagación de variables y estabilidad ante concurrencia multi-hilo en OpenBLAS.
- `test_ocr_performance.py`: Robustez ante imágenes pequeñas, fotos típicas de smartphone (12 MP), imágenes extremas (48 MP), payloads corruptos y regresión de memoria.
- `test_producto_resilience.py`: Pruebas de integración para detección de productos y fallback OCR.

---

## 🐳 Recomendación de Infraestructura y Contenerización

Para entornos de **Staging** y **Producción**, se recomienda estandarizar el despliegue mediante contenedores Docker con imagen base ligera:
- **Imagen recomendada**: `python:3.12-slim-bookworm` (o base CUDA si se cuenta con GPU).
- **Multi-etapa (Multi-stage build)**: Para compilar ruedas y separar dependencias de compilación del runtime final.
- **Ventaja**: Elimina cualquier disparidad entre Windows y Linux respecto al asignador de memoria del kernel y simplifica la orquestación junto al backend NestJS y PostgreSQL en Docker Compose o Kubernetes.
