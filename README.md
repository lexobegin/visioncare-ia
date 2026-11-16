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

### 3. Aprovisionamiento de Modelos (Pesos de Inferencia)
Para garantizar arranque inmediato y evitar *timeouts* en la primera petición de la app móvil, ejecute el gestor automatizado de modelos:
```bash
# Descarga y verifica los modelos necesarios (YOLOv8n + EasyOCR CRAFT & Latin)
python prepare_models.py

# Verificación de integridad sin re-descargar
python prepare_models.py --verify-only
```
> Los modelos se almacenan localmente en el directorio `models/` (excluido en `.gitignore` para no contaminar el repositorio con binarios pesados). Si ya existen en la caché del usuario (`~/.EasyOCR/model`), el script los sincroniza automáticamente sin descargar de internet.

### 4. Variables de Entorno
Copiar `.env.example` como `.env` y ajustar credenciales:
```bash
cp .env.example .env
```

### 5. Arranque del Servicio

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

## 🩺 Chequeos de Salud y Disponibilidad (Readiness Probes)

El microservicio expone dos endpoints de monitoreo para orquestadores (Kubernetes, Docker Compose) y el backend NestJS:

1. **Liveness / Salud General (`GET /api/v1/health` o `GET /health`)**:
   - Verifica si el proceso FastAPI está vivo y el estado de la conexión a PostgreSQL.
   - Si la base de datos no está disponible, reporta `status="degradado"` (el servicio sigue procesando inferencias, pero el monitoreo queda en modo degradado).

2. **Readiness Probe (`GET /api/v1/health/ready` o `GET /health/ready`)**:
   - **HTTP 200 (Listo)**: Devuelve `{"status": "ready"}` cuando tanto YOLO como EasyOCR están instanciados y precargados en memoria RAM.
   - **HTTP 503 (No disponible / Fail-Fast)**: Si algún modelo falta o falló en inicializarse, devuelve `{"status": "not_ready", "mensaje": "..."}` de forma inmediata (< 5 ms).
   - **Recomendación para NestJS / Gateway**: Consultar este endpoint antes de enrutar tráfico de inferencia, o en el arranque del orquestador, para garantizar que la primera petición no sufra *cold-start* ni sobrepase el timeout configurado (`IA_SERVICE_TIMEOUT_MS=30000`).

---

## 🧪 Pruebas Automatizadas

El proyecto utiliza `unittest` nativo de Python para pruebas de regresión, rendimiento, robustez de imagen, preparación de modelos y verificación de arranque:

```bash
# Ejecutar todas las pruebas del servicio de IA
python -m unittest discover -s tests -p "test_*.py" -v
```

### Cobertura de Pruebas:
- `test_model_readiness.py`: Verificación de aprovisionamiento de modelos (`prepare_models.py --verify-only`), respuesta 200/503 en `/health/ready`, y comportamiento fail-fast ante ausencia de pesos.
- `test_startup_resilience.py`: Verificación de bootstrap de entorno, propagación de variables y estabilidad ante concurrencia multi-hilo en OpenBLAS.
- `test_ocr_performance.py`: Robustez ante imágenes pequeñas, fotos típicas de smartphone (12 MP), imágenes extremas (48 MP), payloads corruptos y regresión de memoria.
- `test_producto_resilience.py`: Pruebas de integración para detección de productos y fallback OCR.

---

## 🐳 Recomendación de Infraestructura y Contenerización

Para entornos de **Staging** y **Producción**, se recomienda estandarizar el despliegue mediante Docker precargando los artefactos:

1. **Pre-bake de Modelos en la Imagen Docker**:
   Ejecutar `RUN python prepare_models.py` durante el `docker build`.
   ```dockerfile
   # Ejemplo de Dockerfile (fragmento recomendado)
   FROM python:3.12-slim-bookworm AS runtime
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   # Pre-descarga de pesos en build time (garantiza arranque 100% offline en < 4s)
   RUN python prepare_models.py
   EXPOSE 8000
   CMD ["python", "run.py"]
   ```
2. **Ventajas Operativas**:
   - **Arranque en frío nulo**: El contenedor arranca en < 4 segundos sin descargas en runtime.
   - **Despliegues deterministas**: El contenedor no depende de la conectividad externa a GitHub Releases para funcionar.
   - **Aislamiento total**: Evita problemas de permisos en directorios de usuario del host (`~/.EasyOCR/model`).
