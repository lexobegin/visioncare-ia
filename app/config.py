"""
═══════════════════════════════════════════════════════════
VisionCare AI — Configuración del servicio IA (FastAPI)

Lee el archivo .env y expone `settings` (singleton tipado).
Equivalente a las configs registerAs() del backend NestJS.

Uso:
    from app.config import settings
    settings.yolo_model
    settings.database_url
═══════════════════════════════════════════════════════════
"""

from functools import lru_cache
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Variables de entorno tipadas (nombres case-insensitive)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignora variables no declaradas
    )

    # ─── Aplicación ───
    app_name: str = "VisionCare AI - Servicio IA"
    node_env: str = "development"
    port: int = 8000
    api_prefix: str = "api/v1"

    # ─── Base de datos PostgreSQL (la misma del backend) ───
    db_host: str = "localhost"
    db_port: int = 5432
    db_username: str = "postgres"
    db_password: str = ""
    db_database: str = "visioncare_db"
    db_ssl: bool = False
    db_echo: bool = True

    # ─── JWT (mismo secret que el backend NestJS) ───
    jwt_secret: str = ""
    jwt_issuer: str = "visioncare-ai"
    jwt_algorithm: str = "HS256"

    # ─── API Key compartida (NestJS ↔ IA) ───
    # Vacía = sin autenticación (SOLO desarrollo local)
    ia_api_key: str = ""

    # ─── Modelo YOLO ───
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.5
    yolo_device: str = "cpu"
    yolo_imgsz: int = 640

    # ─── Límites de imágenes ───
    max_image_size_mb: int = 10

    # ─── CORS ───
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ═══════════════ Propiedades calculadas ═══════════════

    @property
    def es_produccion(self) -> bool:
        return self.node_env.strip().lower() == "production"

    @property
    def database_url(self) -> str:
        """URL de conexión async para SQLAlchemy + asyncpg.

        La contraseña se codifica (quote_plus) para soportar
        caracteres especiales como @, : o / sin romper la URL.
        """
        password = quote_plus(self.db_password)
        ssl = "?ssl=require" if self.db_ssl else ""
        return (
            f"postgresql+asyncpg://{self.db_username}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.db_database}{ssl}"
        )

    @property
    def cors_origins_list(self) -> list[str]:
        """Convierte 'http://a,http://b' → ['http://a', 'http://b']."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def api_prefix_path(self) -> str:
        """Prefijo con barra inicial garantizada ('api/v1' → '/api/v1').

        FastAPI exige que los prefijos de include_router empiecen
        con '/'; normalizamos aquí para tolerar ambas formas en
        el .env (api/v1 o /api/v1).
        """
        prefix = self.api_prefix.strip()
        if not prefix:
            return ""
        return prefix if prefix.startswith("/") else f"/{prefix}"

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    @property
    def api_key_activa(self) -> bool:
        """True si se exige X-API-Key en las peticiones."""
        return len(self.ia_api_key.strip()) > 0


@lru_cache
def get_settings() -> Settings:
    """Factory cacheada (patrón oficial de FastAPI para settings)."""
    return Settings()


# Instancia única usada en toda la app
settings = get_settings()