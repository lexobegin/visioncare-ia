"""
═══════════════════════════════════════════════════════════
VisionCare AI — Conexión a PostgreSQL (SQLAlchemy 2.0 async)

- Base: clase declarativa raíz de los modelos ORM.
- engine: pool de conexiones async (asyncpg).
- get_db(): dependencia FastAPI — una sesión por petición
  con commit/rollback automático.

El servicio IA escribe SOLO en sus tablas de monitoreo
(modelos_ia, registros_ia); las capturas/detecciones las
persiste el backend NestJS.
═══════════════════════════════════════════════════════════
"""

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    """Base declarativa para todos los modelos ORM del servicio."""


# ─── Motor async (pool compartido en toda la app) ───
engine = create_async_engine(
    settings.database_url,
    echo=settings.db_echo,      # loguea SQL (solo desarrollo)
    pool_size=5,                # conexiones permanentes
    max_overflow=10,            # conexiones extra bajo demanda
    pool_pre_ping=True,         # valida conexión antes de usarla
    pool_recycle=1800,          # recicla conexiones cada 30 min
)

# ─── Fábrica de sesiones ───
session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,     # objetos usables tras commit
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependencia FastAPI: entrega una sesión por petición.

    Uso en routers:
        @router.get("/ruta")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            ...

    Commit automático al terminar sin errores; rollback
    ante cualquier excepción.
    """
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def verificar_conexion() -> bool:
    """Health-check de la BD (usado por /health)."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def cerrar_conexiones() -> None:
    """Cierra el pool ordenadamente (shutdown de la app)."""
    await engine.dispose()