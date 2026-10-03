"""
═══════════════════════════════════════════════════════════
VisionCare AI — Inicialización de tablas propias del servicio.

Crea modelos_ia y registros_ia SI NO EXISTEN (checkfirst).
- Dev:      las crea automáticamente al arrancar (main.py).
- Prod:     el script SQL del modelo completo ya las creó con
            FKs adicionales → checkfirst no las toca.

Ejecución manual:
    python -m app.init_db
═══════════════════════════════════════════════════════════
"""

import asyncio

from sqlalchemy import text

from app.database import Base, engine
from app.models import ModeloIA, RegistroIA  # noqa: F401 — registran metadata


async def init_db() -> None:
    """Crea las tablas del servicio (si faltan) y verifica."""
    async with engine.begin() as conn:
        # create_all solo crea lo que falta; JAMÁS altera tablas existentes
        await conn.run_sync(Base.metadata.create_all, checkfirst=True)

        resultado = await conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' "
                "AND table_name IN ('modelos_ia', 'registros_ia') "
                "ORDER BY table_name"
            )
        )
        tablas = [fila[0] for fila in resultado]
        print(f"✅ Tablas del servicio IA verificadas/creadas: {tablas}")


async def cerrar() -> None:
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(init_db())
    asyncio.run(cerrar())