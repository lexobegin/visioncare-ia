"""
VisionCare AI — Modelos ORM del servicio IA.

Importar los modelos aquí registra sus tablas en Base.metadata
(necesario para create_all en init_db).
"""

from app.models.modelo import ModeloIA
from app.models.registro import RegistroIA

__all__ = ["ModeloIA", "RegistroIA"]