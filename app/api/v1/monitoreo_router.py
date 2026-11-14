"""
═══════════════════════════════════════════════════════════
VisionCare AI — Router de monitoreo IA (Sprint 3, CU40)

Lado LECTURA del monitoreo: consulta agregada de modelos_ia y
registros_ia (la ESCRITURA ya existe en monitoreo_service, que
registra cada ejecución como best-effort).

GET /api/v1/monitoreo/modelos    — catálogo + métricas por modelo
GET /api/v1/monitoreo/registros  — últimas ejecuciones (filtros)
GET /api/v1/monitoreo/resumen    — agregados globales (CU40)

Seguridad: capa 1 (X-API-Key si IA_API_KEY está configurada).
SIN JWT: el monitoreo es operacional (no user-scoped); la web
admin lo consume vía backend NestJS, que lee la misma BD.
═══════════════════════════════════════════════════════════
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verificar_api_key
from app.database import get_db
from app.models import ModeloIA, RegistroIA

# Estados válidos (BD: ck_registro_estado)
ESTADOS_VALIDOS = {"EXITO", "ERROR", "ADVERTENCIA"}

router = APIRouter(
    prefix="/monitoreo",
    tags=["Monitoreo"],
    dependencies=[Depends(verificar_api_key)],
)


def _iso(valor: datetime | None) -> str | None:
    """Datetime → ISO 8601 (o None)."""
    return valor.isoformat() if valor else None


def _float(valor: Decimal | float | None) -> float | None:
    """Numeric de la BD → float para el JSON (o None)."""
    return float(valor) if valor is not None else None


# ══════════════════════════════════════════════════════════
# CATÁLOGO DE MODELOS + MÉTRICAS
# ══════════════════════════════════════════════════════════


@router.get(
    "/modelos",
    summary="CU40 — Modelos IA desplegados con sus métricas de uso",
    description=(
        "Catálogo de modelos_ia (get-or-create de monitoreo_service) "
        "combinado con agregados de registros_ia: total de ejecuciones, "
        "éxitos, errores, tasa de éxito, duración promedio y última "
        "ejecución. Una sola consulta de agregación (sin N+1)."
    ),
    responses={401: {"description": "API Key ausente o inválida"}},
)
async def listar_modelos(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    modelos = (
        (await db.execute(select(ModeloIA).order_by(ModeloIA.nombre)))
        .scalars()
        .all()
    )

    filas = (
        await db.execute(
            select(
                RegistroIA.id_modelo,
                func.count().label("total"),
                func.count()
                .filter(RegistroIA.resultado_estado == "EXITO")
                .label("exitos"),
                func.count()
                .filter(RegistroIA.resultado_estado == "ERROR")
                .label("errores"),
                func.avg(RegistroIA.duracion_ms).label("duracion_promedio"),
                func.max(RegistroIA.fecha_ejecucion).label("ultima_ejecucion"),
            ).group_by(RegistroIA.id_modelo)
        )
    ).all()

    por_modelo = {fila.id_modelo: fila for fila in filas}

    items: list[dict[str, Any]] = []
    for modelo in modelos:
        fila = por_modelo.get(modelo.id_modelo)
        total = int(fila.total) if fila else 0
        exitos = int(fila.exitos) if fila else 0
        errores = int(fila.errores) if fila else 0

        items.append(
            {
                "id_modelo": modelo.id_modelo,
                "nombre": modelo.nombre,
                "version": modelo.version,
                "tipo_modelo": modelo.tipo_modelo,
                "proveedor": modelo.proveedor,
                "estado": modelo.estado,
                "precision_aproximada": _float(modelo.precision_aproximada),
                "fecha_publicacion": _iso(modelo.fecha_publicacion),
                "metricas": {
                    "total_ejecuciones": total,
                    "exitos": exitos,
                    "errores": errores,
                    "tasa_exito_pct": (
                        round(exitos / total * 100, 1) if total else None
                    ),
                    "duracion_promedio_ms": (
                        round(float(fila.duracion_promedio), 1)
                        if fila and fila.duracion_promedio is not None
                        else None
                    ),
                    "ultima_ejecucion": _iso(fila.ultima_ejecucion) if fila else None,
                },
            }
        )

    return {"total": len(items), "modelos": items}


# ══════════════════════════════════════════════════════════
# ÚLTIMAS EJECUCIONES
# ══════════════════════════════════════════════════════════


@router.get(
    "/registros",
    summary="CU40 — Últimas ejecuciones registradas (registros_ia)",
    description=(
        "Bitácora técnica de ejecuciones IA, la más reciente primero. "
        "Filtros combinables: tipo_operacion (DETECCION_OBJETOS, "
        "LECTURA_OCR, RECONOCER_BILLETE...), resultado_estado "
        "(EXITO/ERROR/ADVERTENCIA), id_usuario, ventana de horas."
    ),
    responses={
        400: {"description": "resultado_estado fuera del CHECK de la BD"},
        401: {"description": "API Key ausente o inválida"},
    },
)
async def listar_registros(
    limit: int = Query(default=50, ge=1, le=200, description="Máximo a devolver"),
    tipo_operacion: str | None = Query(default=None, max_length=50),
    resultado_estado: str | None = Query(default=None, max_length=20),
    id_usuario: int | None = Query(default=None, ge=1),
    horas: int | None = Query(
        default=None, ge=1, le=720, description="Solo ejecuciones de las últimas N horas"
    ),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    if resultado_estado and resultado_estado not in ESTADOS_VALIDOS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"resultado_estado inválido ({resultado_estado}). "
                "Valores: EXITO, ERROR, ADVERTENCIA."
            ),
        )

    stmt = (
        select(RegistroIA, ModeloIA.nombre)
        .outerjoin(ModeloIA, RegistroIA.id_modelo == ModeloIA.id_modelo)
        .order_by(RegistroIA.fecha_ejecucion.desc())
        .limit(limit)
    )

    if tipo_operacion:
        stmt = stmt.where(RegistroIA.tipo_operacion == tipo_operacion)
    if resultado_estado:
        stmt = stmt.where(RegistroIA.resultado_estado == resultado_estado)
    if id_usuario:
        stmt = stmt.where(RegistroIA.id_usuario == id_usuario)
    if horas:
        limite = datetime.now(timezone.utc) - timedelta(hours=horas)
        stmt = stmt.where(RegistroIA.fecha_ejecucion >= limite)

    filas = (await db.execute(stmt)).all()

    registros: list[dict[str, Any]] = []
    for fila in filas:
        registro: RegistroIA = fila[0]
        registros.append(
            {
                "id_registro": registro.id_registro,
                "tipo_operacion": registro.tipo_operacion,
                "resultado_estado": registro.resultado_estado,
                "duracion_ms": registro.duracion_ms,
                "confianza": _float(registro.confianza),
                "mensaje_error": registro.mensaje_error,
                "id_modelo": registro.id_modelo,
                "modelo": fila[1],  # nombre (outerjoin)
                "id_usuario": registro.id_usuario,
                "id_captura": registro.id_captura,
                "metadatos": registro.metadatos,
                "fecha_ejecucion": _iso(registro.fecha_ejecucion),
            }
        )

    return {"total": len(registros), "registros": registros}


# ══════════════════════════════════════════════════════════
# RESUMEN GLOBAL (CU40)
# ══════════════════════════════════════════════════════════


@router.get(
    "/resumen",
    summary="CU40 — Resumen agregado del monitoreo IA",
    description=(
        "KPIs del servicio IA: totales históricos, desglose por "
        "estado y por tipo de operación (top 10), actividad de las "
        "últimas 24 horas y modelos activos. Ideal para tarjetas "
        "del panel web (el backend también puede leer la BD directa)."
    ),
    responses={401: {"description": "API Key ausente o inválida"}},
)
async def resumen(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    # ─── Totales históricos ───
    totales = (
        await db.execute(
            select(
                func.count().label("total"),
                func.count()
                .filter(RegistroIA.resultado_estado == "EXITO")
                .label("exitos"),
                func.avg(RegistroIA.duracion_ms).label("duracion_promedio"),
                func.avg(RegistroIA.confianza).label("confianza_promedio"),
            )
        )
    ).one()

    # ─── Por estado ───
    filas_estado = (
        await db.execute(
            select(
                RegistroIA.resultado_estado,
                func.count().label("cantidad"),
            ).group_by(RegistroIA.resultado_estado)
        )
    ).all()

    # ─── Por tipo de operación (top 10) ───
    filas_operacion = (
        await db.execute(
            select(
                RegistroIA.tipo_operacion,
                func.count().label("total"),
                func.count()
                .filter(RegistroIA.resultado_estado == "EXITO")
                .label("exitos"),
                func.avg(RegistroIA.duracion_ms).label("duracion_promedio"),
            )
            .group_by(RegistroIA.tipo_operacion)
            .order_by(desc("total"))
            .limit(10)
        )
    ).all()

    # ─── Últimas 24 horas ───
    desde = datetime.now(timezone.utc) - timedelta(hours=24)
    fila_24h = (
        await db.execute(
            select(
                func.count().label("total"),
                func.count()
                .filter(RegistroIA.resultado_estado == "ERROR")
                .label("errores"),
            ).where(RegistroIA.fecha_ejecucion >= desde)
        )
    ).one()

    # ─── Modelos activos ───
    modelos_activos = (
        await db.execute(
            select(func.count())
            .select_from(ModeloIA)
            .where(ModeloIA.estado == "ACTIVO")
        )
    ).scalar_one()

    total = int(totales.total)
    exitos = int(totales.exitos)

    return {
        "total_ejecuciones": total,
        "tasa_exito_pct": round(exitos / total * 100, 1) if total else None,
        "duracion_promedio_ms": (
            round(float(totales.duracion_promedio), 1)
            if totales.duracion_promedio is not None
            else None
        ),
        "confianza_promedio": _float(totales.confianza_promedio),
        "por_estado": {
            fila.resultado_estado: int(fila.cantidad) for fila in filas_estado
        },
        "por_operacion": [
            {
                "operacion": fila.tipo_operacion,
                "total": int(fila.total),
                "exitos": int(fila.exitos),
                "tasa_exito_pct": (
                    round(int(fila.exitos) / int(fila.total) * 100, 1)
                    if int(fila.total)
                    else None
                ),
                "duracion_promedio_ms": (
                    round(float(fila.duracion_promedio), 1)
                    if fila.duracion_promedio is not None
                    else None
                ),
            }
            for fila in filas_operacion
        ],
        "ultimas_24h": {
            "total": int(fila_24h.total),
            "errores": int(fila_24h.errores),
        },
        "modelos_activos": int(modelos_activos),
        "generado_en": datetime.now(timezone.utc).isoformat(),
    }