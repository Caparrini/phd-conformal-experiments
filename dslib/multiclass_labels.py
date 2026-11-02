"""Etiquetas de tres clases para préstamos P2P de Lending Club.

Las clases siguen la idea de los tres niveles de IFRS 9 y de los *scorecards*
(bueno / indeterminado / malo). Son análogas, no equivalentes: IFRS 9 asigna la
etapa en cada fecha de balance según el deterioro, y aquí cada préstamo recibe
una sola clase según cómo acabó.

    0 = no default  (~Stage 1) pagado sin evidencia de atraso
    1 = dudoso      (~Stage 2) pagado, pero con evidencia de atraso
    2 = default     (~Stage 3) Charged Off o Default

Lending Club da 15 días de gracia y después cobra recargo. No se mide el umbral
de 30 días de IFRS 9 porque el historial mensual de pagos (PMTHIST) ya no se
distribuye.

Evidencias de atraso en un préstamo pagado, en orden de prioridad:

    settlement           quita negociada
    hardship             plan de dificultades
    late_fee             recargo por retraso cobrado en algún momento
    late_2018            estado Late en la foto de 2018Q4 (luego se pagó)
    paid_after_maturity  último pago más de ``maturity_grace_months`` después
                         del vencimiento
    extra_interest       intereses pagados por encima del calendario

Las cinco primeras son registros directos de Lending Club de un atraso de 16
días o más, o de que el cliente no pudo seguir el calendario. ``extra_interest``
se calcula, no se registra, y **no se usa por defecto**: solo el 20% de los
préstamos que marca con un umbral del 1% tiene un recargo real, el resto
probablemente pagó con pocos días de demora, dentro del periodo de gracia. El
notebook 01 muestra los datos.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

NO_DEFAULT = 0
DOUBTFUL = 1
DEFAULT = 2

CLASS_NAMES = {NO_DEFAULT: "No default", DOUBTFUL: "Dudoso", DEFAULT: "Default"}

ALL_EVIDENCES = [
    "settlement",
    "hardship",
    "late_fee",
    "late_2018",
    "paid_after_maturity",
    "extra_interest",
]
DEFAULT_EVIDENCES = [e for e in ALL_EVIDENCES if e != "extra_interest"]

REQUIRED_COLUMNS = [
    "Default",
    "issue_d",
    "term_months",
    "installment",
    "funded_amnt",
    "total_rec_int",
    "total_rec_late_fee",
    "last_pymnt_d",
    "hardship_flag",
    "debt_settlement_flag",
    "loan_status_2018",
]

LATE_STATUSES = ["Late (16-30 days)", "Late (31-120 days)"]


def scheduled_interest(
    installment: pl.Expr, term_months: pl.Expr, funded_amnt: pl.Expr
) -> pl.Expr:
    """Intereses totales del calendario: cuotas * plazo - principal."""
    return installment * term_months - funded_amnt


def maturity_date(issue_d: pl.Expr, term_months: pl.Expr) -> pl.Expr:
    """Fecha de vencimiento: ``issue_d`` + ``term_months`` meses."""
    return issue_d.dt.offset_by(pl.format("{}mo", term_months))


def evidence_flags(
    maturity_grace_months: int = 1,
    extra_interest_tol: float = 0.01,
) -> dict[str, pl.Expr]:
    """Una expresión booleana por evidencia, con nulos tratados como ``False``.

    Devuelve todas las evidencias de ``ALL_EVIDENCES``, se usen o no.
    """
    maturity = maturity_date(pl.col("issue_d"), pl.col("term_months"))
    sched = scheduled_interest(
        pl.col("installment"), pl.col("term_months"), pl.col("funded_amnt")
    )
    evidence = {
        "settlement": pl.col("debt_settlement_flag") == "Y",
        "hardship": pl.col("hardship_flag") == "Y",
        "late_fee": pl.col("total_rec_late_fee") > 0,
        "late_2018": pl.col("loan_status_2018").is_in(LATE_STATUSES),
        "paid_after_maturity": pl.col("last_pymnt_d")
        > maturity.dt.offset_by(f"{maturity_grace_months}mo"),
        "extra_interest": pl.col("total_rec_int") > sched * (1 + extra_interest_tol),
    }
    return {name: expr.fill_null(False) for name, expr in evidence.items()}


def build_three_class_target(
    df: pl.DataFrame,
    maturity_grace_months: int = 1,
    extra_interest_tol: float = 0.01,
    evidences: Sequence[str] = DEFAULT_EVIDENCES,
) -> pl.DataFrame:
    """Añade ``target`` (0/1/2), ``incident_reason`` y ``n_evidences``.

    Solo cuentan las evidencias de ``evidences``. ``incident_reason`` es la
    primera de ellas (según ``ALL_EVIDENCES``) que cumple un préstamo no default,
    o ``"none"``. ``n_evidences`` cuenta cuántas cumple. Los nulos cuentan como
    "sin evidencia".

    Un préstamo con ``Default == 1`` es default aunque tenga evidencias.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    unknown = [e for e in evidences if e not in ALL_EVIDENCES]
    if unknown:
        raise ValueError(f"Evidencias desconocidas: {unknown}. Disponibles: {ALL_EVIDENCES}")

    flags = evidence_flags(maturity_grace_months, extra_interest_tol)
    used = [e for e in ALL_EVIDENCES if e in set(evidences)]

    reason = pl.lit("none")
    for name in reversed(used):
        reason = pl.when(flags[name]).then(pl.lit(name)).otherwise(reason)

    n_evidences = pl.sum_horizontal([flags[name].cast(pl.Int8) for name in used])

    return df.with_columns(
        reason.alias("incident_reason"),
        n_evidences.alias("n_evidences"),
    ).with_columns(
        pl.when(pl.col("Default") == 1)
        .then(pl.lit(DEFAULT))
        .when(pl.col("n_evidences") > 0)
        .then(pl.lit(DOUBTFUL))
        .otherwise(pl.lit(NO_DEFAULT))
        .cast(pl.Int8)
        .alias("target")
    )
