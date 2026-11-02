"""Tests for dslib.multiclass_labels — three-class target for P2P loans."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from dslib.multiclass_labels import (
    ALL_EVIDENCES,
    DEFAULT,
    DEFAULT_EVIDENCES,
    DOUBTFUL,
    NO_DEFAULT,
    build_three_class_target,
    maturity_date,
    scheduled_interest,
)

# Préstamo base: 36 meses, 1000 de principal, cuota 33, emitido en 2015-01,
# último pago en 2017-12 (a tiempo), intereses dentro del calendario.
# Calendario: 33 * 36 - 1000 = 188 de intereses.


def _frame(**overrides):
    row = {
        "Default": 0,
        "issue_d": date(2015, 1, 1),
        "term_months": 36,
        "installment": 33.0,
        "funded_amnt": 1000.0,
        "total_rec_int": 188.0,
        "total_rec_late_fee": 0.0,
        "last_pymnt_d": date(2017, 12, 1),
        "hardship_flag": "N",
        "debt_settlement_flag": "N",
        "loan_status_2018": "Fully Paid",
    }
    row.update(overrides)
    schema = {
        "Default": pl.Int64,
        "issue_d": pl.Date,
        "term_months": pl.Int64,
        "installment": pl.Float64,
        "funded_amnt": pl.Float64,
        "total_rec_int": pl.Float64,
        "total_rec_late_fee": pl.Float64,
        "last_pymnt_d": pl.Date,
        "hardship_flag": pl.String,
        "debt_settlement_flag": pl.String,
        "loan_status_2018": pl.String,
    }
    return pl.DataFrame({k: [v] for k, v in row.items()}, schema=schema)


def _one(evidences=ALL_EVIDENCES, **overrides):
    """Por defecto con todas las evidencias, para probar cada regla por separado."""
    out = build_three_class_target(_frame(**overrides), evidences=evidences)
    return out["target"][0], out["incident_reason"][0], out["n_evidences"][0]


def test_clean_paid_loan_is_no_default():
    assert _one() == (NO_DEFAULT, "none", 0)


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"debt_settlement_flag": "Y"}, "settlement"),
        ({"hardship_flag": "Y"}, "hardship"),
        ({"total_rec_late_fee": 15.0}, "late_fee"),
        ({"loan_status_2018": "Late (31-120 days)"}, "late_2018"),
        ({"loan_status_2018": "Late (16-30 days)"}, "late_2018"),
        ({"last_pymnt_d": date(2018, 3, 1)}, "paid_after_maturity"),
        ({"total_rec_int": 200.0}, "extra_interest"),
    ],
)
def test_each_evidence_makes_a_paid_loan_doubtful(overrides, reason):
    assert _one(**overrides) == (DOUBTFUL, reason, 1)


def test_grace_period_status_is_not_evidence():
    assert _one(loan_status_2018="In Grace Period")[0] == NO_DEFAULT


def test_payment_at_maturity_boundary_is_not_late():
    # Vencimiento 2018-01; con 1 mes de margen el límite es 2018-02.
    assert _one(last_pymnt_d=date(2018, 1, 1))[0] == NO_DEFAULT
    assert _one(last_pymnt_d=date(2018, 2, 1))[0] == NO_DEFAULT
    assert _one(last_pymnt_d=date(2018, 3, 1))[0] == DOUBTFUL


def test_extra_interest_boundary():
    # Tolerancia 1 %: 188 * 1.01 = 189.88.
    assert _one(total_rec_int=189.5)[0] == NO_DEFAULT
    assert _one(total_rec_int=190.0)[0] == DOUBTFUL


def test_custom_tolerances():
    df = _frame(last_pymnt_d=date(2018, 2, 1), total_rec_int=190.0)
    out = build_three_class_target(
        df, maturity_grace_months=0, extra_interest_tol=0.05
    )
    assert out["incident_reason"][0] == "paid_after_maturity"
    assert out["n_evidences"][0] == 1


def test_charged_off_is_default_even_with_evidence():
    target, reason, n = _one(
        Default=1, total_rec_late_fee=30.0, hardship_flag="Y", debt_settlement_flag="Y"
    )
    assert target == DEFAULT
    assert n == 3


def test_reason_priority_and_count():
    target, reason, n = _one(
        debt_settlement_flag="Y", total_rec_late_fee=15.0, total_rec_int=250.0
    )
    assert (target, reason, n) == (DOUBTFUL, "settlement", 3)

    assert _one(total_rec_late_fee=15.0, total_rec_int=250.0)[1] == "late_fee"
    assert _one(loan_status_2018="Late (16-30 days)", total_rec_int=250.0)[1] == "late_2018"


def test_nulls_count_as_no_evidence():
    out = _one(
        total_rec_late_fee=None,
        hardship_flag=None,
        debt_settlement_flag=None,
        loan_status_2018=None,
        last_pymnt_d=None,
        total_rec_int=None,
    )
    assert out == (NO_DEFAULT, "none", 0)


def test_helpers():
    df = pl.DataFrame(
        {"i": [33.0], "n": [36], "f": [1000.0], "d": [date(2015, 1, 1)]}
    )
    out = df.select(
        scheduled_interest(pl.col("i"), pl.col("n"), pl.col("f")).alias("s"),
        maturity_date(pl.col("d"), pl.col("n")).alias("m"),
    )
    assert out["s"][0] == 188.0
    assert out["m"][0] == date(2018, 1, 1)


def test_original_columns_are_kept():
    df = _frame()
    out = build_three_class_target(df)
    assert set(df.columns) <= set(out.columns)
    assert out.height == df.height


def test_missing_column_raises():
    with pytest.raises(ValueError, match="hardship_flag"):
        build_three_class_target(_frame().drop("hardship_flag"))


def test_extra_interest_is_not_used_by_default():
    # Mismo préstamo: con todas las evidencias es dudoso, con las de por defecto no.
    assert "extra_interest" not in DEFAULT_EVIDENCES
    assert _one(total_rec_int=200.0) == (DOUBTFUL, "extra_interest", 1)
    assert _one(evidences=DEFAULT_EVIDENCES, total_rec_int=200.0) == (NO_DEFAULT, "none", 0)
    default_call = build_three_class_target(_frame(total_rec_int=200.0))
    assert default_call["target"][0] == NO_DEFAULT


def test_only_selected_evidences_count():
    one_rule = ["late_fee"]
    assert _one(evidences=one_rule, total_rec_late_fee=15.0)[0] == DOUBTFUL
    assert _one(evidences=one_rule, hardship_flag="Y")[0] == NO_DEFAULT
    # n_evidences solo cuenta las seleccionadas.
    n = _one(evidences=one_rule, total_rec_late_fee=15.0, hardship_flag="Y")[2]
    assert n == 1


def test_unknown_evidence_raises():
    with pytest.raises(ValueError, match="desconocidas"):
        build_three_class_target(_frame(), evidences=["late_fee", "inventada"])
