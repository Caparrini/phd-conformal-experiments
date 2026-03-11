import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import polars as pl

    return Path, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Data Preparation — Conformal Correctness Margin

    This notebook downloads and cleans 3 UCI datasets for studying
    FCOD and SHAP values over conformal correctness margins.

    | Dataset | Task | Source |
    |---------|------|--------|
    | **Adult** | Binary classification | Kohavi & Becker (1996), UCI ML Repository |
    | **Credit Card Default** | Binary classification | Yeh & Lien (2009), UCI ML Repository |
    | **Wine Quality** | Multiclass classification | Cortez et al. (2009), UCI ML Repository |
    """)
    return


@app.cell
def _(Path):
    EXPERIMENT_DIR = Path(__file__).parent
    CLEANED_DIR = EXPERIMENT_DIR / "data" / "cleaned"
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    return (CLEANED_DIR,)


@app.cell
def _(mo):
    mo.md("""
    ## Adult Dataset
    """)
    return


@app.cell
def _():
    from sklearn.datasets import fetch_openml

    adult_raw = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    adult_raw.data.shape
    return (adult_raw,)


@app.cell
def _(adult_raw, pl):
    df_adult = pl.from_pandas(adult_raw.data)
    target_adult = adult_raw.target

    # Encode target: '>50K' -> 1, '<=50K' -> 0
    df_adult = df_adult.with_columns(
        pl.Series("income", (target_adult == ">50K").astype(int))
    )

    # Drop fnlwgt (sampling weight) and education (redundant with education-num)
    # Drop rows with missing values
    df_adult = (
        df_adult.drop(["fnlwgt", "education"])
        .drop_nulls()
    )

    df_adult.shape
    return (df_adult,)


@app.cell
def _(df_adult):
    df_adult.head()
    return


@app.cell
def _(df_adult):
    df_adult.describe()
    return


@app.cell
def _(CLEANED_DIR, df_adult):
    adult_path = CLEANED_DIR / "adult.parquet"
    df_adult.write_parquet(adult_path)
    adult_path
    return (adult_path,)


@app.cell
def _(mo):
    mo.md("""
    ## Credit Card Default Dataset
    """)
    return


@app.cell
def _():
    from sklearn.datasets import fetch_openml as _fetch

    credit_raw = _fetch(data_id=42477, as_frame=True, parser="auto")
    credit_raw.data.shape
    return (credit_raw,)


@app.cell
def _(credit_raw, pl):
    CREDIT_COLUMNS = {
        "x1": "limit_bal",
        "x2": "sex",
        "x3": "education",
        "x4": "marriage",
        "x5": "age",
        "x6": "pay_0",
        "x7": "pay_2",
        "x8": "pay_3",
        "x9": "pay_4",
        "x10": "pay_5",
        "x11": "pay_6",
        "x12": "bill_amt1",
        "x13": "bill_amt2",
        "x14": "bill_amt3",
        "x15": "bill_amt4",
        "x16": "bill_amt5",
        "x17": "bill_amt6",
        "x18": "pay_amt1",
        "x19": "pay_amt2",
        "x20": "pay_amt3",
        "x21": "pay_amt4",
        "x22": "pay_amt5",
        "x23": "pay_amt6",
    }

    df_credit = pl.from_pandas(credit_raw.data).rename(CREDIT_COLUMNS)

    # Remap integer-encoded categoricals to human-readable string labels.
    # Cast to String first so replace operates on str keys, avoiding i64 dtype coercion.
    _PAY_OLD = [str(v) for v in [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]]
    _PAY_NEW = ["No consumption", "Paid full", "Min. paid",
                "1m late", "2m late", "3m late", "4m late",
                "5m late", "6m late", "7m late", "8m late"]

    df_credit = df_credit.with_columns([
        pl.col("sex").cast(pl.String).replace({"1": "Male", "2": "Female"}),
        pl.col("education").cast(pl.String).replace(
            {"0": "Unknown", "1": "Graduate", "2": "University",
             "3": "High School", "4": "Others", "5": "Unknown", "6": "Unknown"}
        ),
        pl.col("marriage").cast(pl.String).replace(
            {"0": "Others", "1": "Married", "2": "Single", "3": "Others"}
        ),
        *[pl.col(f"pay_{m}").cast(pl.String).replace(dict(zip(_PAY_OLD, _PAY_NEW)))
          for m in ["0", "2", "3", "4", "5", "6"]],
    ])

    # Encode target: categorical '0'/'1' -> int 0/1
    df_credit = df_credit.with_columns(
        pl.Series("default", credit_raw.target.astype(int))
    )

    df_credit.shape
    return (df_credit,)


@app.cell
def _(df_credit):
    df_credit.head()
    return


@app.cell
def _(df_credit):
    df_credit.describe()
    return


@app.cell
def _(CLEANED_DIR, df_credit):
    credit_path = CLEANED_DIR / "credit_card_default.parquet"
    df_credit.write_parquet(credit_path)
    credit_path
    return (credit_path,)


@app.cell
def _(mo):
    mo.md("""
    ## Wine Quality Dataset
    """)
    return


@app.cell
def _():
    from sklearn.datasets import fetch_openml as _fetch_wine

    # OpenML 40691 = winequality-red (1599 rows, correct column names)
    # OpenML 40498 = winequality-white (4898 rows, generic V1-V11 column names)
    wine_red_raw = _fetch_wine(data_id=40691, as_frame=True, parser="auto")
    wine_white_raw = _fetch_wine(data_id=40498, as_frame=True, parser="auto")
    (wine_red_raw.data.shape, wine_white_raw.data.shape)
    return wine_red_raw, wine_white_raw


@app.cell
def _(pl, wine_red_raw, wine_white_raw):
    import pandas as pd

    # Wine feature names in UCI order (shared by red and white)
    _WINE_COLS = [
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
        "ph", "sulphates", "alcohol",
    ]

    def _clean(df):
        return {c: c.strip().lower().replace(" ", "_") for c in df.columns}

    red = wine_red_raw.data.copy().rename(columns=_clean(wine_red_raw.data))
    red["wine_type"] = "red"
    red["quality"] = wine_red_raw.target.astype(int)

    # White wine has generic V1-V11 column names → rename to match red
    # OpenML 40498 encodes quality as 1-7 (shifted: actual UCI quality = openml + 2)
    _white_rename = {f"V{i+1}": name for i, name in enumerate(_WINE_COLS)}
    white = wine_white_raw.data.copy().rename(columns=_white_rename)
    white["wine_type"] = "white"
    white["quality"] = wine_white_raw.target.astype(int) + 2

    combined = pd.concat([red, white], ignore_index=True)
    combined["quality_class"] = combined["quality"].map(
        lambda q: "low" if q <= 5 else ("high" if q >= 7 else "medium")
    )
    combined = combined.drop(columns=["quality"])

    df_wine = pl.from_pandas(combined)
    df_wine.shape
    return (df_wine,)


@app.cell
def _(df_wine):
    df_wine.head()
    return


@app.cell
def _(df_wine):
    df_wine.describe()
    return


@app.cell
def _(CLEANED_DIR, df_wine):
    wine_path = CLEANED_DIR / "wine_quality.parquet"
    df_wine.write_parquet(wine_path)
    wine_path
    return (wine_path,)


@app.cell
def _(mo):
    mo.md("""
    ## Summary
    """)
    return


@app.cell
def _(adult_path, credit_path, wine_path, df_adult, df_credit, df_wine, mo):
    def _size_mb(path):
        return path.stat().st_size / 1024 / 1024

    summary = f"""
    | Dataset | Rows | Features | Target | Task | Size |
    |---------|------|----------|--------|------|------|
    | Adult | {df_adult.shape[0]:,} | {df_adult.shape[1] - 1} | `income` | Binary | {_size_mb(adult_path):.1f} MB |
    | Credit Card Default | {df_credit.shape[0]:,} | {df_credit.shape[1] - 1} | `default` | Binary | {_size_mb(credit_path):.1f} MB |
    | Wine Quality | {df_wine.shape[0]:,} | {df_wine.shape[1] - 1} | `quality_class` | Multiclass (3) | {_size_mb(wine_path):.1f} MB |
    """
    mo.md(summary)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
