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
    | **Dry Bean** | Multiclass classification | Koklu & Ozkan (2020), UCI ML Repository |
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
    ## Dry Bean Dataset
    """)
    return


@app.cell
def _():
    from ucimlrepo import fetch_ucirepo

    beans_raw = fetch_ucirepo(id=602)
    beans_raw.data.features.shape
    return (beans_raw,)


@app.cell
def _(beans_raw, pl):
    df_beans = pl.from_pandas(beans_raw.data.features)

    # Lowercase column names and add target as bean_type
    df_beans = df_beans.rename({c: c.lower() for c in df_beans.columns})
    df_beans = df_beans.with_columns(
        pl.Series("bean_type", beans_raw.data.targets.iloc[:, 0])
    )

    df_beans.shape
    return (df_beans,)


@app.cell
def _(df_beans):
    df_beans.head()
    return


@app.cell
def _(df_beans):
    df_beans.describe()
    return


@app.cell
def _(CLEANED_DIR, df_beans):
    beans_path = CLEANED_DIR / "dry_beans.parquet"
    df_beans.write_parquet(beans_path)
    beans_path
    return (beans_path,)


@app.cell
def _(mo):
    mo.md("""
    ## Summary
    """)
    return


@app.cell
def _(adult_path, beans_path, credit_path, df_adult, df_beans, df_credit, mo):
    def _size_mb(path):
        return path.stat().st_size / 1024 / 1024

    summary = f"""
    | Dataset | Rows | Features | Target | Task | Size |
    |---------|------|----------|--------|------|------|
    | Adult | {df_adult.shape[0]:,} | {df_adult.shape[1] - 1} | `income` | Binary | {_size_mb(adult_path):.1f} MB |
    | Credit Card Default | {df_credit.shape[0]:,} | {df_credit.shape[1] - 1} | `default` | Binary | {_size_mb(credit_path):.1f} MB |
    | Dry Bean | {df_beans.shape[0]:,} | {df_beans.shape[1] - 1} | `bean_type` | Multiclass (7) | {_size_mb(beans_path):.1f} MB |
    """
    mo.md(summary)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
