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
    # Lending Club — Data Preparation

    This notebook downloads the raw Lending Club dataset from Zenodo
    and prepares a cleaned Parquet file for further analysis.

    **Source**: [Zenodo Record 11295916](https://zenodo.org/records/11295916)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Download Raw Data

    The dataset is downloaded from Zenodo if not already present locally.
    """)
    return


@app.cell
def _(Path):
    from dslib.data_utils import download_if_missing

    EXPERIMENT_DIR = Path(__file__).parent
    RAW_DIR = EXPERIMENT_DIR / "data" / "raw"

    csv_path = download_if_missing(
        url="https://zenodo.org/records/11295916/files/LC_loans_granting_model_dataset.csv?download=1",
        dest=RAW_DIR / "LC_loans_granting_model_dataset.csv",
    )
    csv_path
    return EXPERIMENT_DIR, csv_path


@app.cell
def _(mo):
    mo.md("""
    ## 2. Load and Explore

    Quick look at the raw dataset: shape, columns, types, and null counts.
    """)
    return


@app.cell
def _(csv_path, pl):
    df = pl.read_csv(csv_path)
    df.shape
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df.null_count()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Select and Clean

    We keep 10 columns relevant for conformal prediction experiments
    and convert `issue_d` from string to date.

    Dropped columns:
    - `id`: row identifier, not a feature
    - `experience_c`: 99.998% single value, no discriminative power
    - `zip_code`, `title`, `desc`: only columns with nulls, not needed
    """)
    return


@app.cell
def _(df, pl):
    selected_columns = [
        "fico_n", "dti_n", "loan_amnt", "revenue",
        "emp_length", "purpose", "home_ownership_n",
        "addr_state", "issue_d", "Default",
    ]

    df_clean = (
        df
        .select(selected_columns)
        .with_columns(
            pl.col("issue_d").str.to_date("%b-%Y")
        )
    )
    df_clean.schema
    return (df_clean,)


@app.cell
def _(df_clean):
    df_clean.head()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Save as Parquet
    """)
    return


@app.cell
def _(EXPERIMENT_DIR, df_clean):
    CLEANED_DIR = EXPERIMENT_DIR / "data" / "cleaned"
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_path = CLEANED_DIR / "lending_club.parquet"
    df_clean.write_parquet(cleaned_path)
    cleaned_path
    return (cleaned_path,)


@app.cell
def _(cleaned_path, csv_path, df_clean, mo):
    csv_size = csv_path.stat().st_size / 1024 / 1024
    parquet_size = cleaned_path.stat().st_size / 1024 / 1024
    ratio = csv_size / parquet_size

    mo.md(f"""
    ## Summary

    | | Value |
    |---|---|
    | **Rows** | {df_clean.shape[0]:,} |
    | **Columns** | {df_clean.shape[1]} |
    | **Date range** | {df_clean["issue_d"].min()} → {df_clean["issue_d"].max()} |
    | **Default rate** | {df_clean["Default"].mean():.1%} |
    | **CSV size** | {csv_size:.1f} MB |
    | **Parquet size** | {parquet_size:.1f} MB |
    | **Compression** | {ratio:.0f}x smaller |
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
