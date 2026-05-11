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
    # copa2026 — Hotel Booking Demand (H2, Lisboa)

    This notebook downloads the Hotel Booking Demand dataset and prepares
    a cleaned Parquet file with the H2 (City Hotel, Lisboa) subset.

    **Source**: [TidyTuesday 2020-02-11](https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-02-11)
    — based on [Antonio, Almeida & Nunes (2019)](https://www.sciencedirect.com/science/article/pii/S2352340918315191), *Data in Brief*.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Download Raw Data

    The combined H1+H2 CSV is downloaded from TidyTuesday if not already present.
    """)
    return


@app.cell
def _(Path):
    from dslib.data_utils import download_if_missing

    EXPERIMENT_DIR = Path(__file__).parent
    RAW_DIR = EXPERIMENT_DIR / "data" / "raw"

    csv_path = download_if_missing(
        url="https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv",
        dest=RAW_DIR / "hotels.csv",
        sha256="7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06",
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
    df = pl.read_csv(csv_path, null_values=["NA"])
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
    ## 3. Filter and Select

    We keep only **H2 (City Hotel, Lisboa)** and select 8 features + 1 target.

    - `stays_total_nights` is derived as `stays_in_week_nights + stays_in_weekend_nights`.
    - `hotel`, `stays_in_week_nights`, and `stays_in_weekend_nights` are dropped after use.

    Dropped features: row identifiers, date components, agent/company codes,
    columns with near-constant distributions, and other high-cardinality fields.
    """)
    return


@app.cell
def _(df, pl):
    df_clean = (
        df
        .filter(pl.col("hotel") == "City Hotel")
        .with_columns(
            (pl.col("stays_in_week_nights") + pl.col("stays_in_weekend_nights")).alias("stays_total_nights")
        )
        .select([
            "is_canceled",
            "lead_time",
            "arrival_date_week_number",
            "stays_total_nights",
            "adr",
            "market_segment",
            "customer_type",
            "deposit_type",
            "is_repeated_guest",
        ])
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

    cleaned_path = CLEANED_DIR / "copa2026.parquet"
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
    | **Rows (H2)** | {df_clean.shape[0]:,} |
    | **Columns** | {df_clean.shape[1]} |
    | **Cancellation rate** | {df_clean["is_canceled"].mean():.1%} |
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
