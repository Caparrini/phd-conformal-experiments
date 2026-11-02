import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from dslib.experiment_io import load_cfg

    pl.Config.set_tbl_rows(30)  # que ninguna tabla salga recortada con "…"

    EXPERIMENT_DIR = Path(__file__).parent
    cfg = load_cfg(EXPERIMENT_DIR, "funcas")
    return EXPERIMENT_DIR, cfg, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # funcas: préstamos P2P con tres clases

    Partimos de los préstamos de Lending Club. En globafa cada préstamo es
    *default* o *no default*. Aquí queremos tres clases, con el mismo esquema
    de tres niveles que usa IFRS 9 para provisionar riesgo de crédito:

    | Clase | Análogo a | Regla |
    |---|---|---|
    | 0 · No default | Stage 1 | Pagado sin evidencia de atraso |
    | 1 · Dudoso | Stage 2 | Pagado, pero con evidencia de atraso |
    | 2 · Default | Stage 3 | *Charged Off* o *Default* |

    Las etapas de IFRS 9 se asignan en cada fecha de balance según el
    deterioro. Aquí cada préstamo recibe una sola clase según cómo acabó, así
    que la analogía es de idea, no de definición.

    Lending Club da 15 días de gracia y después cobra un recargo. Las evidencias
    de dudoso que usamos son registros directos de Lending Club: recargo por
    retraso, estado *Late* en 2018 y último pago tardío (atraso de 16 días o
    más), y quita o plan de dificultades (el cliente no pudo seguir el
    calendario). IFRS 9 usa 30 días como umbral de respaldo, pero el historial
    mensual de pagos de Lending Club ya no se distribuye y no podemos medir los
    30 exactos. Probamos una evidencia más, calculada y no registrada, y la
    descartamos (sección 6).

    No usamos el estado *In Grace Period* (1 a 15 días), ni el FICO de la
    última consulta de crédito: mide cómo está el cliente con todas sus deudas,
    no cómo pagó este préstamo.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Por qué no basta con los datos de globafa

    El CSV de Zenodo solo trae préstamos que ya estaban cerrados a finales de
    2018. De las emisiones de 2017 y 2018 solo aparecen los que se cerraron
    pronto, por pago anticipado o por pérdida rápida. Los que seguían abiertos
    o atrasados en esa fecha no están, y son justo los dudosos.

    La solución es usar una foto posterior de Lending Club, con el estado de
    cada préstamo en el tercer trimestre de 2020. Los préstamos a 36 meses
    emitidos hasta principios de 2017 ya han vencido en esa fecha, así que de
    casi todos vemos la vida completa. La sección 4 explica dónde ponemos el
    corte y por qué.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Descarga y lectura

    Descargamos la foto de 2020 con `kagglehub`. Hace falta un token de Kaggle
    en `~/.kaggle/kaggle.json` o las variables `KAGGLE_USERNAME` y
    `KAGGLE_KEY`. Solo leemos las columnas que necesitamos.
    """)
    return


@app.cell
def _(cfg, pl):
    import kagglehub
    from pathlib import Path as _Path

    dir_2020 = _Path(kagglehub.dataset_download(cfg.kaggle.snapshot_2020))
    file_2020 = next(dir_2020.glob("Loan_status_2007-2020Q3*"))

    cols_2020 = [
        "id", "loan_status", "term", "issue_d", "installment", "funded_amnt",
        "total_rec_int", "total_rec_late_fee", "last_pymnt_d", "hardship_flag",
        "debt_settlement_flag", "fico_range_low", "fico_range_high", "dti",
        "annual_inc", "annual_inc_joint", "application_type", "loan_amnt",
        "emp_length", "purpose", "home_ownership", "addr_state",
    ]
    # Todo como texto y se convierte después: el fichero trae alguna fila rota.
    raw_2020 = pl.read_csv(file_2020, columns=cols_2020, infer_schema=False)

    float_cols = [
        "installment", "funded_amnt", "total_rec_int", "total_rec_late_fee",
        "fico_range_low", "fico_range_high", "dti", "annual_inc",
        "annual_inc_joint", "loan_amnt",
    ]
    loans_2020 = (
        raw_2020.with_columns(
            pl.col("id").cast(pl.Int64, strict=False),
            pl.col("term").str.extract(r"(\d+)").cast(pl.Int64).alias("term_months"),
            pl.col("issue_d").str.to_date("%b-%Y", strict=False),
            pl.col("last_pymnt_d").str.to_date("%b-%Y", strict=False),
            *[pl.col(c).cast(pl.Float64, strict=False) for c in float_cols],
        )
        .filter(pl.col("id").is_not_null())
    )
    loans_2020.shape
    return (loans_2020,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Variables

    Las variables del modelo son las mismas que en globafa. Zenodo las trae ya
    procesadas, así que las reconstruimos desde la foto de 2020:

    - `fico_n` es la media de `fico_range_low` y `fico_range_high`.
    - `revenue` es `annual_inc`, salvo en solicitudes conjuntas, donde Zenodo
      usa `annual_inc_joint`.
    - `home_ownership_n` agrupa `ANY` y `NONE` en `OTHER`.
    - La antigüedad laboral que falta se marca `NI`, como en Zenodo.
    - `dti_n` es `dti` tal cual.
    """)
    return


@app.cell
def _(loans_2020, pl):
    features_2020 = loans_2020.with_columns(
        ((pl.col("fico_range_low") + pl.col("fico_range_high")) / 2).alias("fico_n"),
        pl.col("dti").alias("dti_n"),
        pl.when(
            (pl.col("application_type") == "Joint App")
            & pl.col("annual_inc_joint").is_not_null()
        )
        .then(pl.col("annual_inc_joint"))
        .otherwise(pl.col("annual_inc"))
        .alias("revenue"),
        pl.col("emp_length").fill_null("NI"),
        pl.when(pl.col("home_ownership").is_in(["ANY", "NONE"]))
        .then(pl.lit("OTHER"))
        .otherwise(pl.col("home_ownership"))
        .alias("home_ownership_n"),
    )
    features_2020.select("id", "fico_n", "dti_n", "revenue", "emp_length", "home_ownership_n").head()
    return (features_2020,)


@app.cell
def _(mo):
    mo.md("""
    ### Comprobación contra Zenodo

    Antes de fiarnos de las variables reconstruidas, las comparamos con las de
    Zenodo en los préstamos que están en los dos ficheros. Exigimos una
    coincidencia de al menos el 99% en cada una.
    """)
    return


@app.cell
def _(EXPERIMENT_DIR, features_2020, mo, pl):
    from dslib.data_utils import download_if_missing

    zenodo_path = download_if_missing(
        url="https://zenodo.org/records/11295916/files/LC_loans_granting_model_dataset.csv?download=1",
        dest=EXPERIMENT_DIR / "data" / "raw" / "LC_loans_granting_model_dataset.csv",
    )
    zenodo = pl.read_csv(
        zenodo_path,
        columns=[
            "id", "fico_n", "dti_n", "revenue", "loan_amnt", "emp_length",
            "purpose", "home_ownership_n", "addr_state", "Default",
        ],
    )
    common = zenodo.join(features_2020, on="id", suffix="_k")

    checks = {
        "fico_n": (pl.col("fico_n") == pl.col("fico_n_k")),
        "dti_n": ((pl.col("dti_n") - pl.col("dti_n_k")).abs() < 1e-6),
        "revenue": ((pl.col("revenue") - pl.col("revenue_k")).abs() < 1e-6),
        "loan_amnt": (pl.col("loan_amnt") == pl.col("loan_amnt_k")),
        "emp_length": (pl.col("emp_length") == pl.col("emp_length_k")),
        "purpose": (pl.col("purpose") == pl.col("purpose_k")),
        "home_ownership_n": (pl.col("home_ownership_n") == pl.col("home_ownership_n_k")),
        "addr_state": (pl.col("addr_state") == pl.col("addr_state_k")),
    }
    match = {
        name: float(common.select(expr.fill_null(False).mean()).item())
        for name, expr in checks.items()
    }
    assert all(v >= 0.99 for v in match.values()), match

    _rows = "\n".join(f"| `{k}` | {v:.2%} |" for k, v in match.items())
    mo.md(
        f"""
    {common.height:,} préstamos en común.

    | Variable | Coincide con Zenodo |
    |---|---|
    {_rows}

    `dti_n` coincide en el {match["dti_n"]:.2%} de los préstamos. No hemos
    investigado por qué difiere el resto.
    """
    )
    return


@app.cell
def _(cfg, mo):
    mo.md(f"""
    ## 4. Población

    Nos quedamos con los préstamos a {cfg.population.term_months} meses emitidos
    hasta {cfg.population.issued_until} y con estado final conocido. Quedan
    fuera los de 60 meses, que son otro producto y no han vencido todos en 2020,
    y los que en la foto siguen abiertos (*Current*, *Late* o *In Grace
    Period*).

    Los *Does not meet the credit policy. Status: ...* sí entran: son préstamos
    cerrados cuyo estado final (*Fully Paid* o *Charged Off*) va dentro del
    texto. Les quitamos el prefijo.

    ### Por qué cortamos en {cfg.population.issued_until}

    La foto es del tercer trimestre de 2020. Un préstamo a 36 meses emitido a
    mitad de 2017 vence justo entonces y muchos siguen abiertos. Los que ya están
    cerrados son los que acabaron pronto o mal, y eso distorsiona la mezcla de
    clases de esos meses. La tabla lo muestra mes a mes.
    """)
    return


@app.cell
def _(cfg, features_2020, pl):
    _closed = ["Fully Paid", "Charged Off", "Default"]
    censoring = (
        features_2020.filter(
            (pl.col("term_months") == cfg.population.term_months)
            & (pl.col("issue_d") >= pl.date(2016, 7, 1))
            & (pl.col("issue_d") <= pl.date(2017, 6, 30))
        )
        .group_by("issue_d")
        .agg(
            pl.len().alias("préstamos"),
            (pl.col("loan_status") == "Current").mean().alias("sin cerrar"),
            (
                pl.col("loan_status").is_in(["Charged Off", "Default"]).sum()
                / pl.col("loan_status").is_in(_closed).sum()
            ).alias("default entre cerrados"),
        )
        .sort("issue_d")
        .with_columns(
            (pl.col("issue_d") <= pl.lit(cfg.population.issued_until).str.to_date("%Y-%m-%d")).alias("en la población"),
            pl.col("sin cerrar").round(4),
            pl.col("default entre cerrados").round(4),
        )
    )
    censoring
    return (censoring,)


@app.cell
def _(censoring, cfg, features_2020, mo, pl):
    _worst = censoring.sort("sin cerrar", descending=True).row(0, named=True)
    _in = censoring.filter(censoring["en la población"])
    _window = features_2020.filter(
        (pl.col("term_months") == cfg.population.term_months)
        & (pl.col("issue_d") <= pl.date(2017, 6, 30))
    )
    mo.md(f"""
    El último pago observado en estos préstamos es de {_window["last_pymnt_d"].max():%Y-%m}.
    El mes con más préstamos sin cerrar es {_worst["issue_d"]:%Y-%m}, con el
    {_worst["sin cerrar"]:.1%}, y su default entre los cerrados es
    {_worst["default entre cerrados"]:.1%}. Dentro de la población, el máximo de
    préstamos sin cerrar en un mes es {_in["sin cerrar"].max():.2%}.
    """)
    return


@app.cell
def _(cfg, features_2020, mo, pl):
    _term = cfg.population.term_months
    _until = pl.lit(cfg.population.issued_until).str.to_date("%Y-%m-%d")
    _closed = ["Fully Paid", "Charged Off", "Default"]
    # "Does not meet the credit policy. Status:Fully Paid" -> "Fully Paid"
    _status = pl.col("loan_status").str.replace(r"^Does not meet the credit policy\. Status:", "")

    _steps = [("Todos los préstamos de la foto", features_2020)]
    _d = features_2020.filter(pl.col("term_months") == _term)
    _steps.append((f"Plazo de {_term} meses", _d))
    _d = _d.filter(pl.col("issue_d") <= _until)
    _steps.append((f"Emitidos hasta {cfg.population.issued_until}", _d))
    _d = _d.with_columns(_status.alias("loan_status")).filter(pl.col("loan_status").is_in(_closed))
    _steps.append(("Estado final Fully Paid / Charged Off / Default", _d))
    _d = _d.drop_nulls(["dti_n", "fico_n", "revenue"])
    _steps.append(("Sin valores nulos en dti, fico e ingresos", _d))

    population = _d.with_columns(
        (pl.col("loan_status") != "Fully Paid").cast(pl.Int64).alias("Default")
    )
    _rows = "\n".join(f"| {n} | {d.height:,} |" for n, d in _steps)
    mo.md(
        f"""
    | Filtro | Préstamos que quedan |
    |---|---|
    {_rows}
    """
    )
    return (population,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Estado en 2018

    Una de las evidencias es haber estado atrasado (*Late*) en la foto de
    finales de 2018. Esa información está en otro dataset de Kaggle, el de
    2007 a 2018. De él tomamos solo `loan_status`, que llamamos
    `loan_status_2018`. Descargamos únicamente el fichero de préstamos
    aceptados (374 MB).
    """)
    return


@app.cell
def _(cfg, pl, population):
    import kagglehub as _kh

    # Pedimos solo este fichero: el dataset entero pesa 3.8 GB.
    _file = _kh.dataset_download(cfg.kaggle.snapshot_2018, path=cfg.kaggle.snapshot_2018_file)
    snapshot_2018 = (
        pl.read_csv(_file, columns=["id", "loan_status"], infer_schema=False)
        .with_columns(pl.col("id").cast(pl.Int64, strict=False))
        .filter(pl.col("id").is_not_null())
        .rename({"loan_status": "loan_status_2018"})
    )
    with_2018 = population.join(snapshot_2018, on="id", how="left")
    assert with_2018["loan_status_2018"].null_count() == 0, (
        "Hay préstamos de la población que no aparecen en la foto de 2018."
    )
    return (with_2018,)


@app.cell
def _(mo):
    mo.md("""
    Todos los préstamos de la población aparecen en la foto de 2018 (la celda
    de arriba lo comprueba). Esta tabla muestra cómo acabó en 2020 cada grupo
    según su estado en 2018:
    """)
    return


@app.cell
def _(pl, with_2018):
    (
        with_2018.group_by("loan_status_2018", "loan_status")
        .len()
        .pivot(on="loan_status", index="loan_status_2018", values="len")
        .fill_null(0)
        .sort("loan_status_2018")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    Los que en 2018 estaban en *Late (31-120 days)* acabaron en su mayoría como
    *Charged Off*. Por eso un *Late* en 2018 solo cuenta como evidencia de
    dudoso en los que luego se pagaron.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Las tres clases

    Un préstamo es **default** si acabó en *Charged Off* o *Default*. De los
    pagados, es **dudoso** si cumple alguna de estas evidencias:

    | Evidencia | Qué dice |
    |---|---|
    | `settlement` | Negoció una quita de la deuda |
    | `hardship` | Pidió un plan de dificultades |
    | `late_fee` | Pagó algún recargo por retraso |
    | `late_2018` | Estaba en *Late* a finales de 2018 y luego pagó |
    | `paid_after_maturity` | Hizo el último pago más de un mes después del vencimiento |

    La regla está en `dslib/multiclass_labels.py`, con tests. Las evidencias que
    cuentan se eligen en `rules.evidences` de `config/funcas.yaml`.
    """)
    return


@app.cell
def _(cfg, with_2018):
    from dslib.multiclass_labels import (
        ALL_EVIDENCES, CLASS_NAMES, build_three_class_target, evidence_flags,
    )

    labeled = build_three_class_target(
        with_2018,
        maturity_grace_months=cfg.rules.maturity_grace_months,
        extra_interest_tol=cfg.rules.extra_interest_tol,
        evidences=list(cfg.rules.evidences),
    )
    labeled.select("id", "loan_status", "incident_reason", "n_evidences", "target").head(8)
    return ALL_EVIDENCES, CLASS_NAMES, evidence_flags, labeled


@app.cell
def _(CLASS_NAMES, labeled, mo, pl):
    _counts = (
        labeled.group_by("target").len().sort("target")
        .with_columns((pl.col("len") / pl.col("len").sum()).alias("share"))
    )
    _rows = "\n".join(
        f"| {CLASS_NAMES[r['target']]} | {r['len']:,} | {r['share']:.1%} |"
        for r in _counts.iter_rows(named=True)
    )
    mo.md(
        f"""
    ### Reparto de clases

    | Clase | Préstamos | Proporción |
    |---|---|---|
    {_rows}
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Cómo se comporta cada evidencia

    Para cada evidencia, sobre los préstamos pagados: cuántos la cumplen y qué
    porcentaje de ellos tiene además un recargo por retraso real. El recargo es
    el dato más directo de atraso que tenemos, porque Lending Club solo lo cobra
    pasados los 15 días de gracia. Un porcentaje alto indica que la evidencia
    mide atraso de verdad. La última columna cuenta los dudosos que lo son solo
    por esa evidencia.
    """)
    return


@app.cell
def _(ALL_EVIDENCES, cfg, evidence_flags, labeled, pl):
    _flags = evidence_flags(cfg.rules.maturity_grace_months, cfg.rules.extra_interest_tol)
    _paid = labeled.filter(pl.col("Default") == 0)
    _only_one = labeled.filter((pl.col("target") == 1) & (pl.col("n_evidences") == 1))
    _used = list(cfg.rules.evidences)
    evidence_table = pl.DataFrame(
        {
            "evidencia": ALL_EVIDENCES,
            "usada": ["sí" if n in _used else "no" for n in ALL_EVIDENCES],
            "pagados que la cumplen": [
                int(_paid.select(_flags[n].sum()).item()) for n in ALL_EVIDENCES
            ],
            "% con recargo real": [
                _paid.filter(_flags[n]).select((pl.col("total_rec_late_fee") > 0).mean()).item()
                for n in ALL_EVIDENCES
            ],
            "dudosos solo por esa": [
                int(_only_one.select(_flags[n].sum()).item()) if n in _used else 0
                for n in ALL_EVIDENCES
            ],
        }
    ).with_columns(pl.col("% con recargo real").round(3))
    evidence_table
    return (evidence_table,)


@app.cell
def _(mo):
    mo.md("""
    ### Una evidencia que descartamos: intereses por encima del calendario

    Con las cuotas y el plazo se calculan los intereses previstos (`cuota ×
    plazo − principal`). Quien paga con demora paga más intereses, así que
    probamos a marcar como dudoso al que pagó más de un 1% por encima del
    calendario.

    Para ver si funciona miramos solo los préstamos pagados hasta el vencimiento
    (los pagados antes de tiempo pagan menos intereses y no son comparables) y
    contrastamos con el recargo por retraso real.
    """)
    return


@app.cell
def _(labeled, pl):
    from dslib.multiclass_labels import maturity_date, scheduled_interest

    _breaks = [-0.005, 0.005, 0.01, 0.02, 0.05, 0.10]
    _labels = [
        "menos de -0.5%", "de -0.5% a 0.5%", "de 0.5% a 1%", "de 1% a 2%",
        "de 2% a 5%", "de 5% a 10%", "más de 10%",
    ]
    _on_time = (
        labeled.filter(pl.col("Default") == 0)
        .with_columns(
            (
                pl.col("total_rec_int")
                / scheduled_interest(pl.col("installment"), pl.col("term_months"), pl.col("funded_amnt"))
                - 1
            ).alias("exceso"),
            (
                pl.col("last_pymnt_d")
                >= maturity_date(pl.col("issue_d"), pl.col("term_months")).dt.offset_by("-2mo")
            ).alias("a_plazo"),
        )
        .filter(pl.col("a_plazo"))
    )
    excess_table = (
        _on_time.with_columns(pl.col("exceso").cut(_breaks, labels=_labels).cast(pl.String).alias("exceso de intereses"))
        .group_by("exceso de intereses")
        .agg(
            pl.len().alias("préstamos"),
            (pl.col("total_rec_late_fee") > 0).mean().round(3).alias("% con recargo real"),
        )
        .with_columns(pl.col("exceso de intereses").replace_strict(dict(zip(_labels, range(len(_labels)))), return_dtype=pl.Int64).alias("_orden"))
        .sort("_orden")
        .drop("_orden")
    )
    on_time_paid = _on_time
    excess_table
    return excess_table, on_time_paid


@app.cell
def _(cfg, excess_table, mo, on_time_paid, pl):
    _tol = cfg.rules.extra_interest_tol
    _flagged = on_time_paid.filter(pl.col("exceso") > _tol)
    _share = (_flagged["total_rec_late_fee"] > 0).mean()
    _base = (on_time_paid["total_rec_late_fee"] > 0).mean()
    _low = excess_table.filter(pl.col("exceso de intereses") == "de -0.5% a 0.5%")["% con recargo real"][0]
    _high = excess_table.filter(pl.col("exceso de intereses") == "de 5% a 10%")["% con recargo real"][0]
    mo.md(f"""
    El exceso de intereses sí se relaciona con el atraso: la probabilidad de
    recargo real pasa del {_low:.1%} cuando el exceso es casi nulo al {_high:.1%}
    cuando está entre el 5% y el 10%. Pero con un umbral del {_tol:.0%} marca
    {_flagged.height:,} préstamos pagados a plazo y solo el {_share:.1%} tiene un
    recargo real (la tasa base es {_base:.1%}). El resto probablemente pagó con
    pocos días de demora, dentro del periodo de gracia, y eso no es un atraso de
    16 días o más, que es lo que define a un dudoso. No la usamos.

    Tampoco hay un umbral limpio: el exceso va creciendo sin un salto que separe
    a los puntuales de los demorados. Subir el umbral deja muy pocos préstamos
    (ver las filas de 5% y 10%).
    """)
    return


@app.cell
def _(labeled, pl):
    # Proporción de cada clase por año de emisión.
    (
        labeled.with_columns(pl.col("issue_d").dt.year().alias("year"))
        .group_by("year", "target")
        .len()
        .pivot(on="target", index="year", values="len")
        .fill_null(0)
        .sort("year")
        .rename({"0": "no_default", "1": "dudoso", "2": "default"})
        .select("year", "no_default", "dudoso", "default")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    La proporción de cada clase cambia de un año de emisión a otro. Importa
    para el notebook 03, donde veremos qué pasa si calibramos con unos años y
    probamos con otros.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Guardar

    Guardamos las variables del modelo, la fecha de emisión, el motivo del
    dudoso y la clase.
    """)
    return


@app.cell
def _(EXPERIMENT_DIR, labeled, mo):
    out_cols = [
        "id", "fico_n", "dti_n", "loan_amnt", "revenue", "emp_length",
        "purpose", "home_ownership_n", "addr_state", "issue_d",
        "Default", "incident_reason", "n_evidences", "target",
    ]
    out_dir = EXPERIMENT_DIR / "data" / "cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "funcas.parquet"
    labeled.select(out_cols).write_parquet(out_path)

    mo.md(f"""
    Guardado `{out_path.relative_to(EXPERIMENT_DIR)}`: {labeled.height:,} filas,
    {len(out_cols)} columnas, {out_path.stat().st_size / 1024 / 1024:.1f} MB.
    """)
    return


if __name__ == "__main__":
    app.run()
