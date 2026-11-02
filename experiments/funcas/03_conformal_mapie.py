import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import mlflow
    import numpy as np
    import pandas as pd
    import polars as pl
    from mapie.classification import SplitConformalClassifier
    from mapie.metrics.classification import (
        classification_coverage_score,
        classification_mean_width_score,
    )
    from sklearn.metrics import balanced_accuracy_score

    from dslib.conformal_utils import coverage_by_class, mondrian_sets
    from dslib.experiment_io import load_cfg, load_run_model, load_splits
    from dslib.mlflow_config import local_mlflow_config

    EXPERIMENT_DIR = Path(__file__).parent
    pd.set_option("display.max_rows", 60)  # que ninguna tabla salga recortada con "..."
    return (
        EXPERIMENT_DIR, SplitConformalClassifier, balanced_accuracy_score,
        classification_coverage_score, classification_mean_width_score,
        coverage_by_class, load_cfg, load_run_model, load_splits,
        local_mlflow_config, mlflow, mo, mondrian_sets, np, pd, pl, plt,
    )


@app.cell
def _(EXPERIMENT_DIR, load_cfg, local_mlflow_config):
    cfg = load_cfg(EXPERIMENT_DIR, "funcas")

    # Seguimiento local: base SQLite y artefactos dentro de esta carpeta.
    mlflow_config = local_mlflow_config(EXPERIMENT_DIR)
    mlflow_config.validate_and_log()
    return cfg, mlflow_config


@app.cell
def _(mo):
    mo.md("""
    # Predicción conforme en un problema de tres clases

    Un clasificador da una probabilidad por clase y nos quedamos con la mayor.
    Esa probabilidad no dice cuánto podemos fiarnos. La predicción conforme
    cambia la pregunta: en vez de una clase, devuelve un **conjunto de clases**
    que contiene la clase real con una probabilidad que elegimos de antemano,
    por ejemplo el 90%.

    Las secciones van de lo básico (conjuntos construidos a mano, luego con
    MAPIE) a lo que se complica con tres clases: cobertura por clase,
    conformal por clase (Mondrian) y datos que cambian con el tiempo.
    """)
    return


@app.cell
def _(
    EXPERIMENT_DIR, cfg, load_run_model, load_splits, mlflow_config, np, pd, pl,
):
    # mlflow_config fija la URI de seguimiento antes de buscar el modelo.
    splits = load_splits(cfg, EXPERIMENT_DIR)
    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]
    y_cal = y_cal.to_numpy()
    y_test = y_test.to_numpy()
    names = list(cfg.data.class_names)

    # El índice de X_* es la posición de la fila en el parquet: con él
    # recuperamos la fecha de emisión, que usa la sección 9.
    _issue = pd.to_datetime(
        pl.read_parquet(EXPERIMENT_DIR / cfg.data.file, columns=["issue_d"])["issue_d"].to_pandas()
    )
    dates_cal = _issue.loc[X_cal.index].to_numpy()
    dates_test = _issue.loc[X_test.index].to_numpy()

    xgb_pipeline = load_run_model(cfg, cfg.experiment.run_name, trained_in="02_modelo.py")
    proba_cal = xgb_pipeline.predict_proba(X_cal)
    proba_test = xgb_pipeline.predict_proba(X_test)
    return (
        X_cal, X_test, dates_cal, dates_test, names, proba_cal, proba_test,
        xgb_pipeline, y_cal, y_test,
    )


@app.cell
def _(mo, names, proba_cal, proba_test):
    mo.md(f"""
    Modelo: el XGBoost del notebook 02. Calibración: {len(proba_cal):,}
    préstamos. Test: {len(proba_test):,}. Clases: {", ".join(f"{k} = {n}" for k, n in enumerate(names))}.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Un préstamo

    Tomamos el primer préstamo de test y miramos qué dice el modelo.
    """)
    return


@app.cell
def _(names, pd, proba_test, y_test):
    pd.DataFrame(
        {
            "clase": names,
            "probabilidad": proba_test[0].round(3),
            "real": ["sí" if k == y_test[0] else "" for k in range(len(names))],
        }
    )
    return


@app.cell
def _(mo, proba_test):
    mo.md(f"""
    El modelo se queda con la clase de mayor probabilidad, aquí
    {proba_test[0].max():.2f}. ¿Es una decisión fiable? Las probabilidades no lo
    dicen. Para saberlo hay que ver cuánto se equivocó el modelo en préstamos
    cuya clase ya conocemos: el conjunto de calibración.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Conformal a mano (puntuación LAC)

    **Paso 1. Puntuar los errores.** Para cada préstamo de calibración miramos
    la probabilidad que el modelo dio a la clase **real** y calculamos

    $$s = 1 - \hat p(\text{clase real})$$

    Un $s$ cercano a 0 significa que el modelo acertó con confianza. Uno
    cercano a 1, que daba casi cero probabilidad a lo que ocurrió.
    """)
    return


@app.cell
def _(np, proba_cal, y_cal):
    scores_cal = 1 - proba_cal[np.arange(len(y_cal)), y_cal]
    n_cal = len(scores_cal)
    scores_cal[:5].round(3)
    return n_cal, scores_cal


@app.cell
def _(cfg, mo, n_cal, np):
    alpha = cfg.conformal.alpha
    rank = int(np.ceil((n_cal + 1) * (1 - alpha)))

    mo.md(f"""
    **Paso 2. Elegir el umbral.** Queremos cubrir el {1 - alpha:.0%} de los
    casos. Ordenamos los {n_cal:,} puntajes de menor a mayor y tomamos el que
    ocupa la posición

    $$k = \\lceil (n+1)(1-\\alpha) \\rceil = \\lceil {n_cal + 1:,} \\cdot {1 - alpha:.2f} \\rceil = {rank:,}$$

    Ese valor es el umbral $\\hat q$. El $+1$ corrige que tenemos un número
    finito de ejemplos de calibración.
    """)
    return alpha, rank


@app.cell
def _(np, rank, scores_cal):
    q_hat = np.sort(scores_cal)[rank - 1]
    float(q_hat)
    return (q_hat,)


@app.cell
def _(alpha, plt, q_hat, scores_cal):
    fig_scores, ax_scores = plt.subplots(figsize=(7, 3.5))
    ax_scores.hist(scores_cal, bins=50, color="#4c78a8")
    ax_scores.axvline(q_hat, color="#e45756", lw=2, label=f"umbral q̂ = {q_hat:.3f}")
    ax_scores.set_xlabel("s = 1 − probabilidad de la clase real")
    ax_scores.set_ylabel("préstamos de calibración")
    ax_scores.set_title(f"El {1 - alpha:.0%} de los puntajes queda a la izquierda de q̂")
    ax_scores.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    ax_scores.legend()
    fig_scores.tight_layout()
    fig_scores
    return (fig_scores,)


@app.cell
def _(mo, q_hat):
    mo.md(f"""
    **Paso 3. Construir el conjunto.** Para un préstamo nuevo incluimos todas
    las clases con $1 - \\hat p \\le \\hat q$, es decir, con

    $$\\hat p(\\text{{clase}}) \\ge 1 - \\hat q = {1 - q_hat:.3f}$$

    Si el modelo está muy seguro entra una sola clase; si duda, varias; si
    ninguna llega a ese mínimo, el conjunto queda vacío.
    """)
    return


@app.cell
def _(proba_test, q_hat):
    sets_manual = proba_test >= 1 - q_hat
    sets_manual.sum(axis=1)[:10]
    return (sets_manual,)


@app.cell
def _(names, np, pd, proba_test, sets_manual, y_test):
    # Un préstamo de cada tamaño de conjunto (1, 2 y 3 clases), si existe.
    _sizes = sets_manual.sum(axis=1)
    _examples = []
    for _size in (1, 2, 3):
        _found = np.flatnonzero(_sizes == _size)
        if len(_found) == 0:
            continue
        _loan = _found[0]
        _examples.append(
            {
                "tamaño": _size,
                **{f"p({n})": round(float(proba_test[_loan, k]), 3) for k, n in enumerate(names)},
                "conjunto": ", ".join(n for k, n in enumerate(names) if sets_manual[_loan, k]),
                "real": names[y_test[_loan]],
            }
        )
    pd.DataFrame(_examples)
    return


@app.cell
def _(mo):
    mo.md("""
    Un conjunto de una sola clase es una decisión con respaldo. Uno con dos o
    tres clases dice que, con esta información, el modelo no puede descartar
    ninguna de ellas.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Lo mismo con MAPIE

    MAPIE hace estos tres pasos en dos llamadas. `prefit=True` indica que el
    modelo ya está entrenado. `conformalize` calcula el umbral con los datos de
    calibración y `predict_set` devuelve los conjuntos.
    """)
    return


@app.cell
def _(SplitConformalClassifier, X_cal, X_test, alpha, xgb_pipeline, y_cal):
    mapie_lac = SplitConformalClassifier(
        estimator=xgb_pipeline,
        confidence_level=1 - alpha,
        conformity_score="lac",
        prefit=True,
    ).conformalize(X_cal, y_cal)

    # MAPIE devuelve (n_préstamos, n_clases, n_niveles_de_confianza).
    _, _raw = mapie_lac.predict_set(X_test)
    sets_lac = _raw[:, :, 0]
    return (sets_lac,)


@app.cell
def _(mo, sets_lac, sets_manual):
    _different = int((sets_lac != sets_manual).sum())
    mo.md(f"""
    Celdas distintas entre el conjunto de MAPIE y el calculado a mano:
    **{_different}** de {sets_lac.size:,}.
    {"Son idénticos." if _different == 0 else "Hay diferencias: revisar."}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. ¿Se cumple el 90%?

    La garantía dice que, en promedio, el conjunto contiene la clase real en al
    menos el 90% de los préstamos. La comprobamos en test, que no se usó para
    calibrar. Medimos también el tamaño medio: un conjunto con las tres clases
    siempre cubre, pero no informa.
    """)
    return


@app.cell
def _(
    alpha, classification_coverage_score, classification_mean_width_score, mo,
    sets_lac, y_test,
):
    cov_lac = float(classification_coverage_score(y_test, sets_lac[:, :, None])[0])
    width_lac = float(classification_mean_width_score(sets_lac[:, :, None])[0])

    mo.md(f"""
    - Cobertura en test: {cov_lac:.1%} (objetivo {1 - alpha:.0%}).
    - Tamaño medio: {width_lac:.2f} clases por conjunto.
    """)
    return cov_lac, width_lac


@app.cell
def _(names, pd, sets_lac):
    _sizes = sets_lac.sum(axis=1)
    _counts = pd.Series(_sizes).value_counts().reindex(range(len(names) + 1), fill_value=0)
    pd.DataFrame(
        {
            "clases en el conjunto": _counts.index,
            "préstamos": _counts.values,
            "proporción": (_counts.values / _counts.sum()).round(3),
        }
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Cobertura por clase

    La garantía anterior es **marginal**: vale para todos los préstamos juntos,
    no para cada clase. Si una clase es difícil de predecir, el modelo puede
    fallar mucho en ella y compensarlo con aciertos fáciles en las otras.
    Miramos la cobertura dentro de cada clase real.
    """)
    return


@app.cell
def _(coverage_by_class, names, pd, sets_lac, y_test):
    cov_by_class_lac = coverage_by_class(sets_lac, y_test)
    pd.DataFrame(
        {
            "clase real": names,
            "préstamos": [int((y_test == k).sum()) for k in range(len(names))],
            "cobertura": cov_by_class_lac.round(3),
        }
    )
    return (cov_by_class_lac,)


@app.cell
def _(alpha, cov_by_class_lac, mo, names, np):
    _lowest, _highest = int(np.argmin(cov_by_class_lac)), int(np.argmax(cov_by_class_lac))
    mo.md(f"""
    La cobertura va de {cov_by_class_lac[_lowest]:.1%} ({names[_lowest]}) a
    {cov_by_class_lac[_highest]:.1%} ({names[_highest]}). El {1 - alpha:.0%}
    global es un promedio: con un solo umbral cada clase recibe una cobertura
    distinta. La sección 8 lo corrige.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. LAC frente a APS

    LAC solo mira la probabilidad de la clase real. APS (*Adaptive Prediction
    Sets*) suma las probabilidades de las clases, de mayor a menor, hasta
    llegar a la real. Es más conservadora y suele dar conjuntos más grandes.
    """)
    return


@app.cell
def _(
    SplitConformalClassifier, X_cal, X_test, alpha, classification_coverage_score,
    classification_mean_width_score, coverage_by_class, cov_by_class_lac,
    cov_lac, names, pd, width_lac, xgb_pipeline, y_cal, y_test,
):
    mapie_aps = SplitConformalClassifier(
        estimator=xgb_pipeline,
        confidence_level=1 - alpha,
        conformity_score="aps",
        prefit=True,
    ).conformalize(X_cal, y_cal)
    _, _raw = mapie_aps.predict_set(X_test)
    sets_aps = _raw[:, :, 0]

    cov_aps = float(classification_coverage_score(y_test, sets_aps[:, :, None])[0])
    width_aps = float(classification_mean_width_score(sets_aps[:, :, None])[0])
    cov_by_class_aps = coverage_by_class(sets_aps, y_test)

    comparison = pd.DataFrame(
        {
            "score": ["LAC", "APS"],
            "cobertura": [cov_lac, cov_aps],
            "tamaño medio": [width_lac, width_aps],
            **{
                f"cobertura {n}": [cov_by_class_lac[k], cov_by_class_aps[k]]
                for k, n in enumerate(names)
            },
        }
    ).round(3)
    comparison
    return comparison, cov_aps, width_aps


@app.cell
def _(alpha, cov_aps, cov_lac, mo, proba_cal, width_aps, width_lac, y_cal):
    # Cuántas veces la clase real era la menos probable según el modelo.
    _least_likely = float((proba_cal.argmin(axis=1) == y_cal).mean())
    _reason = (
        f"""En el {_least_likely:.0%} de los préstamos de calibración la clase real
    era la menos probable del modelo. Para esos préstamos la puntuación de APS
    llega a 1, y como son más del {alpha:.0%}, el umbral queda en 1 y todos los
    conjuntos incluyen las tres clases."""
        if _least_likely > alpha
        else ""
    )
    mo.md(f"""
    Con APS la cobertura es {cov_aps:.1%} y el conjunto medio tiene
    {width_aps:.2f} clases. Con LAC, {cov_lac:.1%} y {width_lac:.2f}.

    {_reason}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Variar el nivel de confianza

    `confidence_level` acepta una lista, así que con una sola calibración
    obtenemos todos los niveles a la vez.
    """)
    return


@app.cell
def _(
    SplitConformalClassifier, X_cal, X_test, cfg, classification_coverage_score,
    classification_mean_width_score, pd, xgb_pipeline, y_cal, y_test,
):
    alphas = list(cfg.conformal.alpha_grid)
    mapie_grid = SplitConformalClassifier(
        estimator=xgb_pipeline,
        confidence_level=[1 - a for a in alphas],
        conformity_score="lac",
        prefit=True,
    ).conformalize(X_cal, y_cal)
    _, sets_grid = mapie_grid.predict_set(X_test)

    alpha_table = pd.DataFrame(
        {
            "alpha": alphas,
            "confianza": [1 - a for a in alphas],
            "cobertura": classification_coverage_score(y_test, sets_grid),
            "tamaño medio": classification_mean_width_score(sets_grid),
            "% conjuntos de 1 clase": (sets_grid.sum(axis=1) == 1).mean(axis=0),
        }
    ).round(3)
    alpha_table
    return (alpha_table,)


@app.cell
def _(alpha_table, mo):
    _first, _last = alpha_table.iloc[0], alpha_table.iloc[-1]
    mo.md(f"""
    - Con una confianza del {_first["confianza"]:.0%}: tamaño medio
      {_first["tamaño medio"]:.2f}; una sola clase en el
      {_first["% conjuntos de 1 clase"]:.1%} de los conjuntos.
    - Con el {_last["confianza"]:.0%}: tamaño medio {_last["tamaño medio"]:.2f};
      una sola clase en el {_last["% conjuntos de 1 clase"]:.1%}.
    """)
    return


@app.cell
def _(alpha_table, plt):
    fig_alpha, (ax_coverage, ax_size) = plt.subplots(1, 2, figsize=(9.5, 3.5))
    _confidence = alpha_table["confianza"]
    ax_coverage.plot(_confidence, alpha_table["cobertura"], "o-", label="cobertura en test")
    ax_coverage.plot(_confidence, _confidence, "k--", lw=1, label="objetivo")
    ax_coverage.set_xlabel("confianza pedida")
    ax_coverage.set_ylabel("cobertura observada")
    ax_coverage.set_title("La cobertura sigue al objetivo")
    ax_coverage.legend()

    ax_size.plot(_confidence, alpha_table["tamaño medio"], "o-", color="#e45756")
    ax_size.axhline(3, color="gray", ls=":", lw=1)
    ax_size.text(_confidence.min(), 3.02, "las tres clases", color="gray", fontsize=8)
    ax_size.set_ylim(1, 3.15)
    ax_size.set_xlabel("confianza pedida")
    ax_size.set_ylabel("tamaño medio del conjunto")
    ax_size.set_title("Más confianza, conjuntos más grandes")
    fig_alpha.tight_layout()
    fig_alpha
    return (fig_alpha,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Un umbral por clase (Mondrian)

    En la sección 5 unas clases quedaban por debajo del 90% y otras muy por
    encima. El umbral $\hat q$ es uno solo para todos los préstamos, y nada
    obliga a que cada clase reciba el mismo 90%.

    La solución es calcular el umbral **dentro de cada clase**:

    1. Para cada clase $k$, tomamos solo los préstamos de calibración cuya
       clase real es $k$.
    2. Con sus puntajes $s = 1 - \hat p(k)$ calculamos su propio $\hat q_k$,
       con la misma fórmula pero usando $n_k$ en lugar de $n$.
    3. La clase $k$ entra en el conjunto de un préstamo si
       $1 - \hat p(k) \le \hat q_k$.

    Ahora la garantía es por clase: entre los préstamos cuya clase real es $k$,
    al menos el 90% tiene $k$ en su conjunto. Esta variante se llama conformal
    *Mondrian*. MAPIE no la incluye y la calculamos nosotros
    (`dslib/conformal_utils.py`).
    """)
    return


@app.cell
def _(
    alpha, coverage_by_class, cov_by_class_lac, mondrian_sets, names, np, pd,
    proba_cal, proba_test, y_cal, y_test,
):
    sets_mond, q_mond = mondrian_sets(proba_cal, y_cal, proba_test, alpha)
    cov_by_class_mond = coverage_by_class(sets_mond, y_test)

    mondrian_table = pd.DataFrame(
        {
            "clase": names,
            "préstamos en calibración": [int((y_cal == k).sum()) for k in range(len(names))],
            "umbral q̂ (Mondrian)": q_mond.round(3),
            "cobertura (un umbral)": cov_by_class_lac.round(3),
            "cobertura (Mondrian)": cov_by_class_mond.round(3),
        }
    )
    mondrian_table
    return cov_by_class_mond, mondrian_table, sets_mond


@app.cell
def _(alpha, cov_by_class_mond, mo, np, sets_lac, sets_mond, y_test):
    _global = float(sets_mond[np.arange(len(y_test)), y_test].mean())
    _size_one, _size_mond = sets_lac.sum(axis=1).mean(), sets_mond.sum(axis=1).mean()
    mo.md(f"""
    La cobertura por clase queda entre {cov_by_class_mond.min():.1%} y
    {cov_by_class_mond.max():.1%} (objetivo {1 - alpha:.0%} en cada una). La
    global es {_global:.1%}.

    El tamaño medio del conjunto pasa de {_size_one:.2f} con un umbral a
    {_size_mond:.2f} con Mondrian.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Qué pasa si los datos cambian con el tiempo

    La garantía necesita que calibración y uso sean *intercambiables*: que un
    préstamo de test se comporte como uno de calibración. Con la partición
    aleatoria se cumple. En la práctica a veces calibramos con préstamos
    antiguos y aplicamos el modelo a préstamos nuevos.

    Lo probamos con los préstamos que el modelo no ha visto (calibración y
    test juntos), separados por año de emisión:

    - **Antiguos**: emitidos hasta el corte de `config/funcas.yaml`
      (`data.splits.train_cutoff`).
    - **Nuevos**: el resto. Los partimos al azar en dos mitades: una para
      calibrar ("misma época") y otra para medir.

    Comparamos dos calibraciones sobre el mismo test: la de los antiguos y la
    de la mitad de nuevos.
    """)
    return


@app.cell
def _(
    SplitConformalClassifier, X_cal, X_test, alpha, cfg, coverage_by_class,
    dates_cal, dates_test, names, np, pd, xgb_pipeline, y_cal, y_test,
):
    _X = pd.concat([X_cal, X_test])
    _y = np.concatenate([y_cal, y_test])
    _dates = np.concatenate([dates_cal, dates_test])
    _cutoff = np.datetime64(cfg.data.splits.train_cutoff)

    _old = np.flatnonzero(_dates <= _cutoff)
    _new = np.random.default_rng(cfg.experiment.seed).permutation(np.flatnonzero(_dates > _cutoff))
    _new_cal, _new_test = _new[: len(_new) // 2], _new[len(_new) // 2 :]

    def _evaluate(cal_idx, label):
        _conformal = SplitConformalClassifier(
            estimator=xgb_pipeline, confidence_level=1 - alpha,
            conformity_score="lac", prefit=True,
        ).conformalize(_X.iloc[cal_idx], _y[cal_idx])
        _, _raw = _conformal.predict_set(_X.iloc[_new_test])
        _sets = _raw[:, :, 0]
        _truth = _y[_new_test]
        _cov = coverage_by_class(_sets, _truth)
        return {
            "calibración": label,
            "préstamos para calibrar": len(cal_idx),
            "cobertura": _sets[np.arange(len(_truth)), _truth].mean(),
            "tamaño medio": _sets.sum(axis=1).mean(),
            **{f"cobertura {n}": _cov[k] for k, n in enumerate(names)},
        }

    temporal_table = pd.DataFrame(
        [
            _evaluate(_new_cal, "misma época (nuevos)"),
            _evaluate(_old, "época anterior (antiguos)"),
        ]
    ).round(3)

    def _class_shares(idx):
        return np.bincount(_y[idx], minlength=len(names)) / len(idx)

    drift_table = pd.DataFrame(
        {
            "grupo": ["antiguos", "nuevos"],
            "préstamos": [len(_old), len(_new)],
            **{n: [_class_shares(_old)[k], _class_shares(_new)[k]] for k, n in enumerate(names)},
        }
    ).round(3)
    return drift_table, temporal_table


@app.cell
def _(drift_table):
    drift_table
    return


@app.cell
def _(temporal_table):
    temporal_table
    return


@app.cell
def _(alpha, drift_table, mo, temporal_table):
    _same, _previous = temporal_table["cobertura"].iloc[0], temporal_table["cobertura"].iloc[1]
    _gap = (_same - _previous) * 100
    _dudoso_old, _dudoso_new = drift_table["Dudoso"].iloc[0], drift_table["Dudoso"].iloc[1]
    mo.md(f"""
    Calibrando con la misma época, la cobertura es {_same:.1%}. Con la época
    anterior, {_previous:.1%}: {abs(_gap):.1f} puntos {"menos" if _gap > 0 else "más"}
    (objetivo {1 - alpha:.0%}). La proporción de dudosos pasa de {_dudoso_old:.1%}
    en los antiguos a {_dudoso_new:.1%} en los nuevos.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. XGBoost frente al random forest optimizado

    En el notebook 02b buscamos con Optuna los pesos de un random forest,
    puntuando con la *balanced accuracy*. Aquí aplicamos a los dos modelos el
    mismo conformal (LAC al 90%) con los mismos préstamos de calibración y
    test. Además de la balanced accuracy miramos cuántos préstamos reciben una
    sola clase y cuánto mide el conjunto medio.
    """)
    return


@app.cell
def _(X_cal, X_test, cfg, load_run_model):
    rf_pipeline = load_run_model(cfg, cfg.rf.run_name, trained_in="02b_rf_optuna.py")
    proba_cal_rf = rf_pipeline.predict_proba(X_cal)
    proba_test_rf = rf_pipeline.predict_proba(X_test)
    return proba_cal_rf, proba_test_rf, rf_pipeline


@app.cell
def _(mo, rf_pipeline):
    _params = rf_pipeline.named_steps["classifier"].get_params()
    _names = ["w_dudoso", "w_default", "n_estimators", "max_depth", "min_samples_leaf"]
    _rows = "\n".join(
        f"| `{k}` | {_params[k]:.1f} |" if isinstance(_params[k], float) else f"| `{k}` | {_params[k]} |"
        for k in _names
    )
    mo.md(
        f"""
    Parámetros del random forest, encontrados por Optuna en el notebook 02b. El
    XGBoost usa pesos balanceados fijos (`n / (3 · n_clase)`).

    | Parámetro | Valor |
    |---|---|
    {_rows}
    """
    )
    return


@app.cell
def _(
    SplitConformalClassifier, X_cal, X_test, alpha, balanced_accuracy_score,
    coverage_by_class, mondrian_sets, names, np, pd, proba_cal, proba_cal_rf,
    proba_test, proba_test_rf, rf_pipeline, xgb_pipeline, y_cal, y_test,
):
    def _summarize(label, model, proba_cal_model, proba_test_model):
        _conformal = SplitConformalClassifier(
            estimator=model, confidence_level=1 - alpha,
            conformity_score="lac", prefit=True,
        ).conformalize(X_cal, y_cal)
        _, _raw = _conformal.predict_set(X_test)
        _sets = _raw[:, :, 0]
        _size = _sets.sum(axis=1)
        _sets_mondrian, _ = mondrian_sets(proba_cal_model, y_cal, proba_test_model, alpha)
        _cov = coverage_by_class(_sets, y_test)
        return {
            "modelo": label,
            "balanced accuracy": balanced_accuracy_score(y_test, proba_test_model.argmax(axis=1)),
            "cobertura": _sets[np.arange(len(y_test)), y_test].mean(),
            "tamaño medio": _size.mean(),
            "% de 1 clase": (_size == 1).mean(),
            "% de 3 clases": (_size == 3).mean(),
            **{f"cobertura {n}": _cov[k] for k, n in enumerate(names)},
            "tamaño medio (Mondrian)": _sets_mondrian.sum(axis=1).mean(),
        }

    models_table = pd.DataFrame(
        [
            _summarize("XGBoost (pesos balanceados)", xgb_pipeline, proba_cal, proba_test),
            _summarize("Random forest (pesos optimizados)", rf_pipeline, proba_cal_rf, proba_test_rf),
        ]
    ).round(3)
    models_table
    return (models_table,)


@app.cell
def _(alpha, mo, models_table, names):
    _xgb, _rf = models_table.iloc[0], models_table.iloc[1]
    _similar = 0.05  # por debajo de esta diferencia de tamaño medio decimos "casi igual"

    def _compare(column, with_what):
        _difference = _rf[column] - _xgb[column]
        _numbers = (
            f"{_rf[column]:.2f} clases de media con el random forest y "
            f"{_xgb[column]:.2f} con el XGBoost"
        )
        if abs(_difference) < _similar:
            return f"{with_what} los dos modelos dan conjuntos de tamaño casi igual: {_numbers}."
        _smaller = "el random forest" if _difference < 0 else "el XGBoost"
        return f"{with_what} {_smaller} da conjuntos más pequeños: {_numbers}."

    _by_class = "\n".join(
        f"| {n} | {_xgb[f'cobertura {n}']:.1%} | {_rf[f'cobertura {n}']:.1%} |" for n in names
    )

    # Con un solo umbral, un tamaño menor puede venir de cubrir mal una clase.
    _under = [
        f"el {label} cubre {n} al {row[f'cobertura {n}']:.1%}"
        for label, row in (("XGBoost", _xgb), ("random forest", _rf))
        for n in names
        if row[f"cobertura {n}"] < 1 - alpha - 0.05
    ]
    _caution = (
        f"Con un solo umbral, un tamaño menor puede venir de cubrir peor a una clase: {'; '.join(_under)}."
        if _under
        else "Con un solo umbral ninguna clase queda a más de 5 puntos por debajo del objetivo en ninguno de los dos modelos."
    )

    mo.md(
        f"""
    {_compare("tamaño medio", "Con un solo umbral,")} Con el random forest el
    {_rf["% de 1 clase"]:.1%} de los préstamos recibe una sola clase y con el
    XGBoost el {_xgb["% de 1 clase"]:.1%}. La balanced accuracy es
    {_rf["balanced accuracy"]:.3f} frente a {_xgb["balanced accuracy"]:.3f}.

    La cobertura global es {_rf["cobertura"]:.1%} con el random forest y
    {_xgb["cobertura"]:.1%} con el XGBoost: se cumple en los dos porque la
    garantía no depende del modelo. El reparto por clase es:

    | Clase | Cobertura XGBoost | Cobertura random forest |
    |---|---|---|
    """
        + _by_class
        + f"""

    {_caution}

    {_compare("tamaño medio (Mondrian)", "Con un umbral por clase (sección 8), donde los dos cubren cada clase al " + f"{1 - alpha:.0%},")}
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 11. Guardar resultados
    """)
    return


@app.cell
def _(
    EXPERIMENT_DIR, alpha_table, cfg, comparison, cov_aps, cov_lac, fig_alpha,
    fig_scores, mlflow, mlflow_config, mo, models_table, mondrian_table,
    temporal_table, width_aps, width_lac,
):
    _results = EXPERIMENT_DIR / "results"
    _results.mkdir(exist_ok=True)
    _tables = {
        "lac_vs_aps.csv": comparison,
        "alpha_table.csv": alpha_table,
        "mondrian.csv": mondrian_table,
        "temporal.csv": temporal_table,
        "models.csv": models_table,
    }
    _figures = {
        "scores_calibration.png": fig_scores,
        "coverage_vs_confidence.png": fig_alpha,
    }
    for _name, _table in _tables.items():
        _table.to_csv(_results / _name, index=False)
    for _name, _figure in _figures.items():
        _figure.savefig(_results / _name, dpi=150)

    _experiment_id = mlflow_config.get_or_create_experiment(cfg.experiment.conformal_experiment)
    with mlflow.start_run(experiment_id=_experiment_id, run_name=cfg.experiment.conformal_run_name):
        mlflow.log_metrics(
            {
                "coverage_lac": cov_lac, "width_lac": width_lac,
                "coverage_aps": cov_aps, "width_aps": width_aps,
            }
        )
        for _name in [*_tables, *_figures]:
            mlflow.log_artifact(str(_results / _name))

    mo.md("Guardado en `results/`: " + ", ".join(f"`{n}`" for n in [*_tables, *_figures]) + ".")
    return


if __name__ == "__main__":
    app.run()
