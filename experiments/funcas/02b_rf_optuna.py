import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import mlflow
    import numpy as np
    import optuna
    import pandas as pd
    from omegaconf import OmegaConf
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    from dslib.conformal_utils import mondrian_sets
    from dslib.experiment_io import (
        load_cfg, load_splits, make_preprocessor, plot_confusion, split_metrics,
    )
    from dslib.mlflow_config import local_mlflow_config
    from dslib.tracking import tracked_run
    from dslib.weighted_rf import WeightedRandomForest, run_search

    EXPERIMENT_DIR = Path(__file__).parent
    return (
        EXPERIMENT_DIR, OmegaConf, Pipeline, RandomForestClassifier,
        WeightedRandomForest, balanced_accuracy_score, load_cfg, load_splits,
        local_mlflow_config, make_preprocessor, mlflow, mo, mondrian_sets, np,
        optuna, pd, plot_confusion, plt, roc_auc_score, run_search, split_metrics,
        time, tracked_run, train_test_split,
    )


@app.cell
def _(EXPERIMENT_DIR, load_cfg, local_mlflow_config):
    cfg = load_cfg(EXPERIMENT_DIR, "funcas")

    # Seguimiento local: base SQLite y artefactos dentro de esta carpeta.
    mlflow_config = local_mlflow_config(EXPERIMENT_DIR)
    mlflow_config.validate_and_log()
    return cfg, mlflow_config


@app.cell
def _(cfg, mo):
    mo.md(f"""
    # Random forest con pesos optimizados

    En el notebook 02 el XGBoost usa pesos "balanceados": cada clase pesa en
    proporción inversa a su frecuencia, una regla fija. Aquí buscamos los pesos
    con un random forest. Se busca a la vez:

    | Parámetro | Qué controla |
    |---|---|
    | `w_dudoso` | Peso de la clase dudosa (la clase no default pesa 1) |
    | `w_default` | Peso de la clase default |
    | `n_estimators` | Número de árboles |
    | `max_depth` | Profundidad máxima de cada árbol |
    | `min_samples_leaf` | Préstamos mínimos por hoja |

    Un peso mayor hace que equivocarse con esa clase cueste más al entrenar. Los
    pesos y la profundidad se influyen entre sí, por eso se buscan juntos.

    La búsqueda usa Optuna (optimización bayesiana, algoritmo TPE): cada prueba
    se puntúa con validación cruzada y las siguientes se proponen donde las
    anteriores salieron bien. La puntuación es la *balanced accuracy*, la media
    del recall de las tres clases.

    Los dos pesos se buscan como múltiplo del peso balanceado (el que iguala la
    suma de pesos de las tres clases), entre {cfg.rf.search.weight_multiplier.low}
    y {cfg.rf.search.weight_multiplier.high} veces. La balanced accuracy se
    maximiza cerca de ese reparto, así que el rango queda centrado en él. Una
    versión anterior buscaba los pesos con un rango absoluto muy ancho y la
    búsqueda caía en soluciones que nunca predecían la clase dudosa.

    **Tiempo.** Son {cfg.rf.search.n_trials} pruebas. En el equipo donde se
    escribió (12 núcleos) tardan entre 7 y 12 minutos. Para una ejecución rápida,
    baja `rf.search.n_trials` en `config/funcas.yaml`.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Partición

    Es la del notebook 02: misma semilla, mismos préstamos en entrenamiento,
    calibración y test. Así los dos modelos son comparables en el notebook 03.
    """)
    return


@app.cell
def _(EXPERIMENT_DIR, cfg, load_splits):
    splits = load_splits(cfg, EXPERIMENT_DIR)
    X_train, y_train = splits["train"]
    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]
    names = list(cfg.data.class_names)
    return X_cal, X_test, X_train, names, y_cal, y_test, y_train


@app.cell
def _(mo):
    mo.md("""
    ## 2. Preparar la búsqueda

    Las categóricas pasan a *one-hot* antes de entrar al bosque. El codificador
    solo aprende qué categorías existen y no mira la clase, así que ajustarlo
    con todo el entrenamiento no filtra información.

    Evaluar cientos de configuraciones con todos los préstamos de entrenamiento
    sería lento. La búsqueda usa una submuestra estratificada y, dentro de ella,
    validación cruzada de 3 pliegues. El modelo final se entrena con todo el
    entrenamiento. Calibración y test no intervienen.
    """)
    return


@app.cell
def _(X_train, cfg, make_preprocessor, mo, names, np, train_test_split, y_train):
    preprocessor = make_preprocessor(cfg)
    _features = preprocessor.fit_transform(X_train)
    _labels = y_train.to_numpy()

    A_search, _, y_search, _ = train_test_split(
        _features, _labels, train_size=cfg.rf.search.n_rows, stratify=_labels,
        random_state=cfg.experiment.seed,
    )

    mo.md(f"""
    Submuestra de búsqueda: {len(y_search):,} de {len(y_train):,} préstamos de
    entrenamiento, con {A_search.shape[1]} columnas tras el *one-hot*.
    Proporción de clases: {", ".join(f"{n} {(y_search == k).mean():.1%}" for k, n in enumerate(names))}.
    """)
    return A_search, y_search


@app.cell
def _(mo):
    mo.md("""
    ## 3. Dos referencias sin búsqueda

    Buscar tiene sentido si mejora lo que ya funciona sin buscar. Entrenamos con
    todo el entrenamiento dos bosques y los medimos en test:

    - El random forest de scikit-learn **por defecto**.
    - Uno **sencillo y razonable**: hojas de al menos 20 préstamos y pesos
      balanceados.

    Además de la balanced accuracy y el AUC, miramos lo que importa en este
    experimento: el tamaño medio de los conjuntos Mondrian y qué parte recibe
    una sola clase.
    """)
    return


@app.cell
def _(
    X_cal, X_test, balanced_accuracy_score, cfg, mondrian_sets, preprocessor,
    roc_auc_score, y_cal, y_test,
):
    _cal, _test = preprocessor.transform(X_cal), preprocessor.transform(X_test)
    _y_cal, _y_test = y_cal.to_numpy(), y_test.to_numpy()

    def summarize_rf(label, forest):
        """Métricas en test de un bosque ya entrenado sobre las columnas transformadas."""
        _proba_cal, _proba_test = forest.predict_proba(_cal), forest.predict_proba(_test)
        _sets, _ = mondrian_sets(_proba_cal, _y_cal, _proba_test, cfg.conformal.alpha)
        _size = _sets.sum(axis=1)
        return {
            "modelo": label,
            "balanced accuracy": balanced_accuracy_score(_y_test, _proba_test.argmax(axis=1)),
            "AUC dudoso": roc_auc_score(_y_test == 1, _proba_test[:, 1]),
            "AUC default": roc_auc_score(_y_test == 2, _proba_test[:, 2]),
            "tamaño medio Mondrian": _size.mean(),
            "% de 1 clase": (_size == 1).mean(),
        }

    return (summarize_rf,)


@app.cell
def _(RandomForestClassifier, X_train, cfg, pd, preprocessor, summarize_rf, y_train):
    _features = preprocessor.transform(X_train)
    _seed = cfg.experiment.seed
    _references = {
        "RF por defecto (100 árboles)": RandomForestClassifier(
            n_estimators=100, random_state=_seed, n_jobs=-1
        ),
        "RF con hojas de al menos 20 y pesos balanceados": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20, class_weight="balanced",
            random_state=_seed, n_jobs=-1,
        ),
    }
    _rows = []
    for _label, _forest in _references.items():
        _forest.fit(_features, y_train)
        _rows.append(summarize_rf(_label, _forest))
    reference_table = pd.DataFrame(_rows).round(3)
    reference_table
    return (reference_table,)


@app.cell
def _(mo, reference_table):
    _default, _simple = reference_table.iloc[0], reference_table.iloc[1]
    mo.md(f"""
    El bosque por defecto tiene una balanced accuracy de
    {_default["balanced accuracy"]:.3f} y un {_default["% de 1 clase"]:.1%} de
    conjuntos de una clase. Limitar las hojas a 20 préstamos y balancear los
    pesos la sube a {_simple["balanced accuracy"]:.3f} y a un
    {_simple["% de 1 clase"]:.1%}. Con las hojas sin limitar, cada árbol crece
    hasta aislar casi cada préstamo de entrenamiento y memoriza el ruido de unas
    variables que predicen poco.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Búsqueda

    Cada prueba entrena un random forest con 2 de cada 3 préstamos de la
    submuestra y lo puntúa con el tercio restante. Lo repite 3 veces rotando y
    se queda con la media, así la puntuación nunca usa los préstamos con los que
    se entrenó.
    """)
    return


@app.cell
def _(A_search, cfg, np, run_search, time, y_search, y_train):
    _search = cfg.rf.search

    # Pesos balanceados con la clase no default en 1, y rango = múltiplo de ellos.
    _counts = np.bincount(y_train)
    balanced_weights = {"w_dudoso": _counts[0] / _counts[1], "w_default": _counts[0] / _counts[2]}
    search_space = {name: dict(spec) for name, spec in _search.space.items()}
    for _name, _base in balanced_weights.items():
        search_space[_name] = {
            "type": "float",
            "low": _base * _search.weight_multiplier.low,
            "high": _base * _search.weight_multiplier.high,
            "log": True,
        }

    _start = time.time()
    study = run_search(
        A_search,
        y_search,
        space_cfg=search_space,
        n_trials=_search.n_trials,
        cv_splits=_search.cv_splits,
        scoring=_search.scoring,
        seed=cfg.experiment.seed,
    )
    search_minutes = (time.time() - _start) / 60
    best_params = dict(study.best_params)
    trials = study.trials_dataframe()
    return balanced_weights, best_params, search_minutes, search_space, study, trials


@app.cell
def _(best_params, cfg, mo, search_minutes, study):
    _rows = "\n".join(f"| `{k}` | {v:.1f} |" if isinstance(v, float) else f"| `{k}` | {v} |" for k, v in best_params.items())
    mo.md(f"""
    La búsqueda hizo {len(study.trials)} pruebas en {search_minutes:.1f} minutos.

    | Parámetro | Valor encontrado |
    |---|---|
    {_rows}

    Mejor {cfg.rf.search.scoring.replace("_", " ")} en validación cruzada: {study.best_value:.3f}
    (prueba nº {study.best_trial.number}).
    """)
    return


@app.cell
def _(balanced_weights, best_params, mo, np, search_space, study, trials):
    # Un parámetro pegado al borde del rango indica que el óptimo puede estar fuera.
    _edges = []
    for _name, _spec in search_space.items():
        _low, _high, _value = _spec["low"], _spec["high"], best_params[_name]
        if _spec.get("log", False):
            _position = (np.log(_value) - np.log(_low)) / (np.log(_high) - np.log(_low))
        else:
            _position = (_value - _low) / (_high - _low)
        if _position > 0.95 or _position < 0.05:
            _edges.append(f"`{_name}` ({_value:.3g}, rango {_low:.3g} a {_high:.3g})")

    _best, _total = study.best_trial.number, len(study.trials)
    _edge_text = (
        f"Atención: {', '.join(_edges)} queda pegado al borde del rango, así que el "
        "óptimo puede estar fuera. Amplía el rango en `config/funcas.yaml`."
        if _edges
        else "Ningún parámetro queda pegado al borde del rango."
    )
    _late_text = (
        f"La mejor prueba es la nº {_best} de {_total}: la búsqueda sigue mejorando "
        "al final y más pruebas podrían ayudar."
        if _best > 0.8 * _total
        else f"La mejor prueba es la nº {_best} de {_total}."
    )
    # Si las mejores pruebas están muy juntas, ampliar el rango o buscar más aporta poco.
    _top = trials["value"].nlargest(10)
    _spread = _top.max() - _top.min()
    _plateau_text = (
        f"Las diez mejores pruebas difieren solo en {_spread:.3f} de puntuación: "
        "ampliar el rango o añadir pruebas aportaría poco."
        if _spread < 0.01
        else f"Las diez mejores pruebas difieren en {_spread:.3f} de puntuación."
    )
    mo.md(f"""
    Con pesos balanceados saldría `w_dudoso` = {balanced_weights["w_dudoso"]:.1f} y
    `w_default` = {balanced_weights["w_default"]:.1f}. Optuna encontró
    {best_params["w_dudoso"]:.1f} y {best_params["w_default"]:.1f}, es decir
    {best_params["w_dudoso"] / balanced_weights["w_dudoso"]:.2f} y
    {best_params["w_default"] / balanced_weights["w_default"]:.2f} veces los
    balanceados. {_edge_text} {_late_text} {_plateau_text}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Cómo evolucionó

    Cada punto es una prueba y la línea es la mejor puntuación hasta ese
    momento. Si se aplana pronto, más pruebas no ayudan.
    """)
    return


@app.cell
def _(plt, trials):
    _best_so_far = trials["value"].cummax()
    fig_evolution, ax_evolution = plt.subplots(figsize=(6.5, 3.5))
    ax_evolution.scatter(trials["number"], trials["value"], s=14, alpha=0.5, label="prueba")
    ax_evolution.plot(trials["number"], _best_so_far, color="#e45756", label="mejor hasta ahora")
    ax_evolution.set_xlabel("prueba")
    ax_evolution.set_ylabel("balanced accuracy (validación cruzada)")
    ax_evolution.legend()
    fig_evolution.tight_layout()
    fig_evolution
    return (fig_evolution,)


@app.cell
def _(mo):
    mo.md("""
    ### Las diez mejores pruebas
    """)
    return


@app.cell
def _(trials):
    _columns = ["number", "value"] + [c for c in trials.columns if c.startswith("params_")]
    (
        trials[_columns]
        .sort_values("value", ascending=False)
        .head(10)
        .rename(columns=lambda c: c.replace("params_", ""))
        .round(3)
        .reset_index(drop=True)
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Qué parámetros importan

    Optuna estima cuánto de la variación de la puntuación se debe a cada
    parámetro (método fANOVA). Los valores suman 1.
    """)
    return


@app.cell
def _(optuna, pd, study):
    _importance = optuna.importance.get_param_importances(study)
    pd.DataFrame(
        {"parámetro": list(_importance), "importancia": [round(v, 3) for v in _importance.values()]}
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Modelo final

    Entrenamos el random forest con los parámetros encontrados sobre todo el
    entrenamiento y lo guardamos en MLflow con el nombre que busca el
    notebook 03.
    """)
    return


@app.cell
def _(
    EXPERIMENT_DIR, OmegaConf, Pipeline, WeightedRandomForest, X_cal, X_test,
    X_train, best_params, cfg, fig_evolution, mlflow, mlflow_config, names,
    plot_confusion, preprocessor, search_minutes, split_metrics, study, tracked_run,
    trials, y_cal, y_test, y_train,
):
    rf_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                WeightedRandomForest(**best_params, random_state=cfg.experiment.seed),
            ),
        ]
    )

    mlflow.sklearn.autolog()
    with tracked_run(
        OmegaConf.to_container(cfg, resolve=True),
        EXPERIMENT_DIR / cfg.data.file,
        cfg.experiment.name,
        run_name=cfg.rf.run_name,
        mlflow_config=mlflow_config,
    ):
        rf_pipeline.fit(X_train, y_train)

        rf_metrics = {}
        for _prefix, _X, _y in [
            ("train_", X_train, y_train),
            ("cal_", X_cal, y_cal),
            ("test_", X_test, y_test),
        ]:
            rf_metrics.update(split_metrics(_prefix, _y, rf_pipeline.predict_proba(_X)))
        mlflow.log_metrics(rf_metrics)
        mlflow.log_metrics(
            {
                "search_best_cv_balanced_accuracy": float(study.best_value),
                "search_minutes": search_minutes,
            }
        )
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        fig_confusion = plot_confusion(
            y_test, rf_pipeline.predict(X_test), names,
            "Random forest, test: clase predicha por clase real",
        )
        mlflow.log_figure(fig_confusion, "confusion_matrix_test.png")
        mlflow.log_figure(fig_evolution, "search_evolution.png")

        _trials_csv = EXPERIMENT_DIR / "results" / "optuna_trials.csv"
        _trials_csv.parent.mkdir(exist_ok=True)
        trials.to_csv(_trials_csv, index=False)
        mlflow.log_artifact(str(_trials_csv))
    return fig_confusion, rf_metrics


@app.cell
def _(mo, names, rf_metrics):
    _rows = "\n".join(
        f"| {label} | {rf_metrics[p + 'log_loss']:.3f} | {rf_metrics[p + 'balanced_accuracy']:.3f} | "
        + " | ".join(f"{rf_metrics[f'{p}recall_{k}']:.1%}" for k in range(len(names)))
        + " |"
        for label, p in [("Entrenamiento", "train_"), ("Calibración", "cal_"), ("Test", "test_")]
    )
    _train, _test = rf_metrics["train_balanced_accuracy"], rf_metrics["test_balanced_accuracy"]
    _overfit = (
        f" El bosque rinde {_train - _test:.3f} más en entrenamiento que en test: "
        "memoriza parte del entrenamiento."
        if _train - _test > 0.05
        else ""
    )
    mo.md(
        f"""
    ## 6. Resultados

    | Partición | Log-loss | Balanced acc. | {" | ".join(f"Recall {n}" for n in names)} |
    |---|---|---|{"---|" * len(names)}
    {_rows}

    La balanced accuracy es {_train:.3f} en entrenamiento y {_test:.3f} en
    test.{_overfit} Calibración y test deben parecerse entre sí, porque el
    conformal del notebook 03 se apoya en eso.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### ¿Mejora la búsqueda a las referencias?
    """)
    return


@app.cell
def _(pd, reference_table, rf_pipeline, summarize_rf):
    rf_comparison = pd.concat(
        [
            reference_table,
            pd.DataFrame(
                [summarize_rf("RF optimizado con Optuna", rf_pipeline.named_steps["classifier"])]
            ).round(3),
        ],
        ignore_index=True,
    )
    rf_comparison
    return (rf_comparison,)


@app.cell
def _(mo, rf_comparison):
    _default, _simple, _optuna = (rf_comparison.iloc[i] for i in range(3))
    _gain = _optuna["balanced accuracy"] - _simple["balanced accuracy"]
    mo.md(f"""
    Frente al bosque por defecto, la búsqueda sube la balanced accuracy de
    {_default["balanced accuracy"]:.3f} a {_optuna["balanced accuracy"]:.3f}. Frente
    a la receta sencilla ({_simple["balanced accuracy"]:.3f}), la diferencia es de
    {_gain:+.3f}: {"la búsqueda no mejora de forma apreciable a la receta sencilla" if abs(_gain) < 0.01 else "la búsqueda mejora a la receta sencilla" if _gain > 0 else "la búsqueda queda por debajo de la receta sencilla"}.
    El tamaño medio Mondrian es {_optuna["tamaño medio Mondrian"]:.3f} con la
    búsqueda y {_simple["tamaño medio Mondrian"]:.3f} con la receta.
    """)
    return


@app.cell
def _(fig_confusion):
    fig_confusion
    return


if __name__ == "__main__":
    app.run()
