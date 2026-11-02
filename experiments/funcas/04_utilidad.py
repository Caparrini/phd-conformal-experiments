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
    from mapie.metrics.classification import classification_ssc_score
    from matplotlib.ticker import PercentFormatter
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    from dslib.conformal_utils import (
        bootstrap_auc, coverage_by_class, coverage_by_size, lac_sets,
        mondrian_sets, rate_by_decile, resplit_coverage, triage_table,
        wilson_interval,
    )
    from dslib.experiment_io import load_cfg, load_run_model, load_splits
    from dslib.mlflow_config import local_mlflow_config

    EXPERIMENT_DIR = Path(__file__).parent
    pd.set_option("display.max_rows", 60)  # que ninguna tabla salga recortada con "..."
    return (
        ColumnTransformer, EXPERIMENT_DIR, LogisticRegression, OneHotEncoder,
        Pipeline, PercentFormatter, StandardScaler, XGBClassifier, bootstrap_auc,
        classification_ssc_score, compute_sample_weight, coverage_by_class,
        coverage_by_size, lac_sets, load_cfg, load_run_model, load_splits,
        local_mlflow_config, mlflow, mo, mondrian_sets, np, pd, plt,
        rate_by_decile, resplit_coverage, roc_auc_score, roc_curve, triage_table,
        wilson_interval,
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
    # Utilidad del conformal Mondrian y del modelo

    El notebook 03 mostró que el Mondrian cumple su promesa de cobertura. Aquí
    miramos si sirve para algo. Separamos tres cosas, porque un mal resultado
    puede venir de cualquiera:

    1. **El modelo**: ¿distingue las clases mejor que el azar?
    2. **El conformal**: ¿cumple la cobertura de forma estable? ¿Sus conjuntos
       dicen algo, o devuelven casi siempre las tres clases?
    3. **La decisión**: ¿ayuda a decidir qué hacer con cada préstamo?

    Comparamos los dos modelos del proyecto (XGBoost y random forest con pesos
    optimizados) con un tercero que no sabe nada: un **predictor constante** que
    da a todos los préstamos las proporciones de clase del entrenamiento. Es la
    referencia: lo que un modelo mejora al constante es lo que aporta su
    información.

    Todo usa la partición de test de los notebooks 02 y 03 y una confianza del
    {1 - cfg.conformal.alpha:.0%} (alpha = {cfg.conformal.alpha}), salvo que se diga otra cosa.
    """)
    return


@app.cell
def _(EXPERIMENT_DIR, cfg, load_run_model, load_splits, mlflow_config, np):
    # mlflow_config fija la URI de seguimiento antes de buscar los modelos.
    CONSTANT = "Predictor constante"

    splits = load_splits(cfg, EXPERIMENT_DIR)
    X_train, y_train = splits["train"]
    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]
    y_train = y_train.to_numpy()
    y_cal = y_cal.to_numpy()
    y_test = y_test.to_numpy()
    names = list(cfg.data.class_names)

    probas = {}
    for _label, _run_name, _notebook in [
        ("XGBoost", cfg.experiment.run_name, "02_modelo.py"),
        ("Random forest", cfg.rf.run_name, "02b_rf_optuna.py"),
    ]:
        _model = load_run_model(cfg, _run_name, trained_in=_notebook)
        probas[_label] = {
            "cal": _model.predict_proba(X_cal),
            "test": _model.predict_proba(X_test),
        }

    # El predictor constante da a todos los préstamos las proporciones del entrenamiento.
    prior = np.bincount(y_train, minlength=len(names)) / len(y_train)
    all_models = {
        **probas,
        CONSTANT: {
            "cal": np.tile(prior, (len(y_cal), 1)),
            "test": np.tile(prior, (len(y_test), 1)),
        },
    }
    base_rate = np.bincount(y_test, minlength=len(names)) / len(y_test)
    return (
        CONSTANT, X_test, X_train, all_models, base_rate, names, prior, probas,
        y_cal, y_test, y_train,
    )


@app.cell
def _(base_rate, mo, names, prior, probas, y_cal, y_test):
    mo.md(f"""
    Calibración: {len(y_cal):,} préstamos. Test: {len(y_test):,}. Proporción de
    cada clase en test: {", ".join(f"{n} {r:.1%}" for n, r in zip(names, base_rate))}.
    El predictor constante da siempre {", ".join(f"{p:.3f}" for p in prior)}.
    Modelos: {", ".join(probas)}.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. ¿Es útil el modelo?

    Las probabilidades del modelo son útiles si ordenan los préstamos: los que
    tienen probabilidad alta de una clase deben ser de esa clase más a menudo
    que los demás.

    Medimos el **AUC** de cada clase contra el resto. Un AUC de 0.5 es azar y 1
    es perfecto. El intervalo sale de repetir el cálculo con préstamos de test
    elegidos al azar con reemplazo (*bootstrap*).
    """)
    return


@app.cell
def _(bootstrap_auc, cfg, names, pd, probas, y_test):
    auc_table = pd.DataFrame(
        [
            {
                "modelo": _label,
                "clase": _name,
                "AUC": _auc,
                "IC95 bajo": _low,
                "IC95 alto": _high,
            }
            for _label, _proba in probas.items()
            for _k, _name in enumerate(names)
            for _auc, _low, _high in [
                bootstrap_auc(
                    y_test == _k, _proba["test"][:, _k],
                    n_boot=cfg.analysis.n_bootstrap, seed=cfg.experiment.seed,
                )
            ]
        ]
    ).round(3)
    auc_table
    return (auc_table,)


@app.cell
def _(auc_table, cfg, mo):
    _auc = auc_table.set_index(["modelo", "clase"])
    _dudoso = _auc.loc[("XGBoost", "Dudoso")]
    mo.md(f"""
    Para dudoso, el AUC del XGBoost es {_dudoso["AUC"]:.3f} (IC95
    {_dudoso["IC95 bajo"]:.3f} a {_dudoso["IC95 alto"]:.3f}, con
    {cfg.analysis.n_bootstrap} remuestreos) y el del random forest
    {_auc.loc[("Random forest", "Dudoso"), "AUC"]:.3f}. Para default,
    {_auc.loc[("XGBoost", "Default"), "AUC"]:.3f} y
    {_auc.loc[("Random forest", "Default"), "AUC"]:.3f}. El predictor constante
    tiene AUC 0.5 por construcción.
    """)
    return


@app.cell
def _(auc_table, names, plt, probas, roc_curve, y_test):
    _auc = auc_table.set_index(["modelo", "clase"])["AUC"]
    fig_roc, _axes = plt.subplots(1, len(names), figsize=(11, 3.8), sharey=True)
    for _k, (_ax, _name) in enumerate(zip(_axes, names)):
        for _label, _proba in probas.items():
            _fpr, _tpr, _ = roc_curve(y_test == _k, _proba["test"][:, _k])
            _ax.plot(_fpr, _tpr, label=f"{_label} (AUC {_auc[(_label, _name)]:.2f})")
        _ax.plot([0, 1], [0, 1], "k--", lw=1, label="azar (AUC 0.50)")
        _ax.set_title(f"{_name} contra el resto")
        _ax.set_xlabel("falsos positivos")
        _ax.legend(loc="lower right", fontsize=8)
    _axes[0].set_ylabel("verdaderos positivos")
    fig_roc.tight_layout()
    fig_roc
    return (fig_roc,)


@app.cell
def _(mo):
    mo.md("""
    ### Tasa real por tramos de probabilidad

    Ordenamos los préstamos de test por la probabilidad que el modelo da a una
    clase y los partimos en diez tramos del mismo tamaño. Si el modelo ordena
    bien, la tasa real de esa clase sube de un tramo al siguiente. La línea
    discontinua es la tasa de toda la cartera.
    """)
    return


@app.cell
def _(PercentFormatter, base_rate, names, pd, plt, probas, rate_by_decile, y_test):
    _classes = (1, 2)  # dudoso y default
    fig_deciles, _axes = plt.subplots(1, len(_classes), figsize=(10, 3.8))
    _rows = []
    for _ax, _k in zip(_axes, _classes):
        for _label, _proba in probas.items():
            _t = rate_by_decile(y_test, _proba["test"][:, _k], _k)
            _ax.plot(_t["tramo"], _t["tasa_real"], "o-", label=_label)
            _rows.append(
                {
                    "modelo": _label,
                    "clase": names[_k],
                    "tasa tramo 1": _t["tasa_real"].iloc[0],
                    "tasa tramo 10": _t["tasa_real"].iloc[-1],
                    "veces": _t["tasa_real"].iloc[-1] / _t["tasa_real"].iloc[0],
                }
            )
        _ax.axhline(base_rate[_k], color="k", ls="--", lw=1, label="toda la cartera")
        _ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        _ax.set_title(f"{names[_k]} real por tramo de P({names[_k]})")
        _ax.set_xlabel("tramo de probabilidad (10 = la más alta)")
        _ax.set_ylabel(f"% que acaba en {names[_k].lower()}")
    _axes[0].legend(fontsize=8)
    fig_deciles.tight_layout()
    decile_table = pd.DataFrame(_rows).round(3)
    fig_deciles
    return decile_table, fig_deciles


@app.cell
def _(decile_table):
    decile_table
    return


@app.cell
def _(mo):
    mo.md("""
    ### ¿Es el modelo o son las variables?

    El XGBoost usa los hiperparámetros de globafa, ajustados para dos clases.
    ¿Mejoraría con otro modelo? Probamos una regresión logística y un XGBoost
    con parámetros por defecto, con las mismas variables y los mismos pesos de
    clase. Si dan lo mismo, el límite lo ponen las variables de solicitud.
    """)
    return


@app.cell
def _(
    ColumnTransformer, LogisticRegression, OneHotEncoder, Pipeline, StandardScaler,
    X_test, X_train, XGBClassifier, cfg, compute_sample_weight, names, pd, probas,
    roc_auc_score, y_test, y_train,
):
    _numerical = list(cfg.data.features.numerical)
    _categorical = list(cfg.data.features.categorical)

    def _encoder():
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    _candidates = {
        "Regresión logística": Pipeline(
            [
                ("prep", ColumnTransformer([("n", StandardScaler(), _numerical), ("c", _encoder(), _categorical)])),
                ("model", LogisticRegression(max_iter=300)),
            ]
        ),
        "XGBoost con parámetros por defecto": Pipeline(
            [
                ("prep", ColumnTransformer([("n", "passthrough", _numerical), ("c", _encoder(), _categorical)])),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob", n_jobs=-1, verbosity=0,
                        random_state=cfg.experiment.seed,
                    ),
                ),
            ]
        ),
    }
    _weights = compute_sample_weight("balanced", y_train)
    _probas_test = {}
    for _label, _pipeline in _candidates.items():
        _pipeline.fit(X_train, y_train, model__sample_weight=_weights)
        _probas_test[_label] = _pipeline.predict_proba(X_test)
    _probas_test.update({f"{label} (del proyecto)": p["test"] for label, p in probas.items()})

    baseline_table = pd.DataFrame(
        [
            {
                "modelo": _label,
                **{f"AUC {n}": roc_auc_score(y_test == k, _p[:, k]) for k, n in enumerate(names)},
            }
            for _label, _p in _probas_test.items()
        ]
    ).round(3)
    baseline_table
    return (baseline_table,)


@app.cell
def _(baseline_table, mo, names):
    _gaps = {n: baseline_table[f"AUC {n}"].max() - baseline_table[f"AUC {n}"].min() for n in names}
    _worst = max(_gaps, key=_gaps.get)
    _same = _gaps[_worst] < 0.02  # criterio orientativo: menos de 0.02 de AUC
    mo.md(f"""
    La mayor diferencia de AUC entre estos modelos es {_gaps[_worst]:.3f}
    (clase {_worst}). {"Dan prácticamente lo mismo: el límite lo ponen las variables de solicitud, no el modelo ni sus hiperparámetros." if _same else "Hay diferencias de más de 0.02 de AUC: el modelo del proyecto no es el techo con estas variables."}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. ¿Es válido el Mondrian?

    La promesa del Mondrian es que, dentro de cada clase real, el conjunto
    contiene esa clase al menos el 90% de las veces. La comprobamos de tres
    formas: en el test de siempre, repitiendo la partición muchas veces y por
    tamaño de conjunto. Para comparar calculamos también el conformal de un
    solo umbral (LAC).
    """)
    return


@app.cell
def _(all_models, cfg, lac_sets, mondrian_sets, y_cal):
    alpha = cfg.conformal.alpha
    sets_test = {
        label: {
            "Mondrian": mondrian_sets(proba["cal"], y_cal, proba["test"], alpha)[0],
            "LAC": lac_sets(proba["cal"], y_cal, proba["test"], alpha)[0],
        }
        for label, proba in all_models.items()
    }
    return alpha, sets_test


@app.cell
def _(CONSTANT, alpha, mo, names, pd, sets_test, wilson_interval, y_test):
    _rows = []
    for _label, _sets in sets_test.items():
        if _label == CONSTANT:
            continue  # cubre todo por construcción; sale en la tabla de eficiencia
        for _k, _name in enumerate(names):
            _of_class = y_test == _k
            _hits = int(_sets["Mondrian"][_of_class, _k].sum())
            _low, _high = wilson_interval(_hits, int(_of_class.sum()))
            _rows.append(
                {
                    "modelo": _label,
                    "clase": _name,
                    "préstamos de test": int(_of_class.sum()),
                    "cobertura Mondrian": _hits / _of_class.sum(),
                    "IC95 bajo": _low,
                    "IC95 alto": _high,
                    "cobertura LAC (un umbral)": _sets["LAC"][_of_class, _k].mean(),
                }
            )
    coverage_table = pd.DataFrame(_rows).round(3)
    _inside = int(
        ((coverage_table["IC95 bajo"] <= 1 - alpha) & (coverage_table["IC95 alto"] >= 1 - alpha)).sum()
    )
    mo.md(f"""
    Cobertura por clase en test, con intervalo de Wilson al 95%. El objetivo es
    {1 - alpha:.0%}. En {_inside} de las {len(coverage_table)} filas cae dentro
    del intervalo del Mondrian.
    """)
    return (coverage_table,)


@app.cell
def _(coverage_table):
    coverage_table
    return


@app.cell
def _(cfg, mo):
    mo.md(f"""
    ### Estabilidad al cambiar la partición

    Una partición es una sola muestra. Para ver cuánto varía la cobertura,
    juntamos calibración y test y repetimos {cfg.analysis.n_resplits} veces:
    partimos por la mitad al azar, calibramos con una mitad y medimos en la
    otra. Cada caja resume las repeticiones.
    """)
    return


@app.cell
def _(CONSTANT, alpha, all_models, cfg, np, pd, resplit_coverage, y_cal, y_test):
    _y_pool = np.concatenate([y_cal, y_test])
    resplits = pd.concat(
        [
            resplit_coverage(
                np.vstack([proba["cal"], proba["test"]]), _y_pool, alpha,
                n_splits=cfg.analysis.n_resplits, seed=cfg.experiment.seed,
            ).assign(modelo=label)
            for label, proba in all_models.items()
            if label != CONSTANT
        ],
        ignore_index=True,
    )
    resplits.shape
    return (resplits,)


@app.cell
def _(PercentFormatter, alpha, cfg, names, plt, resplits):
    _colors = {"Mondrian": "#4c78a8", "LAC": "#e45756"}
    fig_resplit, _axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for _ax, (_label, _group) in zip(_axes, resplits.groupby("modelo")):
        for _j, _method in enumerate(("Mondrian", "LAC")):
            _data = [
                _group[(_group["método"] == _method) & (_group["clase"] == _k)]["cobertura"]
                for _k in range(len(names))
            ]
            _positions = [_k + (_j - 0.5) * 0.35 for _k in range(len(names))]
            _boxes = _ax.boxplot(
                _data, positions=_positions, widths=0.3, patch_artist=True,
                showfliers=False, medianprops={"color": "black"},
            )
            for _box in _boxes["boxes"]:
                _box.set_facecolor(_colors[_method])
                _box.set_alpha(0.75)
            _ax.plot([], [], color=_colors[_method], lw=8, alpha=0.75, label=_method)
        _ax.axhline(1 - alpha, color="k", ls="--", lw=1)
        _ax.set_xticks(range(len(names)))
        _ax.set_xticklabels(names)
        _ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        _ax.set_title(f"{_label}: cobertura por clase")
    _axes[0].set_ylabel(f"cobertura en {cfg.analysis.n_resplits} re-particiones")
    _axes[0].legend(fontsize=8, loc="lower left")
    fig_resplit.tight_layout()
    fig_resplit
    return (fig_resplit,)


@app.cell
def _(alpha, cfg, mo, names, resplits):
    _mondrian = resplits[resplits["método"] == "Mondrian"].groupby(["modelo", "clase"])["cobertura"]
    resplit_table = (
        _mondrian.agg(media="mean", mínima="min", máxima="max").round(3).reset_index()
        .assign(clase=lambda d: d["clase"].map(dict(enumerate(names))))
    )
    _lac = resplits[resplits["método"] == "LAC"].groupby(["modelo", "clase"])["cobertura"].mean()
    mo.md(f"""
    Con Mondrian, la cobertura media por clase en las {cfg.analysis.n_resplits}
    repeticiones va de {resplit_table["media"].min():.1%} a
    {resplit_table["media"].max():.1%} (objetivo {1 - alpha:.0%}). Con un solo
    umbral va de {_lac.min():.1%} a {_lac.max():.1%}.
    """)
    return (resplit_table,)


@app.cell
def _(resplit_table):
    resplit_table
    return


@app.cell
def _(mo):
    mo.md("""
    ### ¿Y por tamaño de conjunto?

    La garantía es por clase, no por tamaño. Un conjunto de una sola clase y
    uno de tres cubren de forma distinta. Si los pequeños cubren menos del 90%,
    quien los use como "decisión segura" acertará menos de lo que cree.
    """)
    return


@app.cell
def _(
    CONSTANT, alpha, classification_ssc_score, coverage_by_size, mo, pd,
    sets_test, y_test,
):
    _models = [label for label in sets_test if label != CONSTANT]
    size_cov = pd.concat(
        [coverage_by_size(sets_test[m]["Mondrian"], y_test).assign(modelo=m) for m in _models],
        ignore_index=True,
    )[["modelo", "tamaño", "préstamos", "proporción", "cobertura"]].round(3)
    ssc = {
        m: float(classification_ssc_score(y_test, sets_test[m]["Mondrian"])[0]) for m in _models
    }
    mo.md(f"""
    Cobertura del Mondrian según el tamaño del conjunto. El mínimo entre
    tamaños (la medida `classification_ssc_score` de MAPIE) es
    {ssc["XGBoost"]:.1%} con XGBoost y {ssc["Random forest"]:.1%} con el random
    forest, frente a un objetivo del {1 - alpha:.0%}.
    """)
    return size_cov, ssc


@app.cell
def _(size_cov):
    size_cov
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. ¿Es eficiente?

    Un conjunto válido puede ser inútil: si incluye las tres clases siempre
    cubre. La eficiencia mide cuánto informa: el tamaño medio y la parte de
    préstamos que recibe una clase única.

    La referencia es el predictor constante, que no tiene información. Con
    Mondrian tiene que meter las tres clases en todos los conjuntos para cubrir
    cada una al 90%. Con un solo umbral puede saltarse una clase si las otras
    ya suman la cobertura global, y sus conjuntos salen más pequeños a costa de
    no cubrir esa clase.
    """)
    return


@app.cell
def _(CONSTANT, coverage_by_class, names, pd, sets_test, y_test):
    _constant_size = {m: sets_test[CONSTANT][m].sum(axis=1).mean() for m in ("Mondrian", "LAC")}
    _rows = []
    for _label, _sets in sets_test.items():
        for _method in ("LAC", "Mondrian"):
            _size = _sets[_method].sum(axis=1)
            _cov = coverage_by_class(_sets[_method], y_test)
            _rows.append(
                {
                    "modelo": _label,
                    "método": _method,
                    "tamaño medio": _size.mean(),
                    "% de 1 clase": (_size == 1).mean(),
                    "% de 3 clases": (_size == len(names)).mean(),
                    **{f"cobertura {n}": _cov[k] for k, n in enumerate(names)},
                    "tamaño vs constante": _size.mean() / _constant_size[_method] - 1,
                }
            )
    efficiency = pd.DataFrame(_rows).round(3)
    efficiency
    return (efficiency,)


@app.cell
def _(CONSTANT, efficiency, mo):
    _eff = efficiency.set_index(["modelo", "método"])
    _xgb, _rf = _eff.loc[("XGBoost", "Mondrian")], _eff.loc[("Random forest", "Mondrian")]
    _const_mond = _eff.loc[(CONSTANT, "Mondrian"), "tamaño medio"]
    _const_lac = _eff.loc[(CONSTANT, "LAC")]
    _cov_dudoso = _const_lac["cobertura Dudoso"]
    mo.md(f"""
    Con Mondrian, sin información el tamaño medio es {_const_mond:.2f}. El
    XGBoost lo baja a {_xgb["tamaño medio"]:.2f} ({_xgb["tamaño vs constante"]:.1%})
    y el random forest a {_rf["tamaño medio"]:.2f} ({_rf["tamaño vs constante"]:.1%}).
    Reciben una clase única el {_xgb["% de 1 clase"]:.1%} y el
    {_rf["% de 1 clase"]:.1%} de los préstamos, y las tres clases a la vez el
    {_xgb["% de 3 clases"]:.1%} y el {_rf["% de 3 clases"]:.1%}.

    Con un solo umbral, el predictor constante da {_const_lac["tamaño medio"]:.2f},
    menos que el {_eff.loc[("XGBoost", "LAC"), "tamaño medio"]:.2f} del XGBoost,
    pero cubre la clase dudosa en el {_cov_dudoso:.0%} de los casos. El tamaño
    solo se compara entre métodos que cubren las mismas clases.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### ¿En cuántos conjuntos entra cada clase?

    La clase dudosa es la razón de ser del proyecto. Si entra en casi todos los
    conjuntos, el Mondrian no consigue aislar esos préstamos.
    """)
    return


@app.cell
def _(names, pd, sets_test):
    inclusion = pd.DataFrame(
        [
            {"modelo": label, **{f"{n} en el conjunto": sets["Mondrian"][:, k].mean() for k, n in enumerate(names)}}
            for label, sets in sets_test.items()
        ]
    ).round(3)
    inclusion
    return (inclusion,)


@app.cell
def _(mo):
    mo.md("""
    ### Qué pasa al cambiar el nivel de confianza

    Bajar la confianza (subir alpha) encoge los conjuntos. Miramos cuánto hay
    que bajarla para que una parte apreciable de los préstamos reciba una clase
    única.
    """)
    return


@app.cell
def _(all_models, cfg, lac_sets, mondrian_sets, pd, y_cal):
    alpha_sweep = pd.DataFrame(
        [
            {
                "modelo": label,
                "método": method,
                "alpha": a,
                "tamaño medio": sets.sum(axis=1).mean(),
                "% de 1 clase": (sets.sum(axis=1) == 1).mean(),
            }
            for label, proba in all_models.items()
            for a in cfg.conformal.alpha_grid
            for method, fn in (("Mondrian", mondrian_sets), ("LAC", lac_sets))
            for sets in [fn(proba["cal"], y_cal, proba["test"], a)[0]]
        ]
    )
    alpha_sweep.round(3).pivot_table(index="alpha", columns=["modelo", "método"], values="tamaño medio")
    return (alpha_sweep,)


@app.cell
def _(CONSTANT, PercentFormatter, alpha_sweep, plt):
    _colors = {"XGBoost": "#4c78a8", "Random forest": "#f58518"}
    fig_sweep, (_ax_size, _ax_single) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for (_label, _method), _g in alpha_sweep[alpha_sweep["modelo"] != CONSTANT].groupby(["modelo", "método"]):
        _style = "-" if _method == "Mondrian" else "--"
        _kwargs = {"color": _colors[_label], "marker": "o", "ms": 3}
        _ax_size.plot(_g["alpha"], _g["tamaño medio"], _style, label=f"{_label} · {_method}", **_kwargs)
        _ax_single.plot(_g["alpha"], _g["% de 1 clase"], _style, **_kwargs)
    _ax_size.axhline(3, color="gray", ls=":", lw=1)
    _ax_size.text(alpha_sweep["alpha"].min(), 3.02, "Mondrian sin información (3)", color="gray", fontsize=8)
    _ax_size.set_ylim(1.4, 3.15)
    _ax_size.set_xlabel("alpha (más alto = menos confianza)")
    _ax_size.set_ylabel("tamaño medio del conjunto")
    _ax_size.set_title("Tamaño medio del conjunto")
    _ax_size.legend(fontsize=8)
    _ax_single.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _ax_single.set_xlabel("alpha (más alto = menos confianza)")
    _ax_single.set_ylabel("% de conjuntos de una clase")
    _ax_single.set_title("Conjuntos de una sola clase")
    fig_sweep.tight_layout()
    fig_sweep
    return (fig_sweep,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. ¿Ayuda a decidir?

    Probamos un triaje sencillo, sin inventar costes:

    - Conjunto = {no default}: se **acepta**.
    - Conjunto = {default}: se **escala**.
    - Cualquier otro: se **revisa**.

    Para cada grupo miramos qué parte de la cartera contiene y de qué clase son
    en realidad sus préstamos. El *lift* es esa proporción dividida por la de
    toda la cartera: 1 es igual que la media y por debajo de 1 es menos
    frecuente de lo normal.
    """)
    return


@app.cell
def _(names, pd, probas, sets_test, triage_table, y_test):
    triage = pd.concat(
        [
            triage_table(sets_test[label]["Mondrian"], y_test, names).assign(modelo=label)
            for label in probas
        ],
        ignore_index=True,
    )
    triage = triage[
        ["modelo", "grupo", "préstamos", "% cartera", *names, *[f"lift {n}" for n in names]]
    ].round(3)
    triage
    return (triage,)


@app.cell
def _(base_rate, mo, probas, triage):
    _accepted = triage[triage["grupo"] == "Aceptar"].set_index("modelo")
    mo.md(
        "\n".join(
            f"- {m}: acepta el {_accepted.loc[m, '% cartera']:.1%} de la cartera. Entre "
            f"ellos, el {_accepted.loc[m, 'Default']:.1%} acaba en default (en toda la "
            f"cartera, {base_rate[2]:.1%}) y el {_accepted.loc[m, 'Dudoso']:.1%} en "
            f"dudoso ({base_rate[1]:.1%} en la cartera)."
            for m in probas
        )
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### ¿Mejora a una regla ingenua?

    Una regla sin conformal también puede aceptar préstamos: ordenarlos por la
    probabilidad de no default y aceptar el mismo porcentaje que acepta el
    Mondrian. Si da lo mismo, el conformal no añade acierto, solo la garantía.
    """)
    return


@app.cell
def _(names, np, pd, probas, sets_test, y_test):
    _rows = []
    for _label, _proba in probas.items():
        _sets = sets_test[_label]["Mondrian"]
        _accepted_mondrian = (_sets.sum(axis=1) == 1) & _sets[:, 0]
        _p_no_default = _proba["test"][:, 0]
        _accepted_naive = _p_no_default >= np.quantile(_p_no_default, 1 - _accepted_mondrian.mean())
        for _rule, _mask in [
            ("Mondrian (conjunto = {No default})", _accepted_mondrian),
            ("Ingenua (mayor P(no default))", _accepted_naive),
        ]:
            _rows.append(
                {
                    "modelo": _label,
                    "regla": _rule,
                    "% aceptado": _mask.mean(),
                    **{n: (y_test[_mask] == k).mean() for k, n in enumerate(names)},
                }
            )
    naive_table = pd.DataFrame(_rows).round(3)
    naive_table
    return (naive_table,)


@app.cell
def _(mo):
    mo.md("""
    ### Aceptar más o menos préstamos

    Subir alpha hace que más conjuntos se reduzcan a una clase y se acepten más
    préstamos, cada vez más mezclados. La figura muestra el default real entre
    los aceptados según el porcentaje de cartera aceptado: más abajo es mejor.
    Los puntos son el Mondrian para cada alpha y la línea es la regla ingenua
    para cualquier porcentaje.
    """)
    return


@app.cell
def _(
    PercentFormatter, base_rate, cfg, mondrian_sets, np, pd, plt, probas,
    y_cal, y_test,
):
    accept_sweep = pd.DataFrame(
        [
            {
                "modelo": _label,
                "alpha": _a,
                "% aceptado": _accepted.mean(),
                "default real aceptados": (y_test[_accepted] == 2).mean() if _accepted.any() else np.nan,
                "dudoso real aceptados": (y_test[_accepted] == 1).mean() if _accepted.any() else np.nan,
            }
            for _label, _proba in probas.items()
            for _a in cfg.conformal.alpha_grid
            for _sets in [mondrian_sets(_proba["cal"], y_cal, _proba["test"], _a)[0]]
            for _accepted in [(_sets.sum(axis=1) == 1) & _sets[:, 0]]
        ]
    ).round(3)

    _colors = {"XGBoost": "#4c78a8", "Random forest": "#f58518"}
    fig_accept, _ax = plt.subplots(figsize=(6.8, 4.2))
    for _label, _proba in probas.items():
        _order = np.argsort(-_proba["test"][:, 0])
        _n_accepted = np.arange(1, len(_order) + 1)
        _fraction = _n_accepted / len(_order)
        _default_rate = np.cumsum(y_test[_order] == 2) / _n_accepted
        # Desde el 2%: con pocos préstamos aceptados la tasa es puro ruido.
        _shown = (_fraction >= 0.02) & (_fraction <= 0.30)
        _ax.plot(_fraction[_shown], _default_rate[_shown], color=_colors[_label], alpha=0.6, label=f"{_label} · regla ingenua")
        _g = accept_sweep[accept_sweep["modelo"] == _label]
        _ax.plot(_g["% aceptado"], _g["default real aceptados"], "o", color=_colors[_label], label=f"{_label} · Mondrian")
    _ax.axhline(base_rate[2], color="k", ls="--", lw=1, label="toda la cartera")
    _ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _ax.set_xlabel("% de la cartera aceptada")
    _ax.set_ylabel("% de default real entre los aceptados")
    _ax.set_title("Default entre los aceptados (más abajo es mejor)")
    _ax.legend(fontsize=7)
    fig_accept.tight_layout()
    fig_accept
    return accept_sweep, fig_accept


@app.cell
def _(accept_sweep):
    accept_sweep
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Veredicto

    Una fila por pieza, con la medida que la juzga.
    """)
    return


@app.cell
def _(
    alpha, auc_table, base_rate, cfg, decile_table, efficiency, pd, probas,
    resplit_table, ssc, triage,
):
    _auc = auc_table.set_index(["modelo", "clase"])
    _eff = efficiency.set_index(["modelo", "método"])
    _tri = triage.set_index(["modelo", "grupo"])
    _deciles = decile_table.set_index(["modelo", "clase"])
    _resplit = resplit_table.set_index("modelo")
    _n = cfg.analysis.n_resplits

    _measures = [
        ("Modelo", "AUC dudoso (0.5 = azar)", lambda m: f"{_auc.loc[(m, 'Dudoso'), 'AUC']:.3f}"),
        ("Modelo", "AUC default", lambda m: f"{_auc.loc[(m, 'Default'), 'AUC']:.3f}"),
        ("Modelo", "default real: tramo 10 / tramo 1", lambda m: f"{_deciles.loc[(m, 'Default'), 'veces']:.1f} veces"),
        ("Modelo", "dudoso real: tramo 10 / tramo 1", lambda m: f"{_deciles.loc[(m, 'Dudoso'), 'veces']:.1f} veces"),
        (
            "Mondrian: validez",
            f"cobertura por clase en {_n} re-particiones (objetivo {1 - alpha:.0%})",
            lambda m: f"{_resplit.loc[m, 'media'].min():.1%} a {_resplit.loc[m, 'media'].max():.1%}",
        ),
        ("Mondrian: validez", "cobertura mínima entre tamaños de conjunto", lambda m: f"{ssc[m]:.1%}"),
        ("Mondrian: eficiencia", "tamaño medio (sin información = 3.00)", lambda m: f"{_eff.loc[(m, 'Mondrian'), 'tamaño medio']:.2f}"),
        ("Mondrian: eficiencia", "% de conjuntos de una clase", lambda m: f"{_eff.loc[(m, 'Mondrian'), '% de 1 clase']:.1%}"),
        ("Decisión", "% de cartera aceptada", lambda m: f"{_tri.loc[(m, 'Aceptar'), '% cartera']:.1%}"),
        (
            "Decisión",
            f"default real entre los aceptados (cartera {base_rate[2]:.1%})",
            lambda m: f"{_tri.loc[(m, 'Aceptar'), 'Default']:.1%}",
        ),
    ]
    verdict = pd.DataFrame(
        [{"pieza": piece, "medida": measure, **{m: fn(m) for m in probas}} for piece, measure, fn in _measures]
    )
    verdict
    return (verdict,)


@app.cell
def _(
    alpha, auc_table, cfg, decile_table, inclusion, mo, naive_table, probas,
    resplit_table, sets_test, triage,
):
    _tolerance = cfg.analysis.coverage_tolerance
    _worst = resplit_table["media"].min()
    _valid = _worst >= 1 - alpha - _tolerance

    _auc = auc_table.set_index(["modelo", "clase"])["AUC"]
    _accepted = triage[triage["grupo"] == "Aceptar"].set_index("modelo")["Default"]
    _naive = naive_table[naive_table["regla"].str.startswith("Ingenua")].set_index("modelo")["Default"]
    _naive_better = (_naive <= _accepted).all()
    _only_dudoso = max(
        float(((s["Mondrian"].sum(axis=1) == 1) & s["Mondrian"][:, 1]).mean())
        for m, s in sets_test.items() if m in probas
    )
    _in_sets = inclusion.set_index("modelo")["Dudoso en el conjunto"].loc[list(probas)]
    _decile = decile_table.set_index(["modelo", "clase"]).loc[("XGBoost", "Default")]
    _auc_default = [_auc[(m, "Default")] for m in probas]
    _auc_dudoso = [_auc[(m, "Dudoso")] for m in probas]

    def _describe(values):
        # Escala orientativa: débil < 0.60 <= modesta < 0.70 <= aceptable < 0.80 <= buena.
        _mean = sum(values) / len(values)
        _levels = ["débil", "modesta", "aceptable", "buena"]
        _limits = [0.60, 0.70, 0.80]
        for _k, _limit in enumerate(_limits):
            if abs(_mean - _limit) < 0.01:
                return f"está en el límite entre {_levels[_k]} y {_levels[_k + 1]}"
        return "es " + _levels[sum(_mean >= _limit for _limit in _limits)]

    mo.md(f"""
    **Modelo.** Su capacidad para separar el default {_describe(_auc_default)}
    (AUC {min(_auc_default):.3f} a {max(_auc_default):.3f}; con el XGBoost la
    tasa real pasa de {_decile["tasa tramo 1"]:.1%} en el tramo bajo de
    probabilidad a {_decile["tasa tramo 10"]:.1%} en el alto). Para separar el
    dudoso {_describe(_auc_dudoso)} (AUC {min(_auc_dudoso):.3f} a
    {max(_auc_dudoso):.3f}). Escala orientativa del AUC: 0.5 es azar, menos de
    0.60 débil, de 0.60 a 0.70 modesta, de 0.70 a 0.80 aceptable, más de 0.80
    buena.

    **Mondrian.** {"Cumple" if _valid else "No cumple"} la garantía por clase: la cobertura media más baja en
    {cfg.analysis.n_resplits} re-particiones es {_worst:.1%}
    (tolerancia {_tolerance:.0%} sobre el {1 - alpha:.0%}). Los conjuntos
    incluyen la clase dudosa en el {_in_sets.min():.0%} a {_in_sets.max():.0%} de los
    casos y solo el {_only_dudoso:.1%} son exactamente {{dudoso}}.
    {"El grupo de aceptados lo consigue también una regla ingenua con igual o menos default, así que lo que aporta el Mondrian es la garantía de cobertura." if _naive_better else "El grupo de aceptados del Mondrian tiene menos default que el de la regla ingenua."}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Guardar
    """)
    return


@app.cell
def _(
    EXPERIMENT_DIR, accept_sweep, alpha_sweep, auc_table, baseline_table, cfg,
    coverage_table,
    efficiency, fig_accept, fig_deciles, fig_resplit, fig_roc, fig_sweep,
    inclusion, mlflow, mlflow_config, mo, naive_table, resplit_table, size_cov,
    triage, verdict,
):
    _results = EXPERIMENT_DIR / "results"
    _results.mkdir(exist_ok=True)
    _tables = {
        "utilidad_auc.csv": auc_table,
        "utilidad_cobertura.csv": coverage_table,
        "utilidad_resplits.csv": resplit_table,
        "utilidad_cobertura_por_tamano.csv": size_cov,
        "utilidad_eficiencia.csv": efficiency,
        "utilidad_inclusion.csv": inclusion,
        "utilidad_alpha.csv": alpha_sweep,
        "utilidad_triaje.csv": triage,
        "utilidad_ingenua.csv": naive_table,
        "utilidad_aceptar_alpha.csv": accept_sweep,
        "utilidad_veredicto.csv": verdict,
        "utilidad_modelos_base.csv": baseline_table,
    }
    _figures = {
        "utilidad_roc.png": fig_roc,
        "utilidad_deciles.png": fig_deciles,
        "utilidad_resplits.png": fig_resplit,
        "utilidad_alpha.png": fig_sweep,
        "utilidad_aceptar.png": fig_accept,
    }
    for _name, _table in _tables.items():
        _table.to_csv(_results / _name, index=False)
    for _name, _figure in _figures.items():
        _figure.savefig(_results / _name, dpi=150)

    _experiment_id = mlflow_config.get_or_create_experiment(cfg.experiment.conformal_experiment)
    with mlflow.start_run(experiment_id=_experiment_id, run_name=cfg.experiment.analysis_run_name):
        for _name in [*_tables, *_figures]:
            mlflow.log_artifact(str(_results / _name))

    mo.md("Guardado en `results/`: " + ", ".join(f"`{n}`" for n in [*_tables, *_figures]) + ".")
    return


if __name__ == "__main__":
    app.run()
