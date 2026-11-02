"""Random forest con los pesos por clase como hiperparámetros numéricos.

``RandomForestClassifier(class_weight=...)`` recibe un diccionario, y un
optimizador como Optuna solo sabe buscar números. Este
envoltorio expone el peso de cada clase como un número, de modo que se puede
buscar a la vez que ``n_estimators``, ``max_depth`` o ``min_samples_leaf``.

La clase 0 (no default) tiene peso fijo 1. Los pesos de las otras dos son
relativos a ella.

``suggest_params`` y ``run_search`` buscan esos parámetros con Optuna.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from dslib.multiclass_labels import DEFAULT, DOUBTFUL, NO_DEFAULT


class WeightedRandomForest(RandomForestClassifier):
    """``RandomForestClassifier`` con ``w_dudoso`` y ``w_default`` como parámetros."""

    def __init__(
        self,
        w_dudoso=1.0,
        w_default=1.0,
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        random_state=None,
        n_jobs=-1,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.w_dudoso = w_dudoso
        self.w_default = w_default

    def class_weights(self) -> dict[int, float]:
        return {
            NO_DEFAULT: 1.0,
            DOUBTFUL: float(self.w_dudoso),
            DEFAULT: float(self.w_default),
        }

    def fit(self, X, y, sample_weight=None):
        # Se fija aquí y no en __init__ para que set_params() también surta efecto.
        self.class_weight = self.class_weights()
        return super().fit(X, y, sample_weight=sample_weight)


def suggest_params(trial, space_cfg: dict) -> dict:
    """Pide a un ``trial`` de Optuna un valor para cada parámetro del espacio.

    Cada entrada de ``space_cfg`` es ``{type, low, high}`` con ``type`` igual a
    ``"int"`` o ``"float"`` y, opcionalmente, ``log: true`` para muestrear en
    escala logarítmica (útil cuando el rango abarca órdenes de magnitud).
    """
    params = {}
    for name, spec in space_cfg.items():
        log = bool(spec.get("log", False))
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=log)
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=log)
        else:
            raise ValueError(f"type debe ser 'int' o 'float', no {spec['type']!r} ({name})")
    return params


def run_search(
    X,
    y,
    space_cfg: dict,
    n_trials: int,
    cv_splits: int = 3,
    scoring: str = "balanced_accuracy",
    seed: int = 42,
    n_jobs: int = -1,
):
    """Busca los parámetros de ``WeightedRandomForest`` con Optuna (TPE).

    Cada prueba se puntúa con validación cruzada estratificada de ``cv_splits``
    pliegues. Devuelve el ``optuna.Study`` ya optimizado.
    """
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    def objective(trial):
        model = WeightedRandomForest(
            **suggest_params(trial, space_cfg), random_state=seed, n_jobs=n_jobs
        )
        return float(cross_val_score(model, X, y, cv=cv, scoring=scoring).mean())

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    return study
