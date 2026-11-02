"""Tests for dslib.weighted_rf — random forest con pesos por clase numéricos."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score

from dslib.weighted_rf import WeightedRandomForest, run_search, suggest_params


def _data(n: int = 3000, seed: int = 0):
    """Tres clases desbalanceadas (80/5/15 %) con señal débil."""
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.80, 0.05, 0.15])
    X = rng.normal(size=(n, 5)) + y[:, None] * 0.6
    return X, y


def test_params_are_exposed_and_clonable():
    rf = WeightedRandomForest(w_dudoso=3.0, w_default=2.0, n_estimators=10)
    params = rf.get_params()
    assert params["w_dudoso"] == 3.0 and params["w_default"] == 2.0
    twin = clone(rf)
    assert twin.get_params() == params


def test_class_weight_follows_the_numeric_params():
    X, y = _data()
    rf = WeightedRandomForest(w_dudoso=7.0, w_default=2.5, n_estimators=5, random_state=0)
    rf.fit(X, y)
    assert rf.class_weight == {0: 1.0, 1: 7.0, 2: 2.5}


def test_set_params_takes_effect_on_next_fit():
    X, y = _data()
    rf = WeightedRandomForest(n_estimators=5, random_state=0).fit(X, y)
    assert rf.class_weight[1] == 1.0
    rf.set_params(w_dudoso=9.0).fit(X, y)
    assert rf.class_weight[1] == 9.0


def test_unit_weights_match_plain_forest():
    X, y = _data()
    ours = WeightedRandomForest(n_estimators=20, random_state=3, n_jobs=1).fit(X, y)
    plain = RandomForestClassifier(
        n_estimators=20, random_state=3, n_jobs=1, class_weight={0: 1.0, 1: 1.0, 2: 1.0}
    ).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X), plain.predict_proba(X))


def test_heavier_weight_raises_recall_of_that_class():
    X, y = _data()
    kw = dict(n_estimators=40, min_samples_leaf=20, random_state=0, n_jobs=1)
    light = WeightedRandomForest(w_dudoso=1.0, **kw).fit(X, y)
    heavy = WeightedRandomForest(w_dudoso=30.0, **kw).fit(X, y)
    r_light = recall_score(y, light.predict(X), labels=[1], average=None)[0]
    r_heavy = recall_score(y, heavy.predict(X), labels=[1], average=None)[0]
    assert r_heavy > r_light


SPACE = {
    "w_dudoso": {"type": "float", "low": 0.5, "high": 20.0, "log": True},
    "w_default": {"type": "float", "low": 0.5, "high": 5.0},
    "n_estimators": {"type": "int", "low": 5, "high": 15},
    "min_samples_leaf": {"type": "int", "low": 1, "high": 50, "log": True},
}


def test_suggest_params_respects_types_and_ranges():
    import optuna

    study = optuna.create_study()
    trial = study.ask()
    params = suggest_params(trial, SPACE)
    assert set(params) == set(SPACE)
    assert isinstance(params["n_estimators"], int) and 5 <= params["n_estimators"] <= 15
    assert isinstance(params["min_samples_leaf"], int)
    assert 0.5 <= params["w_dudoso"] <= 20.0
    assert 0.5 <= params["w_default"] <= 5.0


def test_suggest_params_rejects_unknown_type():
    import optuna

    trial = optuna.create_study().ask()
    with pytest.raises(ValueError, match="type"):
        suggest_params(trial, {"x": {"type": "str", "low": 0, "high": 1}})


def test_run_search_returns_best_params_inside_the_space():
    X, y = _data(n=1500)
    study = run_search(X, y, SPACE, n_trials=4, cv_splits=3, seed=0, n_jobs=1)
    assert len(study.trials) == 4
    best = study.best_params
    assert 0.5 <= best["w_dudoso"] <= 20.0
    assert 5 <= best["n_estimators"] <= 15
    assert 0.0 <= study.best_value <= 1.0


def test_run_search_is_reproducible_with_the_same_seed():
    X, y = _data(n=1500)
    a = run_search(X, y, SPACE, n_trials=3, seed=1, n_jobs=1)
    b = run_search(X, y, SPACE, n_trials=3, seed=1, n_jobs=1)
    assert a.best_params == b.best_params
    assert a.best_value == pytest.approx(b.best_value)
