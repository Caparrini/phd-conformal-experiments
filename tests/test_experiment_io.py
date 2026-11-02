"""Tests for dslib.experiment_io — carga común de los notebooks de un experimento."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from dslib.experiment_io import (
    load_cfg,
    load_run_model,
    make_preprocessor,
    plot_confusion,
    split_metrics,
)


def test_load_cfg_reads_the_experiment_yaml(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "demo.yaml").write_text("experiment:\n  name: demo\n  seed: 7\n")
    cfg = load_cfg(tmp_path, "demo")
    assert cfg.experiment.name == "demo" and cfg.experiment.seed == 7


def test_split_metrics_keys_and_values():
    y = np.array([0, 0, 1, 2, 2, 2])
    proba = np.array(
        [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1],
         [0.1, 0.1, 0.8], [0.3, 0.3, 0.4], [0.7, 0.2, 0.1]]
    )
    m = split_metrics("test_", y, proba)
    assert set(m) == {
        "test_log_loss", "test_balanced_accuracy",
        "test_recall_0", "test_recall_1", "test_recall_2",
    }
    # Aciertos por clase: 1/2, 1/1, 2/3.
    assert m["test_recall_0"] == pytest.approx(0.5)
    assert m["test_recall_1"] == pytest.approx(1.0)
    assert m["test_recall_2"] == pytest.approx(2 / 3)
    assert m["test_balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 2 / 3) / 3)


def test_plot_confusion_has_spanish_labels_and_no_colorbar():
    import matplotlib

    matplotlib.use("Agg")
    fig = plot_confusion([0, 1, 1, 2], [0, 1, 2, 2], ["a", "b", "c"], "Título")
    assert len(fig.axes) == 1  # sin barra de color
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Clase predicha"
    assert ax.get_ylabel() == "Clase real"
    assert ax.get_title() == "Título"


class _Cfg:
    class experiment:  # noqa: N801
        name = "demo"


def test_load_run_model_raises_when_the_experiment_does_not_exist():
    with mock.patch("mlflow.get_experiment_by_name", return_value=None):
        with pytest.raises(FileNotFoundError, match="02_modelo.py"):
            load_run_model(_Cfg, "demo-run", trained_in="02_modelo.py")


def test_load_run_model_raises_when_there_is_no_run():
    exp = mock.MagicMock(experiment_id="1")
    with mock.patch("mlflow.get_experiment_by_name", return_value=exp), mock.patch(
        "mlflow.search_runs", return_value=pd.DataFrame()
    ):
        with pytest.raises(FileNotFoundError, match="demo-run"):
            load_run_model(_Cfg, "demo-run", trained_in="02_modelo.py")


def test_load_run_model_loads_the_latest_run():
    exp = mock.MagicMock(experiment_id="1")
    runs = pd.DataFrame({"run_id": ["abc"]})
    with mock.patch("mlflow.get_experiment_by_name", return_value=exp), mock.patch(
        "mlflow.search_runs", return_value=runs
    ) as search, mock.patch("mlflow.sklearn.load_model", return_value="MODEL") as load:
        assert load_run_model(_Cfg, "demo-run", trained_in="02_modelo.py") == "MODEL"
    load.assert_called_once_with("runs:/abc/model")
    assert "demo-run" in search.call_args.kwargs["filter_string"]


def test_make_preprocessor_passes_numbers_and_one_hot_encodes_categories():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {"data": {"features": {"numerical": ["x"], "categorical": ["c"]}}}
    )
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "c": ["a", "b", "a"]})
    out = make_preprocessor(cfg).fit_transform(df)
    assert out.shape == (3, 3)  # x + dos categorías
    np.testing.assert_array_equal(out[:, 0], [1.0, 2.0, 3.0])
    # Una categoría que no vio al ajustar no rompe: sale todo ceros.
    unseen = make_preprocessor(cfg).fit(df).transform(pd.DataFrame({"x": [9.0], "c": ["z"]}))
    assert unseen[0, 1:].sum() == 0


def test_load_cfg_accepts_a_relative_experiment_dir(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "demo.yaml").write_text("experiment:\n  name: demo\n")
    monkeypatch.chdir(tmp_path)
    assert load_cfg(".", "demo").experiment.name == "demo"
