"""Generate a self-contained HTML report for the conformal-correctness-margin experiment.

Fetches metrics and artifacts from MLflow, renders a Jinja2 template, saves the report
locally, and logs it as an MLflow artifact in a new ``experiment-report`` run.

Usage
-----
    uv run experiments/conformal-correctness-margin/generate_report.py
"""

from __future__ import annotations

import base64
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.entities
from jinja2 import Environment, FileSystemLoader

EXPERIMENT_NAME = "conformal-correctness-margin"
EXPERIMENT_DIR = Path(__file__).parent
REPORTS_DIR = EXPERIMENT_DIR / "reports"
TEMPLATE_FILE = "report_template.html.j2"
ALPHA = 0.10

DATASETS: list[dict[str, str]] = [
    {
        "key": "adult",
        "label": "Adult Income",
        "task": "Binary",
        "model_run": "adult-xgboost",
        "conformal_run": "adult-conformal-evaluation",
        "fcod_run": "adult-fcod-analysis",
    },
    {
        "key": "credit_card_default",
        "label": "Credit Card Default",
        "task": "Binary",
        "model_run": "credit-card-xgboost",
        "conformal_run": "credit-card-conformal-evaluation",
        "fcod_run": "credit-card-fcod-analysis",
    },
    {
        "key": "wine_quality",
        "label": "Wine Quality",
        "task": "Multiclass",
        "model_run": "wine-xgboost",
        "conformal_run": "wine-conformal-evaluation",
        "fcod_run": "wine-fcod-analysis",
    },
]

# Primary artifact filenames expected in FCOD runs (order = display order)
FCOD_ARTIFACT_NAMES: list[str] = [
    "fcod_with_ci.png",
    "fcod_stacked.png",
    "outcome_by_category.png",
    "fcod_sc_si_grid.png",
]


# ── MLflow helpers ──────────────────────────────────────────────────────────


def get_latest_run(
    client: mlflow.MlflowClient,
    experiment_id: str,
    run_name: str,
) -> mlflow.entities.Run | None:
    """Return the most recent MLflow run with the given name.

    Parameters
    ----------
    client : mlflow.MlflowClient
        Active MLflow client.
    experiment_id : str
        MLflow experiment ID.
    run_name : str
        Value of the ``mlflow.runName`` tag to match.

    Returns
    -------
    mlflow.entities.Run or None
        Latest matching run, or ``None`` if none found.
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def get_metrics(run: mlflow.entities.Run) -> dict[str, float]:
    """Extract all logged metrics from a run.

    Parameters
    ----------
    run : mlflow.entities.Run
        MLflow run object.

    Returns
    -------
    dict[str, float]
        Metric name → value mapping.
    """
    return dict(run.data.metrics)


def list_artifacts_flat(
    client: mlflow.MlflowClient,
    run_id: str,
    prefix: str = "",
) -> list[str]:
    """Recursively list all artifact paths under a prefix.

    Parameters
    ----------
    client : mlflow.MlflowClient
        Active MLflow client.
    run_id : str
        Run whose artifacts to list.
    prefix : str, optional
        Artifact directory prefix.

    Returns
    -------
    list[str]
        Flat list of artifact file paths.
    """
    paths: list[str] = []
    for artifact in client.list_artifacts(run_id, prefix):
        if artifact.is_dir:
            paths.extend(list_artifacts_flat(client, run_id, artifact.path))
        else:
            paths.append(artifact.path)
    return paths


def download_artifact_b64(run_id: str, artifact_path: str) -> str | None:
    """Download a run artifact and return it base64-encoded.

    Parameters
    ----------
    run_id : str
        MLflow run ID.
    artifact_path : str
        Path of the artifact within the run.

    Returns
    -------
    str or None
        Base64-encoded bytes as ASCII string, or ``None`` on failure.
    """
    try:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/{artifact_path}"
        )
        with open(local_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    except Exception as exc:  # noqa: BLE001
        print(f"  Warning: could not download {artifact_path}: {exc}", file=sys.stderr)
        return None


# ── Formatting helper (also passed into Jinja2 context) ─────────────────────


def fmt(value: float | None, fmt_spec: str = ".3f") -> str:
    """Safely format a numeric value, returning '—' for None.

    Parameters
    ----------
    value : float or None
        Value to format.
    fmt_spec : str, optional
        Python format specification (e.g. ``'.3f'``, ``'.1%'``).

    Returns
    -------
    str
        Formatted string or ``'—'`` if value is None.
    """
    if value is None:
        return "—"
    try:
        return format(value, fmt_spec)
    except (TypeError, ValueError):
        return "—"


# ── Git ──────────────────────────────────────────────────────────────────────


def get_git_commit() -> str | None:
    """Return the short HEAD commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=EXPERIMENT_DIR,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return None


# ── Context building ─────────────────────────────────────────────────────────


def build_dataset_block(
    client: mlflow.MlflowClient,
    experiment_id: str,
    dataset_cfg: dict[str, str],
) -> dict[str, Any]:
    """Build the template context block for one dataset.

    Parameters
    ----------
    client : mlflow.MlflowClient
        Active MLflow client.
    experiment_id : str
        MLflow experiment ID.
    dataset_cfg : dict[str, str]
        Keys: ``key``, ``label``, ``task``, ``model_run``,
        ``conformal_run``, ``fcod_run``.

    Returns
    -------
    dict[str, Any]
        Block with ``model``, ``conformal``, and ``artifacts`` sub-dicts.
    """
    key = dataset_cfg["key"]

    # ── Stage 02: model metrics ──────────────────────────────────────────────
    print(f"  [{key}] model run: {dataset_cfg['model_run']}")
    model_run = get_latest_run(client, experiment_id, dataset_cfg["model_run"])
    mm = get_metrics(model_run) if model_run else {}

    train_roc = mm.get("train_roc_auc")
    test_roc = mm.get("test_roc_auc")
    generalisation_delta = (
        test_roc - train_roc
        if (train_roc is not None and test_roc is not None)
        else None
    )

    model: dict[str, Any] = {
        "train_roc_auc":           train_roc,
        "cal_roc_auc":             mm.get("cal_roc_auc"),
        "test_roc_auc":            test_roc,
        "train_balanced_accuracy": mm.get("train_balanced_accuracy"),
        "cal_balanced_accuracy":   mm.get("cal_balanced_accuracy"),
        "test_balanced_accuracy":  mm.get("test_balanced_accuracy"),
        "train_f1_macro":          mm.get("train_f1_macro"),
        "cal_f1_macro":            mm.get("cal_f1_macro"),
        "test_f1_macro":           mm.get("test_f1_macro"),
        "train_precision_macro":   mm.get("train_precision_macro"),   # None for wine
        "cal_precision_macro":     mm.get("cal_precision_macro"),
        "test_precision_macro":    mm.get("test_precision_macro"),
        "train_recall_macro":      mm.get("train_recall_macro"),      # None for wine
        "cal_recall_macro":        mm.get("cal_recall_macro"),
        "test_recall_macro":       mm.get("test_recall_macro"),
        "log_loss":                mm.get("log_loss"),                # autolog, train only
        "generalisation_delta":    generalisation_delta,
    }

    # ── Stage 02: model training artifacts (autolog) ─────────────────────────
    model_artifacts: dict[str, str | None] = {}
    if model_run:
        for art_name in [
            "training_confusion_matrix.png",
            "training_roc_curve.png",
            "training_precision_recall_curve.png",
        ]:
            art_key = art_name.replace(".png", "")
            model_artifacts[art_key] = download_artifact_b64(model_run.info.run_id, art_name)

    # ── Stage 03: conformal metrics ──────────────────────────────────────────
    print(f"  [{key}] conformal run: {dataset_cfg['conformal_run']}")
    conformal_run = get_latest_run(client, experiment_id, dataset_cfg["conformal_run"])
    cm = get_metrics(conformal_run) if conformal_run else {}

    per_class_coverage: dict[str, float] | None = {
        k.replace("test_coverage_class_", ""): v
        for k, v in cm.items()
        if k.startswith("test_coverage_class_")
    }
    if not per_class_coverage:
        per_class_coverage = None

    cal_qhat: dict[str, float] | None = {
        k.replace("cal_qhat_", ""): v
        for k, v in cm.items()
        if k.startswith("cal_qhat_")
    }
    if not cal_qhat:
        cal_qhat = None

    test_coverage = cm.get("test_coverage")
    coverage_gap = (
        test_coverage - (1.0 - ALPHA)
        if test_coverage is not None
        else None
    )

    conformal: dict[str, Any] = {
        "test_coverage":             test_coverage,
        "test_coverage_gap":         coverage_gap,          # computed, not fetched
        "test_sc_rate":              cm.get("test_sc_rate"),
        "test_si_rate":              cm.get("test_si_rate"),
        "test_uncertainty_rate":     cm.get("test_uncertainty_rate"),
        "test_avg_set_size":         cm.get("test_avg_set_size"),
        "test_singleton_rate":       cm.get("test_singleton_rate"),
        "test_empty_rate":           cm.get("test_empty_rate"),
        "test_accuracy_when_confident": cm.get("test_accuracy_when_confident"),
        "per_class_coverage":        per_class_coverage,
        "test_si_fp_rate":           cm.get("test_si_fp_rate"),           # binary only
        "test_si_fn_rate":           cm.get("test_si_fn_rate"),           # binary only
        "test_si_overpred_rate":     cm.get("test_si_overpred_rate"),     # wine only
        "test_si_underpred_rate":    cm.get("test_si_underpred_rate"),    # wine only
        "cal_qhat":                  cal_qhat,
    }

    # ── Stage 03: conformal artifacts ────────────────────────────────────────
    conformal_artifacts: dict[str, str | None] = {}
    if conformal_run:
        for art_name in [
            "coverage_by_alpha.png",
            "singleton_confusion_matrix.png",
            "prediction_set_size_distribution.png",
            "set_size_vs_true_label.png",
        ]:
            art_key = art_name.replace(".png", "")
            conformal_artifacts[art_key] = download_artifact_b64(
                conformal_run.info.run_id, art_name
            )

    # ── Stage 04: FCOD artifacts ─────────────────────────────────────────────
    print(f"  [{key}] FCOD run: {dataset_cfg['fcod_run']}")
    fcod_run = get_latest_run(client, experiment_id, dataset_cfg["fcod_run"])
    artifacts: dict[str, str | None] = {}

    if fcod_run:
        run_id = fcod_run.info.run_id
        # Download primary artifacts in display order
        for artifact_name in FCOD_ARTIFACT_NAMES:
            artifact_key = artifact_name.replace(".png", "")
            artifacts[artifact_key] = download_artifact_b64(run_id, artifact_name)

        # Collect any additional PNG artifacts not already fetched
        all_paths = list_artifacts_flat(client, run_id)
        for path in all_paths:
            filename = Path(path).name
            if filename.endswith(".png"):
                artifact_key = filename.replace(".png", "")
                if artifact_key not in artifacts:
                    artifacts[artifact_key] = download_artifact_b64(run_id, path)

    return {
        "key": key,
        "label": dataset_cfg["label"],
        "task": dataset_cfg["task"],
        "model": model,
        "model_artifacts": model_artifacts,
        "conformal": conformal,
        "conformal_artifacts": conformal_artifacts,
        "artifacts": artifacts,
    }


def build_context(
    client: mlflow.MlflowClient,
    experiment_id: str,
) -> dict[str, Any]:
    """Build the full Jinja2 template context.

    Parameters
    ----------
    client : mlflow.MlflowClient
        Active MLflow client.
    experiment_id : str
        MLflow experiment ID.

    Returns
    -------
    dict[str, Any]
        Top-level context dict for the template.
    """
    datasets = [
        build_dataset_block(client, experiment_id, cfg) for cfg in DATASETS
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_id": experiment_id,
        "alpha": ALPHA,
        "git_commit": get_git_commit(),
        "datasets": datasets,
    }


# ── Rendering ────────────────────────────────────────────────────────────────


def render_report(ctx: dict[str, Any]) -> str:
    """Render the Jinja2 HTML template with the given context.

    Parameters
    ----------
    ctx : dict[str, Any]
        Template context produced by :func:`build_context`.

    Returns
    -------
    str
        Rendered HTML string.
    """
    env = Environment(
        loader=FileSystemLoader(str(EXPERIMENT_DIR)),
        autoescape=False,
    )
    env.globals["fmt"] = fmt
    template = env.get_template(TEMPLATE_FILE)
    return template.render(**ctx)


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    """Fetch MLflow data, render report HTML, log artifact."""
    client = mlflow.MlflowClient()

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Error: experiment '{EXPERIMENT_NAME}' not found.", file=sys.stderr)
        sys.exit(1)

    experiment_id = experiment.experiment_id
    print(f"Experiment: {EXPERIMENT_NAME} (id={experiment_id})")

    print("Fetching runs from MLflow...")
    ctx = build_context(client, experiment_id)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"report_{timestamp}.html"

    print("Rendering report...")
    html = render_report(ctx)
    report_path.write_text(html, encoding="utf-8")
    print(f"Report saved to {report_path}")

    print("Logging to MLflow...")
    with mlflow.start_run(experiment_id=experiment_id, run_name="experiment-report"):
        mlflow.log_artifact(str(report_path), artifact_path="")
        mlflow.set_tag("report_timestamp", timestamp)
        run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]

    print(f"Logged to MLflow run: {run_id}")


if __name__ == "__main__":
    main()
