"""SHAP value caching utilities using joblib.Memory.

Computes SHAP values via ``shap.KernelExplainer`` and caches them to disk.
Cache invalidation is automatic: any change to ``X_background``, ``X_explain``,
``nsamples``, or ``model_id`` triggers a recomputation. ``predict_fn`` is
intentionally excluded from the hash — use ``model_id`` as its proxy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from joblib import Memory


def _run_kernel_shap(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_background: np.ndarray,
    X_explain: np.ndarray,
    nsamples: int,
    model_id: str,
) -> np.ndarray:
    """Run KernelSHAP. ``model_id`` is a cache-invalidation key only.

    Parameters
    ----------
    predict_fn : callable
        Function ``(n_samples, n_features) -> (n_samples, n_outputs)``
        passed to ``shap.KernelExplainer``.
    X_background : np.ndarray
        Background dataset for the explainer.
    X_explain : np.ndarray
        Samples to explain.
    nsamples : int
        Number of coalitional samples for ``KernelExplainer.shap_values``.
    model_id : str
        Unique model identifier. Not used directly — only for cache keying.

    Returns
    -------
    np.ndarray
        SHAP values array.
    """
    import shap  # lazy import — not needed when loading from cache

    explainer = shap.KernelExplainer(predict_fn, X_background)
    return np.array(explainer.shap_values(X_explain, nsamples=nsamples))


def compute_or_load_shap(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_background: np.ndarray,
    X_explain: np.ndarray,
    *,
    nsamples: int,
    model_id: str,
    cache_dir: str | Path,
    force: bool = False,
) -> np.ndarray:
    """Compute SHAP values or load from joblib disk cache.

    joblib hashes ``X_background``, ``X_explain``, ``nsamples``, and
    ``model_id`` automatically. ``predict_fn`` is excluded from the hash
    (closures are not reliably hashable); rely on ``model_id`` changing
    whenever the model changes.

    Parameters
    ----------
    predict_fn : callable
        Function ``(n_samples, n_features) -> (n_samples, n_outputs)``
        passed to ``shap.KernelExplainer``.
    X_background : np.ndarray
        Background dataset for the explainer.
    X_explain : np.ndarray
        Samples to explain.
    nsamples : int
        Number of coalitional samples for ``KernelExplainer.shap_values``.
    model_id : str
        Unique model identifier (e.g. MLflow run_id). Changing this
        invalidates the cache.
    cache_dir : str or Path
        Directory where the joblib cache is stored. Created if missing.
    force : bool
        If ``True``, clear all cached entries for this function and recompute.

    Returns
    -------
    np.ndarray
        SHAP values array, shape ``(n_explain, n_features)`` or
        ``(n_explain, n_features, n_outputs)`` depending on the model.
    """
    memory = Memory(location=str(cache_dir), verbose=0)
    cached_fn = memory.cache(ignore=["predict_fn"])(_run_kernel_shap)
    if force:
        cached_fn.clear(warn=False)
    return cached_fn(predict_fn, X_background, X_explain, nsamples, model_id)
