#!/bin/bash

# Ejecutar los scripts en orden numérico usando uv run
uv run 01_data_preparation.py
uv run 02_model_training.py
uv run 03_conformal_core.py
uv run 04_fcod.py
uv run 05_shap.py
uv run 06_alpha_sensitivity.py
uv run 07_exact_shapley.py
