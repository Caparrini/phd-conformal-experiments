#!/bin/bash

# Ejecutar los scripts en orden numérico usando uv run
uv run 01_data_preparation.py
uv run 02_model_training.py
uv run 03_conformal_core.py
uv run 04_fcod.py
uv run 05_test_johansson.py
uv run 06_shap_margin.py
uv run generate_report.py
