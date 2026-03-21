#!/bin/bash

# Ejecutar los scripts en orden numérico usando uv run
uv run 01_data_preparation.py
uv run 02-1_model_training_adult.py
uv run 02-2_model_training_credit_card_default.py
uv run 02-3_model_training_wine_quality.py
uv run 03-1_conformal_core_adult.py
uv run 03-2_conformal_core_credit_card_default.py
uv run 03-3_conformal_core_wine_quality.py
uv run 04-1_fcod_adult.py
uv run 04-2_fcod_credit_card_default.py
uv run 04-3_fcod_wine_quality.py
uv run 05-1_shap_adult.py
uv run 05-2_shap_credit_card_default.py
uv run 05-3_shap_wine_quality.py
uv run generate_report.py
