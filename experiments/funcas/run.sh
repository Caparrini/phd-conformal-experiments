#!/bin/bash
# Ejecuta el experimento completo, en orden. Se puede lanzar desde cualquier
# carpeta y se detiene en el primer error.
#
# --no-sync usa el entorno ya instalado (ver el README) en lugar de
# sincronizarlo en cada paso.
set -euo pipefail
cd "$(dirname "$0")"

uv run --no-sync 01_datos_multiclase.py
uv run --no-sync 02_modelo.py
uv run --no-sync 02b_rf_optuna.py
uv run --no-sync 03_conformal_mapie.py
uv run --no-sync 04_utilidad.py
