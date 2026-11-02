# funcas: predicción conforme con tres clases

Préstamos P2P de Lending Club clasificados en **no default**, **dudoso** y
**default**, y predicción conforme con MAPIE. Incluye el conformal por clase
(Mondrian) y un análisis de si sirve para algo.

## Antes de ejecutar

1. Instala las dependencias desde la raíz del repo:

   ```bash
   uv sync
   ```

   El `pyproject.toml` declara `conformalpy` como dependencia local
   (`../conformalpy`). funcas no la usa. Si no tienes esa carpeta, `uv sync`
   falla y debes usar:

   ```bash
   uv sync --frozen --no-install-package conformalpy
   ```

2. Necesitas un token de Kaggle (`~/.kaggle/kaggle.json`, o las variables
   `KAGGLE_USERNAME` y `KAGGLE_KEY`). El notebook 01 descarga dos datasets de
   Kaggle (1.7 GB y 374 MB) y uno de Zenodo (167 MB).

## Ejecutar

```bash
./run.sh
```

Se puede lanzar desde cualquier carpeta y se detiene en el primer error. Tarda
unos 20 minutos en un equipo de 12 núcleos, casi todo en el notebook 02b. Los
notebooks dependen unos de otros: el 03 y el 04 cargan los modelos que entrenan
el 02 y el 02b. Para abrir uno: `uv run --no-sync marimo edit 03_conformal_mapie.py`.

| Notebook | Qué hace |
|---|---|
| `01_datos_multiclase.py` | Descarga los datos y define las tres clases |
| `02_modelo.py` | Entrena un XGBoost con pesos balanceados |
| `02b_rf_optuna.py` | Busca con Optuna los pesos de un random forest |
| `03_conformal_mapie.py` | Conformal paso a paso: LAC, MAPIE, Mondrian, datos que cambian con el tiempo |
| `04_utilidad.py` | ¿Sirven el modelo y el Mondrian? AUC, cobertura, tamaño de los conjuntos, triaje |

Los parámetros están en `config/funcas.yaml`. Para probar rápido, baja
`rf.search.n_trials`: es lo que más tarda.

## Resultados

Todas las tablas y figuras se ven dentro de cada notebook. También se guardan en
`results/`.

MLflow es local y opcional: las ejecuciones van a `mlflow.db` y los artefactos a
`mlartifacts/`, dentro de esta carpeta. Los artefactos ocupan entre 70 y 250 MB
según el random forest que encuentre la búsqueda. Para verlas:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
