# Conformal Experiments

Research experiments on conformal prediction applied to financial and credit risk data.

## purpose

This repository contains experiments for a PhD thesis on machine learning optimization and explainability in fintech. 
The focus is on conformal prediction methods for credit scoring.

## Experiments

- **conformal-correctness-margin**: Methodology contribution of margin to explain results of conformal prediction.
- **conformal-p2p**: Aplication of conformal prediction to P2P lending data (Lending Club).

## Setup

```bash
uv sync
cp .env.example .env  # edit with your local MLflow URI
```

## MLflow Configuration

This project uses **centralized MLflow configuration with validation** to prevent silent failures and ensure reproducibility.

### Quick Start

```bash
# 1. Ensure MLflow server is running
mlflow ui --host 0.0.0.0 --port 5000 &

# 2. Configure in .env
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_MODE=flexible  # or 'remote', 'local'

# 3. Run an experiment
uv run experiments/conformal-correctness-margin/02-1_model_training_adult.py
```

**What happens automatically:**
- ✅ Configuration is loaded from `.env`
- ✅ Tracking server connectivity validated upfront
- ✅ Active configuration logged for debugging
- ❌ Script **blocks with clear error** if validation fails (no silent failures)

### Configuration Modes

| Mode | Remote | Local Fallback | Use Case |
|---|---|---|---|
| `flexible` | Try first | Yes if unavailable | Mixed setups (default) |
| `remote` | Required | No | CI/CD pipelines |
| `local` | No | No | Offline development |

### Environment Variables

Only two variables are required:

```bash
# .env or export
MLFLOW_TRACKING_URI=http://localhost:5000  # MLflow tracking server
MLFLOW_MODE=flexible                       # flexible|remote|local
```

**Note:** Artifact storage is managed by MLflow server (no client-side S3 configuration needed).

## Stack

- **uv**: package management
- **scikit-learn**: base models
- **MLflow**: experiment tracking (centralized config in `dslib/mlflow_config.py`)
- **Hydra**: experiment configuration
- **Polars**: data processing
- **Pandera**: data validation
- **Loguru**: logging
- **Ruff**: linting and formatting