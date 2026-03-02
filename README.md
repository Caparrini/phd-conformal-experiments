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

## Stack

- **uv**: package management
- **scikit-learn**: base models
- **MLflow**: experiment tracking
- **Hydra**: experiment configuration
- **Polars**: data processing
- **Pandera**: data validation
- **Loguru**: logging
- **Ruff**: linting and formatting