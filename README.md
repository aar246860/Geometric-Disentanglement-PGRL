# Geometric Disentanglement of Non-Darcian Flow (PGRL Framework)

This repository contains the refactored Physics-Guided Residual Learning (PGRL) pipeline for geometric feature extraction, lag-parameter learning, and interpretability.

## Installation

```bash
pip install -r requirements.txt
```

## Data

The `data/processed` folder contains pre-computed geometric features and targets:
- `Mission_Hydro_Hybrid_DPL_Rebuilt.csv`
- `Mission_Hydro_Hybrid_DPL_Best.csv`

It also includes cached SHAP values in `data/processed/shap/`, allowing immediate reproduction of ML and plotting steps without re-running 10k simulations.

If a required pre-computed file is missing, the loader raises:
"Pre-computed data [filename] not found. Please download from [Zenodo Link Placeholder] or run simulation script."

## Usage

```bash
python scripts/run_pipeline.py
```

## Structure

```
src/
  hydro_pgrl/
    __init__.py
    analysis.py
    features.py
    models.py
    pcc.py
    physics.py
    utils.py
```

## Citation

TBD (paper under review)
