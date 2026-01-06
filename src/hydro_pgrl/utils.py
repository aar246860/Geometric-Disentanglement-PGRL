"""Utility helpers for configuration and I/O."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

ZENODO_PLACEHOLDER = "Zenodo Link Placeholder"


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_path(root: str | Path, *parts: str) -> Path:
    """Join path parts relative to root."""
    return Path(root, *parts).resolve()


def project_root() -> Path:
    """Return repository root, assuming src/hydro_pgrl layout."""
    return Path(__file__).resolve().parents[2]


def processed_dir(config: Optional[Dict[str, Any]] = None, override: str | Path | None = None) -> Path:
    """Resolve processed data directory from config or override."""
    if override is not None:
        return Path(override).resolve()
    if config and "data" in config and "processed_dir" in config["data"]:
        return resolve_path(project_root(), config["data"]["processed_dir"])
    return resolve_path(project_root(), "data", "processed")


def shap_dir(config: Optional[Dict[str, Any]] = None, override: str | Path | None = None) -> Path:
    """Resolve SHAP cache directory from config or override."""
    if override is not None:
        return Path(override).resolve()
    if config and "data" in config and "shap_dir" in config["data"]:
        return resolve_path(project_root(), config["data"]["shap_dir"])
    return resolve_path(project_root(), "data", "processed", "shap")


def require_precomputed(path: Path) -> Path:
    """Ensure a precomputed file exists with a clear error message."""
    if not path.exists():
        raise FileNotFoundError(
            f"Pre-computed data {path.name} not found. "
            f"Please download from {ZENODO_PLACEHOLDER} or run simulation script."
        )
    return path


def load_processed_csv(
    filename: str,
    config: Optional[Dict[str, Any]] = None,
    processed_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load a processed CSV by filename from the processed data directory."""
    root = processed_dir(config, processed_root)
    path = require_precomputed(root / filename)
    return pd.read_csv(path)


def load_processed_dataset(
    config: Optional[Dict[str, Any]] = None,
    processed_root: str | Path | None = None,
    rebuilt_name: Optional[str] = None,
    best_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load and merge rebuilt + best datasets from processed data."""
    data_cfg = (config or {}).get("data", {})
    rebuilt_name = rebuilt_name or data_cfg.get("rebuilt_csv", "Mission_Hydro_Hybrid_DPL_Rebuilt.csv")
    best_name = best_name or data_cfg.get("best_csv", "Mission_Hydro_Hybrid_DPL_Best.csv")

    df_rebuilt = load_processed_csv(rebuilt_name, config=config, processed_root=processed_root)
    df_best = load_processed_csv(best_name, config=config, processed_root=processed_root)

    cols_to_merge = [c for c in df_best.columns if c not in df_rebuilt.columns and c != "idx"]
    df = pd.merge(df_rebuilt, df_best[["idx"] + cols_to_merge], on="idx", how="left")
    return df


def load_shap_values(
    prefix: str,
    config: Optional[Dict[str, Any]] = None,
    shap_root: str | Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load SHAP values and the corresponding feature matrix for a prefix."""
    root = shap_dir(config, shap_root)
    shap_path = require_precomputed(root / f"{prefix}_shap_values.npy")
    x_path = require_precomputed(root / f"{prefix}_X_data.csv")

    shap_vals = np.load(shap_path)
    X = pd.read_csv(x_path)
    return shap_vals, X
