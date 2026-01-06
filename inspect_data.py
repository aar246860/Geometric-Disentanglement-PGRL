"""Inspect processed data and model implementation details."""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


def print_columns(df: pd.DataFrame) -> None:
    cols = list(df.columns)
    print("Columns:")
    for col in cols:
        print(f"  {col}")


def find_keyword_columns(df: pd.DataFrame, keywords: list[str]) -> None:
    lower_cols = [c.lower() for c in df.columns]
    for key in keywords:
        matches = [df.columns[i] for i, c in enumerate(lower_cols) if key in c]
        if matches:
            print(f"Columns containing '{key}': {matches}")
        else:
            print(f"Columns containing '{key}': NONE")


def inspect_model_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    has_xgb = ("XGBRegressor" in text) or ("xgboost" in text) or ("xgb." in text)
    has_extra = "ExtraTreesRegressor" in text

    print(f"Model file: {path}")
    print(f"Uses XGBoost: {has_xgb}")
    print(f"Uses ExtraTrees: {has_extra}")


def inspect_notebooks() -> None:
    nb_dir = Path("notebooks/exploratory")
    found_xgb = False
    found_extra = False
    for nb in nb_dir.glob("*.ipynb"):
        data = json.loads(nb.read_text(encoding="utf-8"))
        text = "".join("".join(cell.get("source", [])) for cell in data.get("cells", []))
        if "XGBRegressor" in text or "xgboost" in text:
            found_xgb = True
        if "ExtraTreesRegressor" in text or "ExtraTrees" in text:
            found_extra = True
    print(f"Notebook scan - XGBoost detected: {found_xgb}")
    print(f"Notebook scan - ExtraTrees detected: {found_extra}")


def main() -> None:
    data_path = Path("data/processed/Mission_Hydro_Hybrid_DPL_Best.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing: {data_path}")

    df = pd.read_csv(data_path)
    print_columns(df)

    keywords = ["betti", "loop", "dim1", "tort"]
    find_keyword_columns(df, keywords)

    model_path = Path("src/hydro_pgrl/models.py")
    if model_path.exists():
        inspect_model_file(model_path)
    else:
        print("src/hydro_pgrl/models.py not found. Scanning notebooks instead.")
        inspect_notebooks()


if __name__ == "__main__":
    main()
