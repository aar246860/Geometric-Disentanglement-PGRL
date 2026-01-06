"""Interpretability utilities: SHAP and SINDy-like discovery."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import shap

from .physics import solve_real_system
from .models import ResidualLagModel


def explain_with_shap_kernel(model, X_data: pd.DataFrame, sample_size: int = 300, background_k: int = 20):
    """Compute SHAP values using KernelExplainer."""
    def model_predict(data_array):
        return model.predict(pd.DataFrame(data_array, columns=X_data.columns))

    background = shap.kmeans(X_data, background_k)
    explainer = shap.KernelExplainer(model_predict, background)

    sample_data = X_data.iloc[:sample_size]
    shap_values = explainer.shap_values(sample_data, silent=True)
    return shap_values, sample_data


def explain_with_shap_tree(model, X_data: pd.DataFrame):
    """Compute SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    return shap_values, X_data


def select_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Select the explicit feature columns used for modeling.

    Uses Phys_Perc_Betti0 (beta_0), Phys_Perc_Backbone, Phys_Morph_Perimeter, etc.
    """
    feature_cols = ResidualLagModel.infer_feature_columns(df)
    return df[feature_cols].copy(), feature_cols


def calculate_bic(rss: float, n: int, k: int) -> float:
    safe_rss = max(float(rss), 1e-20)
    return float(n * np.log(safe_rss / n) + k * np.log(n))


def band_splits(omega: np.ndarray, n_bands: int = 3, eps: float = 1e-12):
    logw = np.log(omega + eps)
    qs = np.quantile(logw, np.linspace(0, 1, n_bands + 1))
    band_id = np.digitize(logw, qs[1:-1], right=True)
    folds = []
    for b in range(n_bands):
        te = np.where(band_id == b)[0]
        tr = np.where(band_id != b)[0]
        folds.append((tr, te))
    return folds


def build_term_dict(omega: np.ndarray, Qhat: np.ndarray, Jhat: np.ndarray, taus: Iterable[float], deriv_orders: Iterable[int]):
    iw = 1j * omega
    terms = {}
    for n in deriv_orders:
        terms[f"(iw)^{n}*Q"] = (iw**n) * Qhat
        terms[f"-(iw)^{n}*J"] = -(iw**n) * Jhat
    for tau in taus:
        terms[f"-J/(1+iw*tau),tau={tau:g}"] = -(Jhat / (1.0 + 1j * omega * tau))
    return terms


def enumerate_structures(taus: List[float], deriv_orders: List[int], max_total_terms: int, max_mem_terms: int):
    q_terms = [f"(iw)^{n}*Q" for n in deriv_orders]
    j_terms = [f"-(iw)^{n}*J" for n in deriv_orders]
    m_terms = [f"-J/(1+iw*tau),tau={tau:g}" for tau in taus]

    structs = []
    for q1 in q_terms:
        for q2 in [None] + q_terms:
            qset = [q1] + ([] if (q2 is None or q2 == q1) else [q2])

            for j1 in j_terms:
                for j2 in [None] + j_terms:
                    jset = [j1] + ([] if (j2 is None or j2 == j1) else [j2])

                    structs.append(qset + jset)
                    for r in range(1, max_mem_terms + 1):
                        pick = np.linspace(0, len(m_terms) - 1, r, dtype=int)
                        mset = [m_terms[i] for i in pick]
                        structs.append(qset + jset + mset)

    uniq = []
    seen = set()
    for s in structs:
        s2 = tuple(s)
        if len(s2) > max_total_terms:
            continue
        if s2 not in seen:
            seen.add(s2)
            uniq.append(list(s2))
    return uniq


def fit_structure_scores(term_dict: Dict[str, np.ndarray], omega: np.ndarray, folds, structure_terms: List[str], ridge: float = 1e-10):
    pivot = None
    for t in structure_terms:
        if "*Q" in t and not t.startswith("-"):
            pivot = t
            break
    if pivot is None:
        return np.inf, np.inf, np.inf

    other = [t for t in structure_terms if t != pivot]
    if len(other) == 0:
        return np.inf, np.inf, np.inf

    y_all = term_dict[pivot]
    X_all = np.column_stack([term_dict[t] for t in other])

    total_rss = 0.0
    total_n_obs = 0
    fold_errs = []

    for tr, te in folds:
        Xtr = X_all[tr, :]
        ytr = y_all[tr]
        Xte = X_all[te, :]
        yte = y_all[te]

        theta = solve_real_system(Xtr, ytr, ridge=ridge)
        ypred = Xte @ theta

        err = yte - ypred
        rss_fold = np.sum(np.real(err) ** 2 + np.imag(err) ** 2)

        total_rss += rss_fold
        total_n_obs += len(yte) * 2
        fold_errs.append(float(np.linalg.norm(err) / (np.linalg.norm(yte) + 1e-12)))

    k_params = X_all.shape[1]
    bic = calculate_bic(total_rss, total_n_obs, k_params)
    err_mean = float(np.mean(fold_errs))
    err_max = float(np.max(fold_errs))
    return bic, err_mean, err_max


def sindy_discover_structure(
    omega: np.ndarray,
    Qhat: np.ndarray,
    Jhat: np.ndarray,
    taus: List[float],
    deriv_orders: List[int],
    max_total_terms: int,
    max_mem_terms: int,
    n_bands: int = 3,
    ridge: float = 1e-10,
):
    folds = band_splits(omega, n_bands=n_bands)
    term_dict = build_term_dict(omega, Qhat, Jhat, taus, deriv_orders)
    structs = enumerate_structures(taus, deriv_orders, max_total_terms, max_mem_terms)

    best = None
    for terms in structs:
        score, err_mean, err_max = fit_structure_scores(term_dict, omega, folds, terms, ridge=ridge)
        if not np.isfinite(score):
            continue
        if best is None or score < best["score"]:
            best = {
                "terms": terms,
                "score": float(score),
                "err_mean": float(err_mean),
                "err_max": float(err_max),
                "n_terms": int(len(terms)),
            }

    return best


def sindy_fit_dataset(
    omega: np.ndarray,
    Qhat: np.ndarray,
    Jhat: np.ndarray,
    taus: List[float],
    deriv_orders: List[int],
    max_total_terms: int,
    max_mem_terms: int,
    n_bands: int = 3,
    ridge: float = 1e-10,
    universal_max_rel: float = 0.12,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Fit SINDy structures for each sample and optionally save results to CSV.

    The output structure follows the notebook logic used to produce the
    figure3_sindy_results_nature_strict*.csv files.
    """
    rows = []
    n_samples = int(Qhat.shape[0])

    for idx in range(n_samples):
        best = sindy_discover_structure(
            omega=omega,
            Qhat=Qhat[idx, :],
            Jhat=Jhat[idx, :],
            taus=taus,
            deriv_orders=deriv_orders,
            max_total_terms=max_total_terms,
            max_mem_terms=max_mem_terms,
            n_bands=n_bands,
            ridge=ridge,
        )

        if best is None:
            continue

        rows.append(
            {
                "idx": int(idx),
                "universal": int(best["err_max"] <= universal_max_rel),
                "nnz_med": float(best["n_terms"]),
                "max_rel": float(best["err_max"]),
                "n_terms": int(best["n_terms"]),
                "structure": " | ".join(best["terms"]),
                "bic_score": float(best["score"]),
            }
        )

    df_out = pd.DataFrame(rows)
    if output_path is not None:
        df_out.to_csv(output_path, index=False)
    return df_out
