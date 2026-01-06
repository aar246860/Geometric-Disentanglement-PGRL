"""Demonstration pipeline for PCC -> physics -> features -> residual model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydro_pgrl.utils import load_config, load_processed_dataset
from hydro_pgrl.pcc import RobustPCCMesh, generate_hybrid_field_grid, Ss_field_from_lnK
from hydro_pgrl.features import VoronoiFeatureExtractor
from hydro_pgrl.physics import (
    Experiment,
    build_bc_pack,
    assemble_matrices,
    solve_system_direct,
    calculate_dpl_parameters_robust,
)
from hydro_pgrl.models import ResidualLagModel


def sample_params(cfg: dict, rng: np.random.Generator, n: int) -> np.ndarray:
    bounds = cfg["param_bounds"]
    mu_k = rng.uniform(bounds["mu_lnK"][0], bounds["mu_lnK"][1], size=n)
    mu_ss = rng.uniform(bounds["mu_lnSs"][0], bounds["mu_lnSs"][1], size=n)
    sig_k = rng.uniform(bounds["sigma_lnK"][0], bounds["sigma_lnK"][1], size=n)
    corr_k = rng.uniform(bounds["corr_len_K"][0], bounds["corr_len_K"][1], size=n)
    sig_ss = rng.uniform(bounds["sigma_lnSs"][0], bounds["sigma_lnSs"][1], size=n)
    corr_ss = rng.uniform(bounds["corr_len_Ss"][0], bounds["corr_len_Ss"][1], size=n)
    rho = rng.uniform(bounds["rho"][0], bounds["rho"][1], size=n)
    mt_float = rng.uniform(bounds["model_type_float"][0], bounds["model_type_float"][1], size=n)
    return np.column_stack([mu_k, mu_ss, sig_k, corr_k, sig_ss, corr_ss, rho, mt_float])


def build_demo_dataset(cfg: dict) -> pd.DataFrame:
    demo_cfg = cfg["demo"]
    pcc_cfg = cfg["pcc"]
    sim_cfg = cfg["simulation"]

    rng = np.random.default_rng(int(demo_cfg["seed"]))

    mesh = RobustPCCMesh(
        pcc_cfg["Lx"],
        pcc_cfg["Ly"],
        demo_cfg["n_cells"],
        pcc_cfg["fixed_mesh_seed"],
        nx_grid=pcc_cfg["nx_grid"],
        ny_grid=pcc_cfg["ny_grid"],
    )

    exp_cfg = cfg["bc"]["experiments"][0]
    exp = Experiment(
        name=exp_cfg["name"],
        T=exp_cfg["T"],
        dt=exp_cfg["dt"],
        hL_mean=cfg["bc"]["hL_mean"],
        hL_span=cfg["bc"]["hL_span"],
        hR_const=cfg["bc"]["hR_const"],
        K_pe=exp_cfg["K_pe"],
    )
    bc = build_bc_pack(exp, Lx=mesh.Lx, fmax_factor=sim_cfg["fmax_factor"])

    rows = []
    params = sample_params(cfg, rng, int(demo_cfg["n_samples"]))

    for idx, p in enumerate(params):
        try:
            mu_k, mu_ss, sig_k, corr_k, sig_ss, corr_ss, rho, mt_float = p

            lnK = generate_hybrid_field_grid(
                pcc_cfg["nx_grid"],
                pcc_cfg["ny_grid"],
                mesh.dx_grid,
                mu_k,
                sig_k,
                corr_k,
                mt_float,
                rng,
                use_float32_fields=pcc_cfg["use_float32_fields"],
            )
            Ss_grid = Ss_field_from_lnK(
                lnK,
                mu_ss,
                sig_ss,
                corr_ss,
                mesh.dx_grid,
                rho,
                rng,
                use_float32_fields=pcc_cfg["use_float32_fields"],
            )

            K_pcc = np.exp(lnK)[mesh.grid_iy, mesh.grid_ix].astype(np.float64)
            Ss_pcc = Ss_grid[mesh.grid_iy, mesh.grid_ix].astype(np.float64)

            extractor = VoronoiFeatureExtractor(mesh, K_pcc, Ss_pcc)
            feats = {}
            feats.update(extractor.get_percolation_features())
            feats.update(extractor.get_entropy_features())
            feats.update(extractor.get_morphology_features())

            A, idx_L, Tb_L, idx_R, Tb_R = assemble_matrices(mesh, K_pcc)
            if len(idx_L) == 0 or len(idx_R) == 0:
                continue

            q0, h0, q_hat, target_idx = solve_system_direct(
                mesh,
                A,
                idx_L,
                Tb_L,
                idx_R,
                Tb_R,
                Ss_pcc,
                bc,
                target_idx=sim_cfg["target_freq_index"],
                penalty=sim_cfg["penalty"],
            )
            feats.update(extractor.get_energy_features(q0, h0))

            freq_w = 2 * np.pi * bc["freqs"][target_idx]
            dH = bc["hL_mean"] - bc["hR_const"]
            grad_h_static = dH / mesh.Lx
            grad_phase_in = bc["phases"][target_idx]
            grad_h_complex = (bc["tone_amp"] / mesh.Lx) * np.exp(1j * grad_phase_in)

            tau_q, tau_h = calculate_dpl_parameters_robust(
                q0,
                q_hat[target_idx],
                grad_h_static,
                grad_h_complex,
                freq_w,
                darcy_trap_eps=cfg["dpl"]["darcy_trap_eps"],
            )

            feats["Geostat_Mu_K"] = float(mu_k)
            feats["Geostat_Mu_Ss"] = float(mu_ss)
            feats["Geostat_Rho"] = float(rho)
            feats["Target_LogTau_q"] = float(np.log10(tau_q + 1e-15))
            feats["Target_LogTau_h"] = float(np.log10(tau_h + 1e-15))

            rows.append(feats)
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No valid samples generated in demo pipeline.")

    df["Phys_Theory"] = np.log10(df["Geostat_Mu_Ss"] / (df["Geostat_Mu_K"] + 1e-12))
    return df


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    df = load_processed_dataset(cfg)
    df = df.dropna(subset=["Target_LogTau_q", "Target_LogTau_h"]).copy()
    df = df[(df["Target_LogTau_q"] > -2.0) & (df["Target_LogTau_h"] > -2.0)]
    df["Phys_Theory"] = np.log10(df["Geostat_Mu_Ss"] / (df["Geostat_Mu_K"] + 1e-12))

    feature_cols = ResidualLagModel.infer_feature_columns(df)
    print(f"Training XGBoost model using features: {feature_cols}")

    model_cfg = cfg["pgrl"]["tau_q_params"]
    model = ResidualLagModel(cutoff_n=cfg["pgrl"]["cutoff_n"], params=model_cfg, seed=cfg["pgrl"]["seed"])
    r2 = model.fit(df, target_col="Target_LogTau_q", feature_cols=feature_cols)

    print(f"Dataset size: {len(df)}")
    print(f"ResidualLagModel R2 (tau_q): {r2:.3f}")


if __name__ == "__main__":
    main()
