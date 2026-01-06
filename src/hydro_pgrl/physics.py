"""Physics solvers, boundary conditions, and DPL fitting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix, diags


@dataclass(frozen=True)
class Experiment:
    name: str
    T: float
    dt: float
    hL_mean: float = 6.0
    hL_span: float = 2.0
    hR_const: float = 0.0
    K_pe: int = 25


def safe_fmax(dt: float, fmax_factor: float = 0.1) -> float:
    f_nyq = 0.5 / dt
    fmax = fmax_factor / dt
    if fmax >= 0.45 * f_nyq:
        fmax = 0.45 * f_nyq
    return float(fmax)


def build_time_grid(T: float, dt: float) -> Tuple[int, np.ndarray]:
    nt = int(T / dt) + 1
    t_all = np.linspace(0.0, T, nt, dtype=np.float64)
    return nt, t_all


def multisine_spec(T: float, dt: float, K_pe: int, fmax_factor: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    f_min = 2.0 / T
    f_max = safe_fmax(dt, fmax_factor=fmax_factor)
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), K_pe).astype(np.float64)
    k = np.arange(1, K_pe + 1, dtype=np.float64)
    phases = np.pi * (k - 1.0) * (k - 1.0) / float(K_pe)
    return freqs, phases


def multisine_bc_from_spec(t_all: np.ndarray, mean: float, span: float, freqs: np.ndarray, phases: np.ndarray) -> np.ndarray:
    k_pe = len(freqs)
    amps = (span / np.sqrt(k_pe)) * np.ones_like(freqs)
    tt = t_all[:, None]
    ff = freqs[None, :]
    ph = phases[None, :]
    h = mean + np.sum(amps[None, :] * np.sin(2.0 * np.pi * ff * tt + ph), axis=1)
    return h.astype(np.float64)


def build_bc_pack(exp: Experiment, Lx: float, fmax_factor: float = 0.1) -> Dict:
    nt, t_all = build_time_grid(exp.T, exp.dt)
    freqs, phases = multisine_spec(exp.T, exp.dt, exp.K_pe, fmax_factor=fmax_factor)

    h_L_all = multisine_bc_from_spec(t_all, exp.hL_mean, exp.hL_span, freqs, phases)
    h_R_all = np.full_like(h_L_all, float(exp.hR_const), dtype=np.float64)

    grad_template = (h_L_all[1:] - h_R_all[1:]) / float(Lx)
    tone_amp = float(exp.hL_span / np.sqrt(exp.K_pe))

    return {
        "name": exp.name,
        "T": float(exp.T),
        "dt": float(exp.dt),
        "nt": int(nt),
        "t_all": t_all,
        "h_L_all": h_L_all,
        "h_R_all": h_R_all,
        "grad_template": grad_template.astype(np.float32),
        "freqs": freqs,
        "phases": phases,
        "tone_amp": tone_amp,
        "hL_mean": float(exp.hL_mean),
        "hR_const": float(exp.hR_const),
    }


def assemble_matrices(mesh, K_pcc: np.ndarray):
    i, j = mesh.int_i, mesh.int_j
    ki, kj = K_pcc[i], K_pcc[j]
    th = (2 * ki * kj) / (ki + kj + 1e-30)
    t_vals = th * (mesh.int_len / mesh.int_dist)

    rows = np.concatenate([i, j, i, j])
    cols = np.concatenate([j, i, i, j])
    data = np.concatenate([-t_vals, -t_vals, t_vals, t_vals])

    bi = mesh.b_i
    tb = K_pcc[bi] * (mesh.b_len / mesh.b_dist)

    rows = np.concatenate([rows, bi])
    cols = np.concatenate([cols, bi])
    data = np.concatenate([data, tb])

    A = csc_matrix((data, (rows, cols)), shape=(mesh.n, mesh.n))
    return A, bi[(mesh.b_type == 0)], tb[(mesh.b_type == 0)], bi[(mesh.b_type == 1)], tb[(mesh.b_type == 1)]


def solve_system_direct(
    mesh,
    A: csc_matrix,
    idx_L: np.ndarray,
    Tb_L: np.ndarray,
    idx_R: np.ndarray,
    Tb_R: np.ndarray,
    Ss: np.ndarray,
    bc: Dict,
    target_idx: int = 2,
    penalty: float = 1e15,
):
    """Direct solver using a penalty method for Dirichlet boundaries."""
    n_cells = A.shape[0]
    freqs = bc["freqs"]
    phases = bc["phases"]
    tone_amp = bc["tone_amp"]

    b0 = np.zeros(n_cells)
    A_pen = A.copy()

    hL = bc["hL_mean"]
    A_pen[idx_L, idx_L] += penalty
    b0[idx_L] += penalty * hL

    hR = bc["hR_const"]
    A_pen[idx_R, idx_R] += penalty
    b0[idx_R] += penalty * hR

    h0_sol = spla.spsolve(A_pen, b0)

    q_left = np.sum(penalty * (hL - h0_sol[idx_L]))
    q0 = q_left / mesh.Ly

    target_idx = int(target_idx)
    w = 2 * np.pi * freqs[target_idx]

    vol = (mesh.Lx * mesh.Ly) / n_cells
    m_diag = Ss * vol
    M = sp.diags(m_diag)

    z_sys = A_pen + 1j * w * M

    b_hat = np.zeros(n_cells, dtype=complex)
    h_osc_complex = tone_amp * np.exp(1j * phases[target_idx])
    b_hat[idx_L] += penalty * h_osc_complex

    h_hat = spla.spsolve(z_sys, b_hat)

    q_hat_left = np.sum(penalty * (h_osc_complex - h_hat[idx_L]))
    q_hat_val = q_hat_left / mesh.Ly

    q_hat_full = np.full(len(freqs), np.nan, dtype=complex)
    q_hat_full[target_idx] = q_hat_val

    return q0, h0_sol, q_hat_full, target_idx


def solve_system_extended(
    mesh,
    A: csc_matrix,
    idx_L: np.ndarray,
    Tb_L: np.ndarray,
    idx_R: np.ndarray,
    Tb_R: np.ndarray,
    Ss_pcc: np.ndarray,
    bc: Dict,
    target_idx: int = 10,
):
    """GMRES-based solver that targets a single frequency."""
    hL0, hR0 = bc["hL_mean"], bc["hR_const"]
    rhs0 = np.zeros(mesh.n)
    if idx_L.size:
        rhs0[idx_L] += Tb_L * hL0
    if idx_R.size:
        rhs0[idx_R] += Tb_R * hR0

    solve_A = spla.splu(A)
    h0 = solve_A.solve(rhs0)

    q_in0 = np.sum(Tb_L * (hL0 - h0[idx_L])) / mesh.Ly if idx_L.size else 0.0
    q_out0 = np.sum(Tb_R * (h0[idx_R] - hR0)) / mesh.Ly if idx_R.size else 0.0
    q0 = 0.5 * (q_in0 + q_out0)

    freqs = bc["freqs"]
    q_hat = np.zeros(len(freqs), dtype=np.complex128)

    volSs = mesh.areas * Ss_pcc
    M_vec = volSs
    tone_amp = bc["tone_amp"]
    phases = bc["phases"]

    def precond_matvec(x):
        return solve_A.solve(x.real) + 1j * solve_A.solve(x.imag)

    m_pre = spla.LinearOperator(A.shape, matvec=precond_matvec, dtype=np.complex128)

    k = int(target_idx)
    w = 2 * np.pi * freqs[k]

    def matvec(x):
        return A.dot(x) + (1j * w) * (M_vec * x)

    op = spla.LinearOperator(A.shape, matvec=matvec, dtype=np.complex128)

    h_L_val = tone_amp * np.exp(1j * (phases[k] - 0.5 * np.pi))
    rhs = np.zeros(mesh.n, dtype=np.complex128)
    if idx_L.size:
        rhs[idx_L] += Tb_L * h_L_val

    try:
        h_cx, info = spla.gmres(op, rhs, M=m_pre, rtol=1e-5, atol=1e-8, maxiter=500)
        if info == 0:
            q_in = np.sum(Tb_L * (h_L_val - h_cx[idx_L])) / mesh.Ly if idx_L.size else 0j
            q_out = np.sum(Tb_R * (h_cx[idx_R] - 0)) / mesh.Ly if idx_R.size else 0j
            q_hat[k] = 0.5 * (q_in + q_out)
        else:
            q_hat[k] = np.nan
    except Exception:
        q_hat[k] = np.nan

    return q0, h0, q_hat


def solve_multisine_freqdomain(
    mesh,
    A: csc_matrix,
    idx_L: np.ndarray,
    Tb_L: np.ndarray,
    idx_R: np.ndarray,
    Tb_R: np.ndarray,
    Ss_pcc: np.ndarray,
    bc: Dict,
):
    """Compute frequency-domain flux response and synthesize q(t)."""
    t = bc["t_all"][1:]
    freqs = bc["freqs"]
    phases = bc["phases"]
    tone_amp = float(bc["tone_amp"])
    hL0 = float(bc["hL_mean"])
    hR0 = float(bc["hR_const"])

    omega = 2.0 * np.pi * freqs
    volSs = mesh.areas * Ss_pcc
    M = diags(volSs, format="csc")

    rhs0 = np.zeros(mesh.n, dtype=np.float64)
    if idx_L.size > 0:
        rhs0[idx_L] += Tb_L * hL0
    if idx_R.size > 0:
        rhs0[idx_R] += Tb_R * hR0

    h0 = spla.spsolve(A, rhs0)

    if idx_L.size > 0:
        q_in0 = np.sum(Tb_L * (hL0 - h0[idx_L])) / mesh.Ly
    else:
        q_in0 = 0.0
    if idx_R.size > 0:
        q_out0 = np.sum(Tb_R * (h0[idx_R] - hR0)) / mesh.Ly
    else:
        q_out0 = 0.0
    q0 = 0.5 * (q_in0 + q_out0)

    h_L = tone_amp * np.exp(1j * (phases - 0.5 * np.pi))
    h_R = np.zeros_like(h_L)

    q_hat = np.empty(len(freqs), dtype=np.complex128)

    for k, w in enumerate(omega):
        system = A + (1j * w) * M
        solver = spla.factorized(system)

        rhs = np.zeros(mesh.n, dtype=np.complex128)
        if idx_L.size > 0:
            rhs[idx_L] += Tb_L * h_L[k]
        if idx_R.size > 0:
            rhs[idx_R] += Tb_R * h_R[k]

        h_hat = solver(rhs)

        if idx_L.size > 0:
            q_in = np.sum(Tb_L * (h_L[k] - h_hat[idx_L])) / mesh.Ly
        else:
            q_in = 0.0 + 0.0j
        if idx_R.size > 0:
            q_out = np.sum(Tb_R * (h_hat[idx_R] - h_R[k])) / mesh.Ly
        else:
            q_out = 0.0 + 0.0j

        q_hat[k] = 0.5 * (q_in + q_out)

    tt = t[:, None]
    ww = omega[None, :]
    q_ac = np.real(np.sum(q_hat[None, :] * np.exp(1j * ww * tt), axis=1))

    return (q0 + q_ac).astype(np.float32)


def build_design_and_pinv(t: np.ndarray, omega: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    n = t.size
    k = omega.size
    p = 1 + 2 * k
    M = np.empty((n, p), dtype=np.float64)
    M[:, 0] = 1.0
    for kk in range(k):
        wt = omega[kk] * t
        M[:, 1 + 2 * kk] = np.cos(wt)
        M[:, 1 + 2 * kk + 1] = np.sin(wt)
    G = M.T @ M
    G.flat[:: G.shape[0] + 1] += ridge
    return np.linalg.solve(G, M.T)


def batch_phasors(X: np.ndarray, pinv: np.ndarray) -> np.ndarray:
    coef = X @ pinv.T
    A = coef[:, 1::2]
    B = coef[:, 2::2]
    ph = A - 1j * B
    return ph


def solve_real_system(A: np.ndarray, y: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    A_big = np.vstack([A.real, A.imag])
    y_big = np.hstack([y.real, y.imag])
    AtA = A_big.T @ A_big + ridge * np.eye(A_big.shape[1])
    Aty = A_big.T @ y_big
    try:
        x = np.linalg.solve(AtA, Aty)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(AtA, Aty, rcond=None)[0]
    return x


def fit_dpl_parameters(w_tr: np.ndarray, H_tr: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Fit H(w) = (b0 + b1*s + b2*s^2) / (1 + a1*s + a2*s^2)."""
    iw = 1j * w_tr
    col1 = np.ones_like(H_tr)
    col2 = iw
    col3 = iw**2
    col4 = -H_tr * iw
    col5 = -H_tr * (iw**2)

    A = np.column_stack([col1, col2, col3, col4, col5])
    theta = solve_real_system(A, H_tr, ridge=ridge)
    return theta


def calculate_dpl_parameters_robust(
    q0: float,
    q_hat_val: complex,
    grad_h_static: float,
    grad_h_complex: complex,
    omega: float,
    darcy_trap_eps: float = 1e-6,
) -> Tuple[float, float]:
    """Compute DPL parameters with a Darcy trap for near-zero phase."""
    if abs(grad_h_static) < 1e-15 or abs(q0) < 1e-15:
        return np.nan, np.nan

    k_stat = q0 / grad_h_static
    k_star = q_hat_val / grad_h_complex

    z = k_star / k_stat
    zr, zi = np.real(z), np.imag(z)

    if abs(zi) < darcy_trap_eps:
        return 1e-12, 1e-12

    tau_q = (zr - 1.0) / (omega * zi)
    tau_h = (zi / omega) + (tau_q * zr)
    return tau_q, tau_h
