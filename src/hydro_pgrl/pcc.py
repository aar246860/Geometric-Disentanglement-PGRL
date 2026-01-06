"""Voronoi PCC mesh generation and random field synthesis."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi


def _model_type_from_float(model_type_float: float) -> int:
    if model_type_float < 0.3333:
        return 0
    if model_type_float < 0.6666:
        return 1
    return 2


class RobustPCCMesh:
    """Voronoi-based PCC mesh with boundary-aware connectivity."""

    def __init__(self, Lx: float, Ly: float, n_cells: int, seed: int, nx_grid: int = 401, ny_grid: int = 401):
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.n_cells = int(n_cells)
        self.nx_grid = int(nx_grid)
        self.ny_grid = int(ny_grid)
        self.dx_grid = self.Lx / (self.nx_grid - 1)
        self.dy_grid = self.Ly / (self.ny_grid - 1)

        self.rng = np.random.default_rng(int(seed))
        self._generate_bounded_voronoi()
        self._precompute_point_to_grid_indices()

    def _generate_bounded_voronoi(self) -> None:
        points = self.rng.random((self.n_cells, 2))
        points[:, 0] *= self.Lx
        points[:, 1] *= self.Ly

        ghosts_list = [
            np.column_stack((-points[:, 0], points[:, 1])),
            np.column_stack((2 * self.Lx - points[:, 0], points[:, 1])),
            np.column_stack((points[:, 0], -points[:, 1])),
            np.column_stack((points[:, 0], 2 * self.Ly - points[:, 1])),
        ]
        all_points = np.vstack([points] + ghosts_list)
        vor = Voronoi(all_points)

        self.n = self.n_cells
        self.points = points.astype(np.float64)
        self.areas = np.zeros(self.n, dtype=np.float64)

        internal_i, internal_j, internal_len, internal_dist = [], [], [], []
        b_i, b_type, b_len, b_dist = [], [], [], []

        for i in range(self.n):
            region_idx = vor.point_region[i]
            region_verts_idx = vor.regions[region_idx]
            if -1 in region_verts_idx or len(region_verts_idx) == 0:
                continue
            verts = vor.vertices[region_verts_idx]
            center = verts.mean(axis=0)
            angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
            verts = verts[np.argsort(angles)]
            self.areas[i] = 0.5 * np.abs(
                np.dot(verts[:, 0], np.roll(verts[:, 1], 1))
                - np.dot(verts[:, 1], np.roll(verts[:, 0], 1))
            )

        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            if p1 >= self.n and p2 >= self.n:
                continue
            if v1 < 0 or v2 < 0:
                continue

            face_len = float(np.linalg.norm(vor.vertices[v1] - vor.vertices[v2]))
            dist = float(np.linalg.norm(all_points[p1] - all_points[p2]))
            if dist < 1e-12:
                continue

            if p1 < self.n and p2 < self.n:
                internal_i.append(p1)
                internal_j.append(p2)
                internal_len.append(face_len)
                internal_dist.append(dist)
            else:
                p_in = p1 if p1 < self.n else p2
                p_out = p2 if p1 < self.n else p1
                idx_offset = p_out - self.n
                if idx_offset < self.n:
                    bc_type = 0
                elif idx_offset < 2 * self.n:
                    bc_type = 1
                elif idx_offset < 3 * self.n:
                    bc_type = 2
                else:
                    bc_type = 3
                if bc_type in (0, 1):
                    b_i.append(p_in)
                    b_type.append(bc_type)
                    b_len.append(face_len)
                    b_dist.append(dist / 2.0)

        self.int_i = np.asarray(internal_i, dtype=np.int32)
        self.int_j = np.asarray(internal_j, dtype=np.int32)
        self.int_len = np.asarray(internal_len, dtype=np.float64)
        self.int_dist = np.asarray(internal_dist, dtype=np.float64)

        self.b_i = np.asarray(b_i, dtype=np.int32)
        self.b_type = np.asarray(b_type, dtype=np.int8)
        self.b_len = np.asarray(b_len, dtype=np.float64)
        self.b_dist = np.asarray(b_dist, dtype=np.float64)

    def _precompute_point_to_grid_indices(self) -> None:
        ix = np.rint(self.points[:, 0] / self.dx_grid).astype(np.int32)
        iy = np.rint(self.points[:, 1] / self.dy_grid).astype(np.int32)
        ix = np.clip(ix, 0, self.nx_grid - 1)
        iy = np.clip(iy, 0, self.ny_grid - 1)
        self.grid_ix = ix
        self.grid_iy = iy


def generate_hybrid_field_grid(
    nx: int,
    ny: int,
    dx: float,
    mu_ln: float,
    sigma_ln: float,
    corr_len: float,
    model_type_float: float,
    rng: np.random.Generator,
    use_float32_fields: bool = True,
) -> np.ndarray:
    model_type = _model_type_from_float(model_type_float)

    dtype = np.float32 if use_float32_fields else np.float64
    z = rng.normal(0.0, 1.0, size=(ny, nx)).astype(dtype)
    sigma_grid = max(1e-6, corr_len / dx)

    if model_type == 0:
        field = gaussian_filter(z, sigma=sigma_grid, mode="reflect")
        field = (field - field.mean()) / (field.std() + 1e-12)
        return (field * sigma_ln + mu_ln).astype(dtype)

    if model_type == 1:
        field = gaussian_filter(z, sigma=sigma_grid, mode="reflect")
        y_vals = (field - field.mean()) / (field.std() + 1e-12)
        w_vals = -np.abs(y_vals)
        w_vals = (w_vals - w_vals.mean()) / (w_vals.std() + 1e-12)
        return (w_vals * sigma_ln + mu_ln).astype(dtype)

    field = (np.zeros((ny, nx), dtype=dtype) + (mu_ln - sigma_ln)).astype(dtype)
    n_objects = int(rng.integers(10, 50))
    base_r = corr_len / dx
    y_idx, x_idx = np.ogrid[:ny, :nx]
    for _ in range(n_objects):
        cx, cy = int(rng.integers(0, nx)), int(rng.integers(0, ny))
        r = base_r * float(rng.uniform(0.5, 1.5))
        mask = ((x_idx - cx) ** 2 + (y_idx - cy) ** 2) <= r**2
        field[mask] = mu_ln + sigma_ln
    field += rng.normal(0, 0.1 * sigma_ln, (ny, nx)).astype(dtype)
    return field


def Ss_field_from_lnK(
    lnK: np.ndarray,
    mu_ss: float,
    sig_ss: float,
    corr_ss: float,
    dx: float,
    rho: float,
    rng: np.random.Generator,
    use_float32_fields: bool = True,
) -> np.ndarray:
    ny, nx = lnK.shape
    std_lnk = lnK.std()
    if std_lnk > 1e-12:
        z_str = (lnK - lnK.mean()) / std_lnk
    else:
        z_str = np.zeros_like(lnK)

    dtype = np.float32 if use_float32_fields else np.float64
    z_raw = rng.normal(0, 1, (ny, nx)).astype(dtype)
    sig_ss_g = max(1e-6, corr_ss / dx)
    z_corr = gaussian_filter(z_raw, sig_ss_g, mode="reflect")
    z_n = (z_corr - z_corr.mean()) / (z_corr.std() + 1e-12)

    z_ss = rho * z_str + np.sqrt(max(0.0, 1.0 - rho**2)) * z_n
    return np.exp(z_ss * sig_ss + mu_ss).astype(dtype)
