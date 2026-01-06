"""Geometric feature extraction on PCC meshes."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, diags
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import eigsh


class VoronoiFeatureExtractor:
    """Compute topology, morphology, and energy features on PCC meshes."""

    def __init__(self, mesh, K_vals: np.ndarray, Ss_vals: np.ndarray):
        self.mesh = mesh
        self.K = K_vals.astype(np.float64)
        self.Ss = Ss_vals.astype(np.float64)
        self.N = mesh.n

        self.i = mesh.int_i
        self.j = mesh.int_j

        ki, kj = self.K[self.i], self.K[self.j]
        self.Th = (2 * ki * kj) / (ki + kj + 1e-30)
        self.T_vals = self.Th * (mesh.int_len / mesh.int_dist)

        self.Adj = csc_matrix((self.T_vals, (self.i, self.j)), shape=(self.N, self.N))
        self.Adj = self.Adj + self.Adj.T

        ones = np.ones_like(self.T_vals)
        self.Adj_Bin = csc_matrix((ones, (self.i, self.j)), shape=(self.N, self.N))
        self.Adj_Bin = self.Adj_Bin + self.Adj_Bin.T

    def get_percolation_features(self) -> dict:
        """Flow paths, tortuosity, and connectivity metrics."""
        features = {}

        features["Phys_Perc_Kc"] = np.percentile(self.K, 50)
        features["Phys_Perc_MaxK"] = np.max(self.K)

        dist_weights = self.mesh.int_dist / (self.Th + 1e-12)
        g_dist = csc_matrix((dist_weights, (self.i, self.j)), shape=(self.N, self.N))

        start_node = np.argmin(self.mesh.points[:, 0])
        end_node = np.argmax(self.mesh.points[:, 0])

        try:
            d_path = shortest_path(g_dist, method="Dijkstra", indices=start_node)[end_node]
            features["Phys_Perc_Tortuosity"] = d_path / (self.mesh.Lx + 1e-12)
        except Exception:
            features["Phys_Perc_Tortuosity"] = 1.0

        threshold = features["Phys_Perc_Kc"]
        mask_high = self.K > threshold
        high_k_indices = np.where(mask_high)[0]

        if len(high_k_indices) > 0:
            edge_mask = mask_high[self.i] & mask_high[self.j]
            row = self.i[edge_mask]
            col = self.j[edge_mask]
            data = np.ones(len(row))
            adj_sub = csc_matrix((data, (row, col)), shape=(self.N, self.N))

            n_components, labels = connected_components(adj_sub, directed=False)

            high_k_labels = labels[mask_high]
            if len(high_k_labels) > 0:
                _, c2 = np.unique(high_k_labels, return_counts=True)
                num = np.sum(c2**2)
                den = np.sum(c2)
                features["Phys_Perc_ClusterSize"] = (num / den) / self.N
            else:
                features["Phys_Perc_ClusterSize"] = 0.0

            # TODO: Scientific Check - manuscript references Betti-1; code uses Betti-0.
            features["Phys_Perc_Betti0"] = n_components

            degrees = np.array(adj_sub.sum(axis=1)).flatten()
            n_backbone = np.sum((degrees > 1) & mask_high)
            n_total_high = np.sum(mask_high)
            features["Phys_Perc_Backbone"] = n_backbone / (n_total_high + 1e-12)
        else:
            features["Phys_Perc_ClusterSize"] = 0.0
            features["Phys_Perc_Betti0"] = 0.0
            features["Phys_Perc_Backbone"] = 0.0

        return features

    def get_spectral_features(self) -> dict:
        """Spectral graph metrics (connectivity stiffness)."""
        features = {}

        degrees = np.array(self.Adj.sum(axis=1)).flatten()
        deg_inv_sqrt = 1.0 / np.sqrt(degrees + 1e-12)
        d_inv_sqrt = diags(deg_inv_sqrt)
        lap = sp.eye(self.N) - d_inv_sqrt @ self.Adj @ d_inv_sqrt

        try:
            vals = eigsh(lap, k=3, sigma=-0.01, which="LM", return_eigenvectors=False)
            vals = np.sort(vals)

            lambda_2 = vals[1] if len(vals) > 1 else 0.0
            lambda_3 = vals[2] if len(vals) > 2 else 0.0

            features["Phys_Graph_Fiedler"] = lambda_2
            features["Phys_Graph_Gap"] = lambda_3 / (lambda_2 + 1e-12)
            features["Phys_Graph_Resistance"] = np.sum(1.0 / (vals[1:] + 1e-12))
        except Exception:
            features["Phys_Graph_Fiedler"] = 0.0
            features["Phys_Graph_Gap"] = 0.0
            features["Phys_Graph_Resistance"] = 0.0

        return features

    def get_entropy_features(self) -> dict:
        """Information-theoretic descriptors of heterogeneity."""
        features = {}

        bins = np.linspace(np.min(self.K), np.max(self.K), 6)
        digitized = np.digitize(self.K, bins) - 1

        pairs_code = digitized[self.i] * 6 + digitized[self.j]
        counts = np.bincount(pairs_code, minlength=36)
        probs = counts / (np.sum(counts) + 1e-12)
        probs = probs[probs > 0]

        features["Phys_Info_Entropy"] = -np.sum(probs * np.log2(probs))

        mask_left = self.mesh.points[:, 0] < (0.2 * self.mesh.Lx)
        mask_right = self.mesh.points[:, 0] > (0.8 * self.mesh.Lx)

        h_left, _ = np.histogram(self.K[mask_left], bins=bins, density=True)
        h_right, _ = np.histogram(self.K[mask_right], bins=bins, density=True)

        p = h_left + 1e-12
        q = h_right + 1e-12
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)
        jsd = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

        features["Phys_Info_Div"] = jsd
        return features

    def get_morphology_features(self) -> dict:
        """Morphology and lacunarity proxies."""
        features = {}

        threshold = np.median(self.K)
        is_high = self.K > threshold
        boundary_mask = is_high[self.i] ^ is_high[self.j]
        features["Phys_Morph_Perimeter"] = np.sum(self.mesh.int_len[boundary_mask]) / self.mesh.Lx

        local_mass = self.Adj_Bin @ self.Ss
        mean_mass = np.mean(local_mass)
        var_mass = np.var(local_mass)
        features["Phys_Morph_Lacunarity"] = (var_mass / (mean_mass**2 + 1e-12)) + 1.0

        return features

    def get_energy_features(self, q0_scalar: float, h_vector: np.ndarray) -> dict:
        """Energy and dissipation localization metrics."""
        features = {}
        features["Phys_Energy_Power"] = np.log(np.abs(q0_scalar) + 1e-12)

        dh = h_vector[self.i] - h_vector[self.j]
        edge_power = self.T_vals * (dh**2)
        total = np.sum(edge_power) + 1e-12
        weights = edge_power / total
        ipr = np.sum(weights**2)
        features["Phys_Energy_Localization"] = 1.0 / (ipr * len(weights) + 1e-12)
        return features
