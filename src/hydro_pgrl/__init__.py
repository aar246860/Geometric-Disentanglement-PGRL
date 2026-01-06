"""Hydro PGRL package for PCC generation, physics solvers, feature extraction, and ML."""

from .pcc import RobustPCCMesh, generate_hybrid_field_grid, Ss_field_from_lnK
from .features import VoronoiFeatureExtractor
from .physics import (
    Experiment,
    build_bc_pack,
    assemble_matrices,
    solve_system_direct,
    solve_system_extended,
    solve_multisine_freqdomain,
    fit_dpl_parameters,
    calculate_dpl_parameters_robust,
)
from .models import ResidualLagModel
from .analysis import explain_with_shap_kernel, explain_with_shap_tree, sindy_discover_structure
from .utils import load_config
