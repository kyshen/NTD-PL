"""
NTD-PL（非线性张量分解-多项式）模块

一体化的多项式非线性张量分解库。

核心组件：
- core: 主求解器 (ntdpl 函数)
- utils: 工具函数 (Adam, 多项式, 误差等)
- beta: Beta 系数更新 (moment equations, ridge regression)
- normalizer: 归一化和缩放

使用示例：
---------
from src.ntdpl import ntdpl

core, factors, beta = ntdpl(
    X=tensor,
    rank=(10, 10, 10),
    init_n_iter_max=10,
    p_max=3,
    n_iter_max=100,
    use_continuation=True,
    factor_normalize=True,
    lr_core=0.001,
    lr_factors=0.001,
    lambda_core=0.0,
    lambda_factors=0.0,
    lambda_beta=0.0,
    beta_update_method='moments_normal_eq',
)
"""

from .core import ntdpl, ntdpl_optimized, init_ntdpl_factors
from .links import ChebyshevLink, LinearSplineLink, PowerLink, RBFLink, ScalarLink, beta_update_link, make_link
from .utils import (
    adam_init,
    adam_step,
    poly,
    deriv,
    rmse_on_target,
    beta_to_dict,
    build_continuation_schedule,
)
from .beta import beta_update, beta_update_moments_normal_eq, beta_update_ridge_lstsq
from .normalizer import normalize_tucker, mode_scale_core

__all__ = [
    # Core
    "ntdpl",
    "ntdpl_optimized",
    "init_ntdpl_factors",
    # Utils
    "adam_init",
    "adam_step",
    "poly",
    "deriv",
    "rmse_on_target",
    "beta_to_dict",
    "build_continuation_schedule",
    "ScalarLink",
    "PowerLink",
    "ChebyshevLink",
    "RBFLink",
    "LinearSplineLink",
    "make_link",
    "beta_update_link",
    # Beta
    "beta_update",
    "beta_update_moments_normal_eq",
    "beta_update_ridge_lstsq",
    # Normalizer
    "normalize_tucker",
    "mode_scale_core",
]
