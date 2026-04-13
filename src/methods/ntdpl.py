"""
NTDPL 分解方法 - 与 BaseDecomposeMethod 兼容的接口

此模块提供了 NTDPLDecomposition 类，包装了来自 src.ntdpl 模块的核心函数。
核心的 ntdpl 求解器和所有辅助函数已分解到 src/ntdpl/ 子模块中。

使用示例：
---------
from src.methods.ntdpl import NTDPLDecomposition

method = NTDPLDecomposition(
    rank=(10, 10, 10),
    n_iter_max=100,
    p_max=3,
    ...
)
method.fit(data, mask, logcallback)
tensor = method.reconstruct()
"""

from typing import Any, Dict, List, Optional
import numpy as np
from tensorly.tucker_tensor import tucker_to_tensor

from src.methods.base import BaseDecomposeMethod
from src.types import Tensor
from src.utils.completion_ops import mask_to_bool, mean_fill_missing

# 从分解的 ntdpl 模块导入
from src.ntdpl import poly, ntdpl as ntdpl_base, ntdpl_optimized


# ============================================================
# Shared State Mixin
# ============================================================

class _NTDPLStateMixin:
    """
    NTDPL 分解的状态管理混合类
    
    管理核张量、因子和多项式系数的状态。
    """
    fitted: bool
    core: np.ndarray
    factors: List[np.ndarray]
    beta: np.ndarray
    fit_mask: Optional[np.ndarray]

    def reconstruct(self) -> Tensor:
        """
        重构张量
        
        Returns:
        --------
        tensor : Tensor
            重构的张量
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before reconstruction.")
        S = tucker_to_tensor((self.core, self.factors))
        # `core_optim` converts solved coefficients back to the original
        # power basis of S before returning beta, so inference must use S
        # directly here to stay compatible with v0 behavior.
        dense = np.array(poly(np.asarray(S, dtype=np.float32), self.beta), dtype=np.float32)
        return Tensor(shape=dense.shape, dense=dense)

    def get_num_params(self) -> int:
        """
        获取模型参数总数
        
        Returns:
        --------
        num_params : int
            参数总数（核张量 + 因子 + beta 系数）
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before reconstruction.")
        total = int(np.prod(self.core.shape))
        for f in self.factors:
            total += int(np.prod(f.shape))
        total += len(self.beta)
        return total

    def get_state_dict(self) -> Dict[str, Any]:
        """
        获取模型状态字典
        
        Returns:
        --------
        state_dict : dict
            包含 core, factors, beta, fitted 的字典
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before exporting state.")
        state = {
            "core": np.array(self.core, copy=True),
            "factors": [np.array(f, copy=True) for f in self.factors],
            "beta": np.array(self.beta, copy=True),
            "fitted": self.fitted,
        }
        if self.fit_mask is not None:
            state["fit_mask"] = np.array(self.fit_mask, copy=True)
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        加载模型状态字典
        
        Parameters:
        -----------
        state_dict : dict
            包含 core, factors, beta, fitted 的字典
        """
        self.core = np.array(state_dict["core"], dtype=np.float32, copy=True)
        self.factors = [np.array(f, dtype=np.float32, copy=True) for f in state_dict["factors"]]
        self.beta = np.array(state_dict["beta"], dtype=np.float32, copy=True)
        fit_mask = state_dict.get("fit_mask", None)
        self.fit_mask = None if fit_mask is None else np.array(fit_mask, dtype=bool, copy=True)
        self.fitted = bool(state_dict.get("fitted", True))


# ============================================================
# Public Method Class
# ============================================================

class NTDPLDecomposition(_NTDPLStateMixin, BaseDecomposeMethod):
    """
    NTD-PL（非线性张量分解-多项式）分解方法
    
    使用多项式非线性模型进行张量分解和完成。
    基础求解器位于 src.ntdpl 模块。
    """
    
    def __init__(self, **method_cfg: Any) -> None:
        """初始化 NTDPLDecomposition"""
        super().__init__()
        self.cfg = method_cfg
        self.fitted = False
        self.fit_mask = None

    def fit(self, data, mask, logcallback) -> None:
        """拟合 NTDPL 模型"""
        X_obs = np.array(data.dense, dtype=np.float32)
        mask_bool = mask_to_bool(mask, X_obs.shape) if mask is not None else None
        X_fit = mean_fill_missing(X_obs, mask_bool) if mask_bool is not None else X_obs
        return_history = bool(logcallback.log_level >= 1)

        cfg_params = {k: self.cfg[k] for k in [
            'rank', 'init_n_iter_max', 'p_max', 'allow_constant_term', 'n_iter_max', 'use_continuation',
            'factor_normalize', 'lr_core', 'lr_factors',
            'lambda_core', 'lambda_factors', 'lambda_beta', 'beta_update_method',
            'init', 'random_state', 'beta_update_interval'
        ]}

        solver_variant = str(self.cfg.get("solver_variant", "optimized")).lower()
        if solver_variant == "optimized":
            solver = ntdpl_optimized
            cfg_params["stable_beta_update"] = bool(self.cfg.get("stable_beta_update", True))
            cfg_params["beta_update_stage"] = str(self.cfg.get("beta_update_stage", "before_grad"))
        elif solver_variant == "base":
            solver = ntdpl_base
        else:
            raise ValueError(f"Unknown NTD-PL solver variant: {solver_variant}")

        ret = solver(
            X=X_fit,
            mask=mask_bool,
            return_history=return_history,
            **cfg_params
        )

        if return_history:
            (core, factors, beta), history = ret  # type: ignore
            for item in history:
                if isinstance(item, dict):
                    logcallback.addlog(item)
        else:
            core, factors, beta = ret  # type: ignore

        self.core = np.array(core, dtype=np.float32, copy=True)
        self.factors = [np.array(f, dtype=np.float32, copy=True) for f in factors]
        self.beta = np.array(beta, dtype=np.float32, copy=True)
        self.fit_mask = None if mask_bool is None else np.array(mask_bool, dtype=bool, copy=True)
        self.fitted = True
        return None
