"""
NTDPL 核心求解器模块

包含主要的 ntdpl() 函数和因子初始化。
"""

from typing import Dict, List, Optional
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tenalg.core_tenalg import multi_mode_dot

from src.utils.completion_ops import mask_to_bool, mask_to_float
from .utils import adam_init, adam_step, poly, deriv, rmse_on_target, beta_to_dict, build_continuation_schedule
from .beta import beta_update
from .normalizer import normalize_tucker


# ============================================================
# Core Solver
# ============================================================

def ntdpl(
    X,
    rank,
    init_n_iter_max: int,
    p_max: int,
    allow_constant_term: bool,
    n_iter_max: int,
    use_continuation: bool,
    factor_normalize: bool,
    lr_core: float,
    lr_factors: float,
    lambda_core: float,
    lambda_factors: float,
    lambda_beta: float,
    beta_update_method: str,
    init: str,
    random_state: int,
    beta_update_interval: int,
    return_history: bool,
    mask: Optional[np.ndarray] = None,
):
    """
    NTD-PL 核心优化器
    
    对张量进行多项式非线性分解，基于 Tucker 分解。
    支持张量完成（缺失数据）。
    
    Parameters:
    -----------
    X : ndarray
        完整张量或观测张量（用于完成）。
        对于完成任务，缺失项可以任意填充（如0），
        `mask` 指示哪些项被观测。
    rank : tuple of int
        Tucker 分解的秩
    init_n_iter_max : int
        初始化的最大迭代数（用于 Tucker 初始化）
    p_max : int
        最大多项式度数
    n_iter_max : int
        总迭代数
    use_continuation : bool
        是否使用 continuation 策略（逐步增加 p）
    factor_normalize : bool
        是否每次迭代后对因子进行归一化
        是否将核张量的缩放吸收到 beta 中
    lr_core : float
        核张量的学习率
    lr_factors : float
        因子的学习率
    lambda_core : float
        核张量的 L2 正则化参数
    lambda_factors : float
        因子的 L2 正则化参数
    lambda_beta : float
        Beta 的 L2 正则化参数
    beta_update_method : str
        Beta 更新方法 ('moments_normal_eq' 或 'ridge_lstsq')
    mask : ndarray or None
        观测掩码，与 X 形状相同。1/True = 观测，0/False = 缺失。
    init : str
        初始化方法 ('tucker' 或 'random')
    random_state : int or None
        随机种子
    beta_update_interval : int
        每隔多少轮更新一次 beta（>=1）
        是否从输入张量 X 的统计信息初始化 beta0 和 beta1
        如果为 True，beta0 = mean(X)，beta1 = std(X)
    return_history : bool
        如果为 True，返回 ((core, factors, beta), history)，
        其中 history 是包含日志的字典列表。
    
    Returns:
    --------
    result : tuple
        (core, factors, beta) 或 ((core, factors, beta), history)
    """
    X = np.asarray(X, dtype=np.float32)
    mask_bool = mask_to_bool(mask, X.shape) if mask is not None else None
    mask_float = mask_to_float(mask, X.shape, dtype=X.dtype) if mask is not None else None

    if mask_float is None:
        fit_scale = np.float32(1.0 / X.size)
    else:
        n_obs = float(mask_float.sum())
        if n_obs <= 0:
            raise ValueError("`mask` contains no observed entries.")
        fit_scale = np.float32(1.0 / n_obs)

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------
    core, factors = init_ntdpl_factors(
        X=X,
        rank=rank,
        init=init,
        init_n_iter_max=init_n_iter_max,
        mask_float=mask_float,
        random_state=random_state,
    )
    if factor_normalize:
        core, factors = normalize_tucker(core, factors)

    if p_max < 1:
        raise ValueError("`p_max` must be >= 1.")
    if beta_update_interval < 1:
        raise ValueError("`beta_update_interval` must be >= 1.")

    # 初始化多项式度数和系数
    if use_continuation:
        p = 1
        beta = np.zeros(p + 1, dtype=np.float32)
        beta[0] = 0.0
        beta[1] = 1.0
        continuation_schedule = build_continuation_schedule(n_iter_max, p_max)
        continuation_idx = 0
    else:
        p = p_max
        beta = np.zeros(p + 1, dtype=np.float32)
        beta[0] = 0.0
        beta[1] = 1.0
        continuation_schedule = []
        continuation_idx = 0

    # 初始化 Adam 优化器状态
    st_core = adam_init(core.shape)
    st_factors = [adam_init(f.shape) for f in factors]

    history: List[Dict[str, float]] = []

    def _append_history() -> None:
        """记录当前状态到历史"""
        X_hat = poly(tucker_to_tensor((core, factors)), beta)
        err = rmse_on_target(X, X_hat, mask=mask_bool)
        record = {
            "p": int(p),
            "error": float(err),
            **beta_to_dict(beta, p_max),
        }
        if mask_bool is None:
            record["RMSE"] = float(err)
        else:
            record["RMSE_obs"] = float(err)
        history.append(record)

    if return_history:
        _append_history()

    # --------------------------------------------------------
    # Main Optimization Loop
    # --------------------------------------------------------
    N = X.ndim
    modes_all = list(range(N))

    for it in range(1, n_iter_max + 1):
        # Continuation 更新：逐步增加多项式度数
        if use_continuation:
            while continuation_idx < len(continuation_schedule) and it >= continuation_schedule[continuation_idx]:
                p += 1
                beta_new = np.zeros(p + 1, dtype=np.float32)
                beta_new[:p] = beta
                beta_new[p] = 0.0
                beta = beta_new
                continuation_idx += 1

        # Tucker 张量重构
        S = tucker_to_tensor((core, factors))
        
        # 多项式和导数计算
        Xhat = poly(S, beta)
        dfdS = deriv(S, beta)

        # 计算掩码残差（用于完成）或完整残差（用于分解）
        E = Xhat - X
        if mask_float is not None:
            E = E * mask_float
        E = E * fit_scale
        T = E * dfdS

        # 核张量梯度
        grad_core = multi_mode_dot(T, [f.T for f in factors], modes=modes_all)
        grad_core = grad_core.astype(np.float32) + lambda_core * core
        adam_step(core, grad_core, st_core, b1=0.9, b2=0.999, lr=lr_core, eps=1e-8)

        # 因子的梯度
        for n in range(N):
            other_modes = [k for k in range(N) if k != n]
            M = multi_mode_dot(core, [factors[k] for k in other_modes], modes=other_modes)
            Z = tl.unfold(M, mode=n)
            Tn = tl.unfold(T, mode=n)
            grad_A = np.dot(Tn, Z.T)
            grad_A = grad_A.astype(np.float32) + lambda_factors * factors[n]
            adam_step(factors[n], grad_A, st_factors[n], b1=0.9, b2=0.999, lr=lr_factors, eps=1e-8)

        # Beta 更新（在重新构造张量之前执行归一化）
        if factor_normalize:
            core, factors = normalize_tucker(core, factors)
        
        if (it % beta_update_interval) == 0:
            S = tucker_to_tensor((core, factors))
            beta = beta_update(
                X=X,
                S=S,
                p=p,
                lambda_beta=lambda_beta,
                method=beta_update_method,
                allow_constant_term=allow_constant_term,
                mask=mask_bool,
            )
        
        if return_history:
            _append_history()

    result = (core, factors, beta)
    if return_history:
        return result, history
    return result


# ============================================================
# Factor Initialization
# ============================================================

def init_ntdpl_factors(
    X,
    rank,
    init: str,
    init_n_iter_max: int,
    mask_float: Optional[np.ndarray],
    random_state: Optional[int],
):
    """
    初始化 NTDPL 的因子和核张量
    
    Parameters:
    -----------
    X : ndarray
        输入张量
    rank : tuple of int
        分解秩
    init : str
        初始化方法
        - 'tucker': 使用 Tucker HO-SVD 初始化
        - 'random': 使用随机高斯初始化
    init_n_iter_max : int
        Tucker 初始化的最大迭代数
    mask_float : ndarray, optional
        浮点掩码（用于张量完成）
    random_state : int or None
        随机种子
    
    Returns:
    --------
    core : ndarray
        核张量
    factors : list of ndarray
        因子矩阵
    """
    init_name = str(init).lower()
    
    if init_name == "tucker":
        # 使用 Tucker HO-SVD 初始化
        tucker_mask = None if mask_float is None else mask_float
        core, factors = tucker(
            X,
            rank=rank,
            n_iter_max=init_n_iter_max,
            init="svd",
            mask=tucker_mask,
            random_state=random_state,
        )
        core = np.asarray(core, dtype=np.float32)
        factors = [np.asarray(f, dtype=np.float32) for f in factors]
        return core, factors
    
    elif init_name == "random":
        # 使用随机初始化
        rng = np.random.default_rng(random_state)
        tensor_shape = tuple(int(dim) for dim in np.asarray(X).shape)
        rank_shape = tuple(int(r) for r in rank)
        core = rng.normal(size=rank_shape).astype(np.float32)
        factors = [
            rng.normal(size=(mode_dim, mode_rank)).astype(np.float32)
            for mode_dim, mode_rank in zip(tensor_shape, rank_shape)
        ]
        return core, factors
    
    else:
        raise ValueError(f"Unsupported init for NTD-PL: {init}")
