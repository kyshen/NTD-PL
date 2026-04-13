"""
NTDPL 工具函数模块

包括：
- Adam 优化器相关函数
- 多项式计算函数
- 误差计算函数
- 辅助函数
"""

from typing import Dict, List
import numpy as np


# ============================================================
# Adam Optimizer
# ============================================================

def adam_init(shape):
    """初始化 Adam 优化器状态"""
    return {
        "m": np.zeros(shape, dtype=np.float32),
        "v": np.zeros(shape, dtype=np.float32),
        "t": 0,
    }


def adam_step(param, grad, state, b1=0.9, b2=0.999, lr=0.001, eps=1e-8):
    """执行一步 Adam 更新"""
    state["t"] += 1
    t = state["t"]
    state["m"] = b1 * state["m"] + (1 - b1) * grad
    state["v"] = b2 * state["v"] + (1 - b2) * (grad * grad)
    mhat = state["m"] / (1 - b1 ** t)
    vhat = state["v"] / (1 - b2 ** t)
    param -= lr * mhat / (np.sqrt(vhat) + eps)


# ============================================================
# Polynomial Evaluation
# ============================================================

def poly(S, beta):
    """
    计算多项式 p(S) = β₀ + β₁·S + ... + βₚ·Sᵖ
    
    使用 Horner 规则：y = ((βₚ·S + βₚ₋₁)·S + ...)·S + β₀
    
    Parameters:
    -----------
    S : ndarray
        张量值
    beta : ndarray
        多项式系数 [β₀, β₁, ..., βₚ]
    
    Returns:
    --------
    y : ndarray
        多项式计算结果
    """
    p = len(beta) - 1
    if p < 0:
        return np.zeros_like(S)
    y = beta[p] * np.ones_like(S)
    for k in range(p - 1, -1, -1):
        y = y * S + beta[k]
    return y


def deriv(S, beta):
    """
    计算多项式导数 dp/dS = β₁ + 2·β₂·S + 3·β₃·S² + ...
    
    使用 Horner 规则：
    dy = ((p·βₚ·S + (p-1)·βₚ₋₁)·S + ...)·S + β₁
    
    Parameters:
    -----------
    S : ndarray
        张量值
    beta : ndarray
        多项式系数
    
    Returns:
    --------
    dy : ndarray
        导数值
    """
    p = len(beta) - 1
    if p < 1:
        return np.zeros_like(S)

    dy = (p * beta[p]) * np.ones_like(S) if p >= 1 else np.zeros_like(S)
    for k in range(p - 1, 0, -1):
        dy = dy * S + (k * beta[k])
    if p == 0:
        dy = np.zeros_like(S)
    return dy


# ============================================================
# Error Metrics
# ============================================================

def rmse_on_target(X, Xhat, mask=None):
    """
    计算 RMSE（均方根误差）
    
    Parameters:
    -----------
    X : ndarray
        目标张量
    Xhat : ndarray
        预测张量
    mask : ndarray, optional
        观测掩码（1=观测，0=缺失）
    
    Returns:
    --------
    rmse : float
        RMSE 值
    """
    X = np.asarray(X)
    Xhat = np.asarray(Xhat)

    if mask is None:
        return float(np.sqrt(np.mean((Xhat - X) ** 2)))

    diff = Xhat[mask] - X[mask]
    if diff.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(diff ** 2)))


# ============================================================
# Beta Dictionary
# ============================================================

def beta_to_dict(beta, p_max: int) -> Dict[str, float]:
    """
    将 beta 数组转换为字典（用于日志记录）
    
    Parameters:
    -----------
    beta : ndarray
        多项式系数
    p_max : int
        最大多项式度数
    
    Returns:
    --------
    out : dict
        字典 {beta_0: ..., beta_1: ..., ...}
    """
    out = {}
    for i in range(p_max + 1):
        out[f"beta_{i}"] = float(beta[i]) if i < len(beta) else 0.0
    return out


# ============================================================
# Continuation Schedule
# ============================================================

def build_continuation_schedule(n_iter_max: int, p_max: int) -> List[int]:
    """
    构建多项式度数增加的时间表
    
    返回一个长度为 p_max - 1 的列表：
        [it_1, it_2, ..., it_{p_max-1}]
    意义：
        p: 1 -> 2 at it_1,
           2 -> 3 at it_2, ...
    
    Parameters:
    -----------
    n_iter_max : int
        总迭代数
    p_max : int
        最大多项式度数
    
    Returns:
    --------
    schedule : list of int
        迭代里程碑
    """
    if p_max <= 1 or n_iter_max <= 0:
        return []

    raw = [(k * n_iter_max) / p_max for k in range(1, p_max)]

    schedule: List[int] = []
    last = 0
    for x in raw:
        it = int(round(x))
        it = max(last + 1, it)
        it = min(it, n_iter_max)
        schedule.append(it)
        last = it

    return schedule
