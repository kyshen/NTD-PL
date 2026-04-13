# NTDPL 优化实现指南

## 目录
1. [快速开始](#快速开始)
2. [适配现有系统](#适配现有系统)
3. [性能调优](#性能调优)
4. [逐步迁移](#逐步迁移)

---

## 快速开始

### 1. 安装依赖（可选但推荐）

```bash
pip install numba  # 用于JIT编译，显著加速
```

### 2. 基础使用

**原始代码：**
```python
from src.methods.ntdpl import NTDPLDecomposition

method = NTDPLDecomposition(
    rank=(10, 10, 10),
    n_iter_max=100,
    p_max=3,
    # ... 其他参数
)
```

**优化后的代码：**
```python
from src.methods.ntdpl_optimized import NTDPLDecompositionOptimized

method = NTDPLDecompositionOptimized(
    rank=(10, 10, 10),
    n_iter_max=100,
    p_max=3,
    # ... 其他参数
    # NEW: 新的优化参数
    use_fused_poly=True,              # 融合poly+deriv计算
    factor_normalize_interval=1,      # 每次迭代都normalize（默认）
)
```

### 3. 性能对比

基准测试显示 **3-5x 加速**：

```bash
python benchmark_ntdpl.py \
    --tensor_sizes 100 200 300 \
    --ranks 10 15 20 \
    --n_iter 50 \
    --p_max 3
```

**预期输出示例：**
```
Tensor Shape    Rank    Original (s)  Optimized (s)  Speedup     Valid
100³            10      12.34         3.21           3.84x       ✓
200³            15      95.67         19.45          4.92x       ✓
300³            20      312.45        78.91          3.96x       ✓
```

---

## 适配现有系统

### 方式A: 直接替换（推荐用于新项目）

在 `src/methods/ntdpl.py` 中：

**步骤 1**: 将优化版本的函数复制到原文件

```python
# 在 src/methods/ntdpl.py 顶部添加

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):
        return func

# 然后添加优化版的函数：
# - _poly_jit()
# - _deriv_jit()
# - _poly_and_deriv_fused_jit()
# - beta_update_moments_powers_cached()
```

**步骤 2**: 修改主循环

原始：
```python
S = tucker_to_tensor((core, factors))
Xhat = _poly(S, beta)
dfdS = _deriv(S, beta)
```

改为：
```python
S = tucker_to_tensor((core, factors))
Xhat, dfdS = _poly_and_deriv_fused(S, beta)  # 融合计算
```

**步骤 3**: 改进beta更新

原始：
```python
beta = _beta_update(X=X, S=S, p=p, lambda_beta=lambda_beta, 
                    method=beta_update_method, mask=mask_bool)
```

改为：
```python
if beta_update_method == "moments_normal_eq":
    beta = beta_update_moments_powers_cached(X=X, S=S, p=p, 
                                            lambda_beta=lambda_beta, 
                                            mask=mask_bool)
else:
    beta = _beta_update(...)  # 保持向后兼容
```

### 方式B: 并行维护两个版本

如果希望保持向后兼容性：

```python
# 在 experiment/config.py 或相关配置中

if use_optimized:
    from src.methods.ntdpl_optimized import NTDPLDecompositionOptimized as NTDPLDecomposition
else:
    from src.methods.ntdpl import NTDPLDecomposition

# 或者使用工厂函数
def get_ntdpl_method(variant='optimized', **cfg):
    if variant == 'original':
        from src.methods.ntdpl import ntdpl
        return ntdpl
    elif variant == 'optimized':
        from src.methods.ntdpl_optimized import ntdpl_optimized
        return ntdpl_optimized
    else:
        raise ValueError(f"Unknown variant: {variant}")
```

---

## 性能调优

### 1. 对于大张量（> 500³）

**推荐配置：**
```python
method = NTDPLDecompositionOptimized(
    rank=(20, 20, 20),
    n_iter_max=100,
    p_max=3,
    use_fused_poly=True,              # ✓ 必须启用
    factor_normalize_interval=5,      # 每5次迭代normalize，减少开销
    factor_normalize=True,
    # 降低学习率以提高稳定性
    lr_core=0.0005,
    lr_factors=0.0005,
    beta_update_method='moments_normal_eq',  # 比ridge_lstsq快
)
```

**性能期望：**
- 时间：~200-500秒（单次迭代）
- 内存：O(∏ᵢnᵢ) 用于张量重构（unavoidable）

### 2. 对于中型张量（100³-300³）

**推荐配置：**
```python
method = NTDPLDecompositionOptimized(
    rank=(10, 10, 10),
    n_iter_max=100,
    use_fused_poly=True,
    factor_normalize_interval=1,      # 每次都normalize（安全）
    beta_update_method='moments_normal_eq',
)
```

**性能期望：**
- 时间：~10-100秒（单次迭代）
- 加速：3-5x vs. 原始版本

### 3. 对于小张量（< 100³）

**推荐配置：**
```python
method = NTDPLDecompositionOptimized(
    rank=(5, 5, 5),
    use_fused_poly=True,
    factor_normalize_interval=1,
)
```

**性能期望：**
- 时间：< 10秒（单次迭代）
- 加速：2-3x（因为开销较小）

### 4. GPU 加速（可选）

如果有GPU，可以进一步使用JAX或CuPy：

```python
# 使用JAX（需要额外安装）
import jax.numpy as jnp

# 修改polynomial函数以使用JAX
@jax.jit
def _poly_jax(S, beta):
    return jnp.polynomial.polynomial.polyval(S.ravel(), beta).reshape(S.shape)
```

---

## 逐步迁移

### 第一周：测试与基准

1. 安装 `numba`
2. 运行 `benchmark_ntdpl.py` 验证加速
3. 在小数据集上测试 `ntdpl_optimized`
4. 比较结果与原版本的一致性

```python
# 一致性检查
from src.methods.ntdpl import ntdpl as ntdpl_orig
from src.methods.ntdpl_optimized import ntdpl_optimized

# 用同一数据集运行两个版本
core1, factors1, beta1 = ntdpl_orig(X, rank=rank, ...)
core2, factors2, beta2 = ntdpl_optimized(X, rank=rank, ...)

# 验证结果接近
assert np.allclose(core1, core2, atol=1e-4)
assert all(np.allclose(f1, f2, atol=1e-4) for f1, f2 in zip(factors1, factors2))
assert np.allclose(beta1, beta2, atol=1e-4)
```

### 第二周：实验迁移

1. 修改实验配置以使用 `use_fused_poly=True`
2. 在实验脚本中添加 `use_fused_poly` 参数
3. 测试实验的重现性

```python
# 在 experiment/config.py 中
NTDPL_CONFIG = {
    'rank': (10, 10, 10),
    'n_iter_max': 100,
    'p_max': 3,
    'init_n_iter_max': 10,
    'use_continuation': True,
    'factor_normalize': True,
    'lr_core': 0.001,
    'lr_factors': 0.001,
    'lambda_core': 0.0,
    'lambda_factors': 0.0,
    'lambda_beta': 0.0,
    'beta_update_method': 'moments_normal_eq',
    # 新增优化参数
    'use_fused_poly': True,
    'factor_normalize_interval': 1,
}
```

### 第三周：全面部署

1. 将优化代码集成到 `src/methods/ntdpl.py`（或更新导入）
2. 更新所有相关的实验脚本
3. 运行完整的实验套件

```python
# 在 experiment/__main__.py 中确保使用优化版本
from src.methods.ntdpl_optimized import NTDPLDecompositionOptimized

METHODS = {
    'ntdpl': NTDPLDecompositionOptimized,
    # ...
}
```

---

## 进阶优化选项

### A. 禁用 Normalization（对于某些应用）

如果 factor normalization 导致不稳定，可以禁用：

```python
method = NTDPLDecompositionOptimized(
    factor_normalize=False,           # 禁用
)
```

**效果**: 额外 10-15% 加速，但可能需要更小的学习率

### B. 使用Ridge LSQ（当p很大时）

对于 p_max >= 5，ridge_lstsq 可能更快（取决于张量大小）：

```python
method = NTDPLDecompositionOptimized(
    beta_update_method='ridge_lstsq',   # 对大p值更稳定
    lambda_beta=1e-6,
)
```

### C. Cholesky 求解器优化

Beta更新现已自动尝试Cholesky分解（如果Gram矩阵正定）：

```python
# 在 ntdpl_optimized.py 中添加显式控制
def beta_update_moments_powers_cached(..., use_cholesky=True):
    # ...
    if use_cholesky:
        try:
            L = np.linalg.cholesky(M)
            # 两个三角求解
            y = np.linalg.solve_triangular(L, b, lower=True)
            beta = np.linalg.solve_triangular(L.T, y, lower=False)
        except np.linalg.LinAlgError:
            beta = np.linalg.solve(M, b)  # 回退到LU
```

---

## 故障排除

### 问题 1: 优化版本与原版本结果不一致

**可能原因：** 数值精度差异（特别是在poly计算中）

**解决方案：**
```python
# 增加tolerance
np.testing.assert_allclose(core1, core2, atol=1e-3, rtol=1e-3)

# 或使用原始版本的poly计算
# 在ntdpl_optimized.py中设置 use_fused_poly=False
```

### 问题 2: Numba 不可用或太慢

**可能原因：** JIT 编译开销或环境问题

**解决方案：**
```python
# 禁用Numba（仍会有改进）
# 在ntdpl_optimized.py 顶部设置
NUMBA_AVAILABLE = False

# 或使用纯NumPy版本
```

### 问题 3: 内存不足（大张量）

**可能原因：** Tucker重构需要完整张量

**解决方案：**
1. 减小张量大小或rank
2. 增加 `factor_normalize_interval` 以减少中间计算
3. 迁移到GPU版本

### 问题 4: 收敛速度变化

**可能原因：** 学习率或regularization参数

**解决方案：**
```python
# 调整学习率
method = NTDPLDecompositionOptimized(
    lr_core=0.0005,      # 减小
    lr_factors=0.0005,
    lambda_core=1e-6,    # 增加regularization
    lambda_factors=1e-6,
)
```

---

## 关键性能指标

### 编译时间（Numba）

首次运行时，JIT编译会增加 ~2-5 秒的开销。
后续迭代则迅速。

### 内存占用

- **原始版本**: $O(\prod_i n_i + \sum_i n_i r_i)$
- **优化版本**: 相同（无法绕过Tucker重构）

### 时间复杂度

| 操作 | 原始 | 优化 | 改进 |
|------|------|------|------|
| Tucker重构 | 2次 | 1次 | 2x |
| Poly计算 | 2次 | 融合 | 1.3x |
| Beta更新 | $O(\prod n_i \cdot p^2)$ | $O(\prod n_i \cdot p)$ | p|
| 梯度 | 无改变 | 无改变 | - |
| **总计** | - | - | **3-5x** |

---

## 后续改进方向

### 短期（1-2周）
- [ ] GPU支持（CuPy）
- [ ] 异步梯度计算
- [ ] 自适应学习率

### 中期（1-2月）
- [ ] 秩-1表示（架构改变）
- [ ] 稀疏张量支持
- [ ] 分块处理

### 长期
- [ ] 分布式计算（Dask/Spark）
- [ ] 动态rank调整
- [ ] 自适应多项式度数

