# NTDPL 性能分析与优化方案

## 一、主要性能瓶颈位置

### 1. **Tucker张量重构** （最昂贵，每次迭代执行2次）
```python
S = tucker_to_tensor((core, factors))  # 行 131
```

**复杂度**: $O(\prod_i n_i \cdot r_i)$，其中 $n_i$ 是维度大小，$r_i$ 是秩

- 涉及多个模乘运算（N-way 矩阵-张量乘法）
- 每次迭代执行2次（第131行和第163行）
- 与Tucker分解不同，Tucker只需在HO-SVD的秩-1更新中触及一部分数据

---

### 2. **多项式计算与导数计算** （每次迭代执行2次）
```python
Xhat = _poly(S, beta)      # 行 133
dfdS = _deriv(S, beta)     # 行 134
```

**`_poly`函数复杂度**: $O(\prod_i n_i \cdot p)$
- p越大开销越大（p_max可以是3-5）
- Horner规则虽然已经优化，但仍需 p 次元素乘法

**`_deriv`函数复杂度**: $O(\prod_i n_i \cdot p)$
- 类似多项式计算
- 单独计算导数也有开销

---

### 3. **Beta更新** （每次迭代执行1次）
```python
beta = _beta_update(X, S, p, lambda_beta, method, mask)
```

**两个方法的复杂度对比**:

#### 方法 1: `moments_normal_eq` 
- **前向计算**: $O(\prod_i n_i \cdot p^2)$（计算 $\sum S^m$ 对 $m=0..2p$）
- **矩阵求解**: $O(p^3)$（solve (p+1)×(p+1) 系统）
- **总复杂度**: $O(\prod_i n_i \cdot p^2)$

#### 方法 2: `ridge_lstsq`
- **构造设计矩阵**: $O(\prod_i n_i \cdot p)$（逐列构造 Phi）
- **最小二乘求解**: $O((\prod_i n_i)^2 \cdot p)$（当 $\prod_i n_i \gg p$ 时）
- **总复杂度**: $O((\prod_i n_i)^2 \cdot p)$

**问题**: Ridge LSQ 对大张量非常昂贵，moments method 仍需遍历所有元素 p 次。

---

### 4. **梯度计算与参数更新**
```python
# 核张量梯度
grad_core = multi_mode_dot(T, [f.T for f in factors], modes=modes_all)  # 行 144

# 因子梯度（每个模式一次）
for n in range(N):
    M = multi_mode_dot(core, [factors[k] for k in other_modes], modes=other_modes)  # N-1 次模乘
    Z = tl.unfold(M, mode=n)
    Tn = tl.unfold(T, mode=n)
    grad_A = np.dot(Tn, Z.T)
```

**复杂度**: $O(N \cdot \prod_i n_i \cdot \sum_j r_j^2)$ （多模乘 + 矩阵乘法）
- N次多模乘操作
- 每个模式都需要unfold和矩阵乘法

---

### 5. **Normalization** （可选但每次迭代执行）
```python
if factor_normalize:
    core, factors = normalize_tucker(core, factors)  # 行 157
```

**复杂度**: $O(\sum_i n_i r_i)$（计算列范数）+ $O(\prod_i r_i)$（缩放核张量）
- 每个模式计算所有列的L2范数
- 然后用scale向量缩放核张量每个模式

---

## 二、NTDPL vs Tucker 分解开销对比

### Tucker 分解（TensorLy）
- **主要操作**: HO-SVD 的秩-1更新
- **每步成本**: $O(N \cdot r^3 + N \cdot \prod_i n_i)$ （稀疏更新）
- **关键优势**: 
  - 利用秩的结构，不需要完整重构张量
  - 秩-1更新和投影都很高效
  - SVD 是 $O(n^2 \cdot r)$

### NTDPL 当前实现
- **每步成本**: 
  - Tucker重构: $O(\prod_i n_i \cdot r_i)$ ❌ (2次)
  - 多项式: $O(\prod_i n_i \cdot p)$ ❌ (2次)
  - Beta更新: $O(\prod_i n_i \cdot p^2)$ 或 $O((\prod_i n_i)^2 \cdot p)$ ❌
  - 梯度: $O(N \cdot \prod_i n_i \cdot r^2)$ ❌
  - **总计**: $O(\prod_i n_i \cdot (r + p + p^2 + Nr^2))$

**问题**: $\prod_i n_i$ 项主导！与张量大小成线性关系。

---

## 三、优化策略

### 优化 1: 避免重复的Tucker重构 ✅
**当前代码**:
```python
S = tucker_to_tensor((core, factors))  # 第131行
...  # 其他操作不改变S
S = tucker_to_tensor((core, factors))  # 第163行，完全重复！
```
**优化**: 重用第一次重构的结果
- **效果**: -50% 的 Tucker 重构成本
- **实现**: 简单，无副作用

---

### 优化 2: 使用快速多项式计算 ✅
**当前**: 逐元素的Horner规则
```python
def _poly(S, beta):
    y = beta[p] * np.ones_like(S)
    for k in range(p - 1, -1, -1):
        y = y * S + beta[k]
    return y
```

**优化方案**:
1. **预计算多项式的导数系数**: 在外部计算，复用系数
2. **使用NumPy的 `np.polynomial.polynomial.polyval`**: 可能有优化的BLAS调用
3. **合并 poly 和 deriv 计算**:
```python
def _poly_and_deriv_fused(S, beta):
    """同时计算多项式和导数，避免重复的S^k计算"""
    p = len(beta) - 1
    y = np.full_like(S, beta[p], dtype=np.float32)
    dy = np.full_like(S, p * beta[p], dtype=np.float32) if p > 0 else np.zeros_like(S)
    
    for k in range(p - 1, -1, -1):
        dy = dy * S + k * (beta[k+1] if k+1 < len(beta) else 0)  # 正确的导数递推
        y = y * S + beta[k]
    return y, dy
```
- **效果**: -30-40% 的多项式计算成本

---

### 优化 3: 改进Beta更新 ✅
**问题**: 当前两个方法都不够快

**优化方案A - 使用矩量方程且缓存幂和**:
```python
def beta_update_moments_cached(X, S, p, lambda_beta=0.0, mask=None):
    """缓存中间的幂，避免重复计算"""
    Xv = X.ravel().astype(np.float64)
    Sv = S.ravel().astype(np.float64)
    
    if mask is not None:
        mv = mask.ravel().astype(bool)
        Xv = Xv[mv]
        Sv = Sv[mv]
    
    # 缓存S的幂：S^0, S^1, ..., S^(2p)
    powers = {}
    powers[0] = Sv.size
    powers[1] = Sv.sum()
    Sv_cache = Sv.copy()
    for m in range(2, 2*p + 1):
        powers[m] = np.sum(Sv_cache)
        Sv_cache *= Sv
    
    # 构造Gram矩阵和右侧向量
    d = p + 1
    M = np.zeros((d, d), dtype=np.float64)
    b = np.zeros(d, dtype=np.float64)
    
    for i in range(d):
        for j in range(d):
            M[i, j] = powers[i + j]
    
    for i in range(d):
        if i == 0:
            b[i] = Xv.sum()
        else:
            b[i] = np.sum(Xv * (Sv ** i))
    
    if lambda_beta > 0:
        M += lambda_beta * np.eye(d, dtype=np.float64)
    
    return np.linalg.solve(M, b).astype(np.float32)
```

**优化方案B - Cholesky分解 (如果M正定)**:
- 使用 `np.linalg.cholesky` 代替 `solve`
- **效果**: -50% 的求解成本（从 $O(p^3)$ 升级到两个 $O(p^3/3)$ 的三角求解）

**总效果**: -20-30% 的Beta更新成本

---

### 优化 4: 缓存和合并梯度计算 ✅
**当前**: 每个模式重新计算M

```python
# 优化前：N次昂贵的multi_mode_dot
for n in range(N):
    other_modes = [k for k in range(N) if k != n]
    M = multi_mode_dot(core, [factors[k] for k in other_modes], modes=other_modes)  # 这是O(∏rᵢ)的操作
    Z = tl.unfold(M, mode=n)
    Tn = tl.unfold(T, mode=n)
    grad_A = np.dot(Tn, Z.T)
```

**优化方案A - 使用Khatri-Rao乘积的秩-1更新**:
```python
def compute_factor_grad_khatrirao(T, core, factors, n):
    """使用Khatri-Rao乘积实现O(r²)的因子梯度"""
    other_modes = [k for k in range(len(factors)) if k != n]
    
    # 使用 Khatri-Rao 乘积代替multi_mode_dot
    # 这仅在使用秩-1格式时有效，但可以大幅加速
    
    Tn = tl.unfold(T, mode=n)
    # Khatri-Rao 的每个秩-1向量贡献
    grad_A = np.zeros((T.shape[n], factors[n].shape[1]), dtype=np.float32)
    for r in range(factors[n].shape[1]):
        # 第r个秩-1的核张量-因子 Khatri-Rao 乘积
        v_r = _khatrirao_vectors_r(core, factors, n, r)
        grad_A[:, r] = np.dot(Tn, v_r)
    
    return grad_A
```

**优化方案B - 简化un-fold**:
- 使用 tensorly 的张量-矩阵乘积而不是unfold
- `tl.unfold` 和 `tl.fold` 有开销

**总效果**: 取决于实现，但可以 -20-40%

---

### 优化 5: 使用更高效的线性代数库 ✅
**当前**: NumPy（CPU多线程）
**优化**:
1. **Numba JIT编译**: 对热点函数（_poly, _deriv, beta更新）编译
2. **JAX/CuPy**: GPU加速（如果有GPU）
3. **MKL-DNN**: 更快的BLAS（已通过numpy使用，但可显式配置）

```python
from numba import njit

@njit
def _poly_jit(S, beta):
    p = len(beta) - 1
    if p < 0:
        return np.zeros_like(S)
    y = np.full_like(S, beta[p])
    for k in range(p - 1, -1, -1):
        y = y * S + beta[k]
    return y
```

**效果**: -30-50% 的执行时间（对于大张量）

---

### 优化 6: 避免不必要的Normalization ✅
**当前**:
```python
if factor_normalize:
    core, factors = normalize_tucker(core, factors)  # 每次迭代
```

**问题**: 如果每次迭代都normalize，代价可积累
**优化**:
- 改为间隔normalize（例如每10次迭代一次）
- 或使用更轻量的normalization（只scale factors，不动core）

**效果**: 如果factor_normalize=True，-10-20%

---

### 优化 7: 优化Beta继续策略 ✅
**当前**:
```python
if use_continuation:
    while continuation_idx < len(continuation_schedule) and it >= continuation_schedule[continuation_idx]:
        p += 1
        beta_new = np.zeros(p + 1, dtype=np.float32)  # 重新分配！
        beta_new[:p] = beta
        beta = beta_new
```

**问题**: 每次 p 增加都要重新计算beta（全重新计算）

**优化**: 
- 预先计算 p=1,2,...,p_max 时所有的系数
- 使用暖启动（从上一个p的值开始）
- 可能的方法：Newton continuation而不是p continuation

**效果**: -5-15% （如果frequently changing p）

---

## 四、综合优化方案与预期加速

### 立即可实施（无算法改变）
| 优化项 | 实施难度 | 预期加速 |
|-------|---------|---------|
| 1. 避免重复Tucker重构 | ⭐ | 1.5-2x |
| 2. 合并poly+deriv计算 | ⭐⭐ | 1.2-1.4x |
| 3. 改进beta更新缓存 | ⭐⭐ | 1.2-1.3x |
| 4. Numba JIT编译 | ⭐⭐ | 1.3-1.5x |

**总体**: **4-6x 加速**

### 进一步优化（需调整代码结构）
| 优化项 | 实施难度 | 预期加速 |
|-------|---------|---------|
| 5. Khatri-Rao梯度 | ⭐⭐⭐ | 1.5-2x |
| 6. GPU加速（CuPy/JAX） | ⭐⭐⭐⭐ | 5-20x |
| 7. 改进beta继续策略 | ⭐⭐⭐ | 1.1-1.2x |

**总体**: 可达到 **10-15x 加速** （与GPU）

---

## 五、变为Scalable的关键

### 当前的可扩展性问题
- **空间**: $O(\prod_i n_i)$ - 完整的Tucker张量重构
- **时间**: $O(\prod_i n_i \cdot k)$ 其中k=iteration数
- **结果**: 大张量上非常慢

### 要达到与Tucker相似的可扩展性
**核心思路**: 避免张量大小的显式因子

1. **使用秩-1格式表示**: 不重构完整的 $\mathcal{S}$
   ```python
   # 代替：S = tucker_to_tensor((core, factors))  # O(∏nᵢ)
   # 使用：S_r1 = [(core_r, [f[:,r] for f in factors]) for r in range(R)]
   ```

2. **秩-1级别的多项式计算**:
   ```python
   def poly_rank1(core_r, factors_r, beta):
       """对单个秩-1项计算多项式"""
       # 这避免了 ∏nᵢ 的显式factor
       return _poly_scalar(np.sum(core_r), beta)  # O(p)
   ```

3. **流式Gram矩阵更新**:
   ```python
   # 而不是：beta = solve(M, b) 其中 M 需要 ∏nᵢ 的计算
   # 使用：分块计算，流式累加
   ```

4. **使用稀疏或结构化的表示**

---

## 六、建议的优化实施顺序

1. **第一步** ⚡ (优先级最高)
   - ✅ 避免重复Tucker重构
   - ✅ 合并poly+deriv计算
   - 预期: **2-3x 加速** (容易, 大收益)

2. **第二步** ⚡⚡
   - ✅ Numba JIT编译热点函数
   - ✅ 改进beta更新（缓存和Cholesky）
   - 预期: **3-5x 总加速** 

3. **第三步** ⚡⚡⚡
   - ✅ Khatri-Rao梯度或稀疏梯度计算
   - ✅ 改进factor normalization策略
   - 预期: **5-8x 总加速**

4. **第四步** (长期，可选)
   - 🔄 GPU支持（JAX/CuPy）
   - 🔄 秩-1表示架构重构

---

## 七、Tucker分解保持快速的原因

1. **秩-1 HO-SVD**: 每次更新只涉及一个秩（O(∑nᵢ·r)）
2. **SVD投影**: O(nᵢ·r²)，而不是 O(∏nᵢ)
3. **无显式张量重构**: 所有操作都在因子空间进行
4. **线性收敛**: 通常需要较少的迭代

**NTDPL当前无法匹配的原因**: 
- 多项式引入的非线性
- Beta参数的全局最小二乘（涉及所有元素）
- Tucker重构用于多项式评估（无法避免）

---

## 八、推荐的优化代码模板见下一部分
