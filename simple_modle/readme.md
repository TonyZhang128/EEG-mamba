这版可以定位为 **单向 Mamba + 两级全局统计条件化 \(\Delta\) + 均值池化分类** 的最小可行模型。它的核心不是“Mamba 输出后再调”，而是：**先用可获得的全局统计生成条件，再去调制 Mamba 的 \(\Delta\)，最后执行 selective scan**。第一版建议把双向、动态图、通道注意力和复杂池化全部拿掉，只保留一条干净的时间建模链路。

---

## 一、整体流程

输入五频段 DE：

\[
X\in\mathbb R^{B\times C\times T\times 5}
\]

整体步骤：

\[
X
\rightarrow \text{Linear Projection}
\rightarrow X_p\in\mathbb R^{B\times C\times T\times d}
\]

\[
X_p
\rightarrow \text{计算两级全局描述 }G_c,G_g
\rightarrow \text{调制 Mamba 的 }\Delta
\rightarrow \text{单向 selective scan}
\rightarrow H\in\mathbb R^{B\times C\times T\times d}
\]

\[
H
\rightarrow \text{时间均值池化}
\rightarrow \text{通道均值池化}
\rightarrow \text{MLP 分类}
\]

一句话：**投影 → 两级全局统计 → 条件化 \(\Delta\) → 单向 Mamba 扫描 → 时间池化 → 通道池化 → 分类。**

---

## 二、分模块步骤

### 模块 1：输入线性投影

输入：

\[
X\in\mathbb R^{B\times C\times T\times 5}
\]

线性投影到隐维 \(d\)：

\[
X_p=\operatorname{Linear}(X)
\in\mathbb R^{B\times C\times T\times d}
\]

这里 \(C\) 是通道/电极数，\(T\) 是时间长度，\(d\) 是 Mamba 内部特征维。第一版保持所有通道共享同一套 Mamba 参数，沿时间维 \(T\) 扫描。

---

### 模块 2：计算两级全局描述

关键实现点：**全局描述必须在 \(\Delta\) 生成和 selective scan 之前获得**，不能用最终 Mamba 输出 \(H\) 再回头调制 \(\Delta\)，否则存在循环依赖。

第一版最简做法是直接用投影特征 \(X_p\) 计算全局统计：

单通道全局描述：

\[
G_c=\operatorname{Mean}_T(X_p)
\in\mathbb R^{B\times C\times d}
\]

多通道全局描述：

\[
G_g=\operatorname{Mean}_C(G_c)
\in\mathbb R^{B\times d}
\]

其中：

- \(G_c\)：每个通道一个全局描述，广播到该通道所有时间步；
- \(G_g\)：整个样本一个全局描述，广播到所有通道和时间步。

如果后续坚持用“Mamba 输出 \(H\)”的全局描述，则需要两阶段：先跑一个普通 Mamba 得到 \(H_0\)，池化得 \(G_c,G_g\)，再用它们条件化第二个 Mamba 的 \(\Delta\)。但第一版单层 Mamba 不建议这样做，直接用 \(X_p\) 的统计更干净。

---

### 模块 3：两级全局描述调制 \(\Delta\)

先由 Mamba 的输入投影生成原始 \(\Delta_0\)：

\[
\Delta_0=\operatorname{Softplus}(W_\Delta X_p+b_\Delta)
\]

然后计算条件化缩放：

\[
s_c=\alpha_c\tanh(W_cG_c)
\]

\[
s_g=\alpha_g\tanh(W_gG_g)
\]

其中 \(W_c,W_g\) 投影到与 \(\Delta\) 相同的维度，\(\alpha_c,\alpha_g\) 为可学习标量。两级描述广播后调制：

\[
\Delta'=
\Delta_0\odot
\exp\left(s_c+s_g\right)
\]

即：

\[
\Delta'=
\Delta_0\odot
\exp\left(
\alpha_c\tanh(W_cG_c)
+
\alpha_g\tanh(W_gG_g)
\right)
\]

广播规则：

- \(G_c\in\mathbb R^{B\times C\times d}\)：按通道广播到 \(T\)；
- \(G_g\in\mathbb R^{B\times d}\)：广播到 \(C\) 和 \(T\)；
- \(W_cG_c\) 和 \(W_gG_g\) 必须投影到 \(\Delta\) 的实际维度。

这一步的作用是：**通道级全局状态和样本级全局状态共同缩放每个通道、每个时间步的选择性扫描步长。** 注意这里调制的是 \(\Delta\)，不是简单拼接特征。

---

### 模块 4：单向 Mamba selective scan

用条件化后的 \(\Delta'\) 执行沿时间维的单向 selective scan：

\[
H=\operatorname{MambaScan}(X_p;\Delta')
\]

输出：

\[
H\in\mathbb R^{B\times C\times T\times d}
\]

第一版只沿 \(T\) 正向扫描，不做反向，不做通道间交互。每个通道独立沿时间建模，但所有通道共享 Mamba 参数。

建议加一个轻量残差和归一化，稳定单层训练：

\[
H=\operatorname{LN}\left(X_p+\operatorname{MambaScan}(X_p;\Delta')\right)
\]

如果实现上不方便在 Mamba 内部加残差，也可以先不加，但建议至少保留 Pre-LN 或输出 LayerNorm。

---

### 模块 5：时间池化与通道池化

先沿时间维做均值池化：

\[
P_c=\operatorname{Mean}_T(H)
\in\mathbb R^{B\times C\times d}
\]

再沿通道维做均值池化：

\[
z=\operatorname{Mean}_C(P_c)
\in\mathbb R^{B\times d}
\]

第一版固定使用 Mean Pooling，不引入注意力池化、最大池化或可学习池化。

---

### 模块 6：MLP 分类

最后接一个简单分类头：

\[
\hat y=
\operatorname{Softmax}
\left(
W_2\operatorname{GELU}(W_1z+b_1)+b_2
\right)
\]

或先用单层 Linear 也可以。第一版建议一个小 MLP，便于和普通 Mamba 对照。

---

## 三、第一版伪代码

```python
# X: [B, C, T, 5]
Xp = Linear(X)                         # [B, C, T, d]

# 两级全局描述，来自投影特征，避免循环依赖
Gc = Xp.mean(dim=2)                    # [B, C, d]
Gg = Gc.mean(dim=1)                    # [B, d]

# 原始 Delta
Delta0 = Softplus(DeltaProj(Xp))       # [B, C, T, d_delta]

# 两级条件缩放，投影到 Delta 维度并广播
sc = alpha_c * tanh(Wc(Gc))            # [B, C, d_delta] -> 广播到 T
sg = alpha_g * tanh(Wg(Gg))            # [B, d_delta]    -> 广播到 C, T

Delta = Delta0 * torch.exp(sc[:, :, None, :] + sg[:, None, None, :])

# 单向 selective scan
H = MambaScan(Xp, Delta)               # [B, C, T, d]
H = LayerNorm(Xp + H)                  # 可选轻量残差

# 池化
P = H.mean(dim=2)                      # [B, C, d]
z = P.mean(dim=1)                      # [B, d]

# 分类
y = MLP(z)
```

---

## 四、第一版固定配置

| 模块 | 第一版选择 |
|---|---|
| 输入特征 | 五频段 DE |
| 时间建模 | 单向 Mamba，沿 \(T\) 扫描 |
| Mamba 层数 | 1 层 |
| 全局描述来源 | 投影特征 \(X_p\) 的均值统计 |
| 时间全局池化 | Mean |
| 通道全局池化 | Mean |
| 全局调制 | 两级独立控制 \(\Delta\)，通道级 + 样本级 |
| 空间建模 | Mean Channel Pooling |
| 分类 | Linear / 小 MLP |
| 残差 | 建议轻量残差 + LayerNorm |
| 双向/图/注意力 | 全部关闭 |

---

## 五、必须保留的对照组

为了验证“两级全局描述调制 \(\Delta\)”是否真的有效，第一版至少保留三个模型：

1. **普通单向 Mamba**：不调制 \(\Delta\)，其他完全一致；
2. **仅通道级全局引导**：只用 \(G_c\) 调制 \(\Delta\)；
3. **两级全局引导**：同时用 \(G_c\) 和 \(G_g\) 调制 \(\Delta\)。

如果两级版本没有稳定优于普通 Mamba，优先检查 \(\Delta\) 调制的实现位置、广播维度和缩放初始化，而不是急着加双向或图模块。

---

## 六、实现与单次训练

实现文件：

- `model.py`：普通单向 Mamba、通道级条件 Mamba、两级条件 Mamba；
- `train.py`：读取单个 `sXX.npz`，执行一次分层训练/测试并输出三种模型对比。

默认使用 80% 训练集、20% 测试集，随机种子为 42。三种模型共用同一份划分；标准化只在训练集上拟合，不使用测试集选择模型或训练轮数。

训练脚本默认从 `data/processed/overlap50/sXX.npz` 读取被试数据，因此输入时间长度为 119。

Mamba 分支在输入投影后包含一个沿时间维的因果深度可分离一维卷积，卷积核长度默认为 4。卷积只使用当前及过去的时间步，之后才生成 Mamba 的 `Δ`、`B`、`C` 并执行 selective scan。

```powershell
cd D:\PythonProject
.\.venv\Scripts\python.exe EEGMamba\simple_modle\train.py --subject 6
```

若确实需要交叉验证，可显式开启：

```powershell
.\.venv\Scripts\python.exe EEGMamba\simple_modle\train.py `
  --subject 6 `
  --split-mode cv `
  --folds 5
```

单次训练默认结果保存到 `simple_modle/runs/s06_simple_mamba.json` 和同名的 `.csv` 文件。

### 全部被试的 Valence 二分类

使用全部 32 个被试时，默认按完整被试执行一次 80/20 留出：26 个被试训练，6 个被试测试。同一被试的试次不会同时进入训练集和测试集。二分类标签由四分类标签恢复：

```text
low_valence  = 0  (四分类标签 0 或 1)
high_valence = 1  (四分类标签 2 或 3)
```

运行：

```powershell
.\.venv\Scripts\python.exe EEGMamba\simple_modle\train_all_subjects.py
```

指定被试或调整参数：

```powershell
.\.venv\Scripts\python.exe EEGMamba\simple_modle\train_all_subjects.py `
  --subjects "1-32" `
  --test-subject-ratio 0.2 `
  --epochs 40 `
  --batch-size 32 `
  --device cuda
```

结果保存为：

```text
simple_modle/runs/all_subjects_valence_binary.json
simple_modle/runs/all_subjects_valence_binary.csv
simple_modle/runs/all_subjects_valence_binary_by_subject.csv
```
