## 一、整体架构

设第 \(l\) 个双向动态图条件扫描块的输入为：

\[
H^{(l)}\in\mathbb R^{B\times C\times T\times d}
\]

整体流程为：

\[
X_{DE}
\rightarrow \text{特征投影}
\rightarrow H^{(0)}
\rightarrow \text{双向块}^{(1)}
\rightarrow H^{(1)}
\rightarrow \text{双向块}^{(2)}
\rightarrow H^{(2)}
\rightarrow \text{时间池化}
\rightarrow \text{通道池化}
\rightarrow \text{分类}
\]

每个双向块内部：

1. 对输入做 LayerNorm，构造动态邻接矩阵 \(A_t^{(l)}\)；
2. 邻接矩阵展平、投影得到 \(Z^A\)；
3. 正向/反向 Graph Mamba 分别建模图关系时序，得到 \(Q_t^f,Q_t^b\)；
4. 用 \(Q_t^f,Q_t^b\) 分别调制正向/反向信号 Mamba 的 \(\Delta\)；
5. 正向/反向信号 Mamba 分别扫描，反向输出时间对齐；
6. 正反向输出拼接、线性投影；
7. **块级残差回加**：

\[
H^{(l+1)}=H^{(l)}+\operatorname{Proj}\left(\operatorname{Concat}(H^f,H^b)\right)
\]

8. 进入下一层，或进入池化分类。

一句话：**DE 投影 → 动态构图 → 图 Mamba 得图状态 → 图状态调制信号 Mamba 的 \(\Delta\) → 双向扫描 → 对齐融合 → 残差堆叠 → 池化分类。**

---

## 二、分模块描述

### 模块 1：输入投影与初始特征

输入：

\[
X_{DE}\in\mathbb R^{B\times C\times T\times F}
\]

线性投影到隐维 \(d\)：

\[
H^{(0)}=\operatorname{Linear}(X_{DE})
\in\mathbb R^{B\times C\times T\times d}
\]

这里 \(C\) 为通道/电极数，\(T\) 为时间长度。后续每个双向块保持维度 \(d\) 不变，便于残差连接。

---

### 模块 2：双向块输入归一化与动态构图

对第 \(l\) 个块的输入 \(H^{(l)}\) 先做 Pre-Norm：

\[
\tilde H^{(l)}=\operatorname{LN}(H^{(l)})
\]

由归一化特征按时间构造邻接矩阵：

\[
A_t^{(l)}=\operatorname{GraphConstruct}(\tilde H_t^{(l)})
\]

\[
A^{(l)}\in\mathbb R^{B\times T\times C\times C}
\]

第二层邻接矩阵有两种可选来源：

- 固定使用初始 DE 投影构造的邻接矩阵序列；
- 使用上一层残差输出 \(H^{(l)}\) 重新构图：

\[
A_t^{(l)}=\operatorname{GraphConstruct}(\operatorname{LN}(H_t^{(l)}))
\]

推荐将“第二层重新构图”作为可消融设计。

---

### 模块 3：邻接矩阵编码与 Graph Mamba 分支

每个时刻的邻接矩阵展平：

\[
a_t=\operatorname{Flatten}(A_t^{(l)})\in\mathbb R^{C^2}
\]

投影到图状态维度 \(d_A\)：

\[
z_t^A=W_Aa_t+b_A
\]

得到：

\[
Z^A\in\mathbb R^{B\times T\times d_A}
\]

正向 Graph Mamba 带残差：

\[
Q^f=Z^A+\operatorname{Mamba}_A^f(Z^A)
\]

反向 Graph Mamba 带残差，并做时间对齐：

\[
Q^b=Z^A+
\operatorname{Reverse}
\left[
\operatorname{Mamba}_A^b
\left(
\operatorname{Reverse}(Z^A)
\right)
\right]
\]

这里 \(Q^f,Q^b\in\mathbb R^{B\times T\times d_A}\)，分别表示正向、反向的动态图关系时序状态。Graph Mamba 的残差保证图状态至少保留原始边权时序信息。

---

### 模块 4：图状态调制信号 Mamba 的 \(\Delta\)

信号分支输入仍为归一化后的：

\[
\tilde H^{(l)}\in\mathbb R^{B\times C\times T\times d}
\]

对每个通道沿时间扫描。正向 \(\Delta\) 生成为：

\[
\Delta_{b,c,t}^f=
\operatorname{Softplus}
\left(
W_x^f\tilde h_{b,c,t}^{(l)}
+
W_q^fQ_{b,t}^f
+
b^f
\right)
\]

反向 \(\Delta\) 生成为：

\[
\Delta_{b,c,t}^b=
\operatorname{Softplus}
\left(
W_x^b\tilde h_{b,c,t}^{(l)}
+
W_q^bQ_{b,t}^b
+
b^b
\right)
\]

其中 \(Q_{b,t}\) 是整个脑网络的全局图关系状态，需要广播到 \(C\) 个通道。关键点是：**\(Q_t\) 进入信号 Mamba 的 \(\Delta\) 生成网络，而不是简单与 DE 特征拼接。** 这样动态通道关系会影响各通道时间记忆的选择性更新。

---

### 模块 5：正向/反向信号 Mamba 与方向内残差

正向信号 Mamba：

\[
D^f=\operatorname{Mamba}_X^f(\tilde H^{(l)};\Delta^f)
\]

正向方向残差：

\[
H^f=\tilde H^{(l)}+D^f
\]

反向时，信号和邻接矩阵状态必须按相同时间方向对齐。实现上先将 \(\tilde H^{(l)}\) 和 \(Q^b\) Reverse，生成反向扫描所需的 \(\Delta^b_{\mathrm{rev}}\)，执行反向 Mamba，再 Reverse 回原时间顺序：

\[
D^b=
\operatorname{Reverse}
\left[
\operatorname{Mamba}_X^b
\left(
\operatorname{Reverse}(\tilde H^{(l)});
\Delta^b_{\mathrm{rev}}
\right)
\right]
\]

反向方向残差：

\[
H^b=\tilde H^{(l)}+D^b
\]

于是：

\[
H^f,H^b\in\mathbb R^{B\times C\times T\times d}
\]

方向内残差的作用是让每个方向的信号 Mamba 学习“增量修正”，而不是从头重建特征。

---

### 模块 6：正反向融合与块级残差

正反向输出时间对齐后拼接：

\[
H_{\mathrm{cat}}
=
\operatorname{Concat}(H^f,H^b)
\in\mathbb R^{B\times C\times T\times 2d}
\]

线性投影回统一维度：

\[
\Delta H^{(l)}
=
W_o^{(l)}H_{\mathrm{cat}}+b_o^{(l)}
\]

块级残差回加：

\[
H^{(l+1)}
=
H^{(l)}
+
\Delta H^{(l)}
\]

即：

\[
H^{(l+1)}
=
H^{(l)}
+
W_o^{(l)}
\operatorname{Concat}(H^f,H^b)
+
b_o^{(l)}
\]

这样每个双向块学习的是对输入的残差修正。块级残差是主残差路径，方向内残差是分支残差；两者可同时保留，也可通过消融比较“仅块级残差”“块级+方向内残差”的差异。

---

### 模块 7：两层堆叠与第二层构图

经过两层后：

\[
H^{(2)}\in\mathbb R^{B\times C\times T\times d}
\]

第二层可以：

- 沿用第一层邻接矩阵序列；
- 或根据第一层残差输出重新构图：

\[
A_t^{(2)}
=
\operatorname{GraphConstruct}
\left(
\operatorname{LN}(H_t^{(1)})
\right)
\]

推荐把第二种作为研究方向，因为每一层都能根据更新后的时序表征重新估计动态通道关系；但需注意，完全由特征确定的邻接矩阵可能学到任务驱动相似性，而不一定是稳定生理连接。

---

### 模块 8：池化与分类

池化前可再做一个 LayerNorm：

\[
\tilde H^{(2)}=\operatorname{LN}(H^{(2)})
\]

先时间池化：

\[
P=\operatorname{Pool}_T(\tilde H^{(2)})
\in\mathbb R^{B\times C\times d}
\]

再通道池化：

\[
z=\operatorname{Pool}_C(P)
\in\mathbb R^{B\times d}
\]

最后分类：

\[
\hat y=
\operatorname{Softmax}(W_cz+b_c)
\]

池化建议比较平均池化、最大池化和可学习注意力池化。通道注意力池化可学习不同电极对分类任务的贡献。

---

## 三、残差连接汇总

| 残差位置 | 形式 | 作用 |
|---|---|---|
| Graph Mamba 残差 | \(Q^f=Z^A+\operatorname{Mamba}_A^f(Z^A)\)，\(Q^b=Z^A+\operatorname{Reverse}[\operatorname{Mamba}_A^b(\operatorname{Reverse}(Z^A))]\) | 保留原始邻接矩阵边权时序信息 |
| 信号 Mamba 方向内残差 | \(H^f=\tilde H^{(l)}+\operatorname{Mamba}_X^f(\tilde H^{(l)};\Delta^f)\)，\(H^b=\tilde H^{(l)}+D^b\) | 让方向分支学习增量修正 |
| 双向块级残差 | \(H^{(l+1)}=H^{(l)}+W_o\operatorname{Concat}(H^f,H^b)+b_o\) | 稳定深层堆叠，保留输入表征 |
| 归一化方式 | 块内 Pre-Norm，池化前 LayerNorm | 配合残差稳定训练 |

最终推荐配置是：**Pre-Norm + Graph Mamba 残差 + 信号 Mamba 方向内残差 + 双向块级残差**。这样每个模块都有明确的残差路径，同时主残差路径保证两层堆叠时信息不退化。