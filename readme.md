# EEGMamba 当前数据与建模流程

本 README 描述当前实际运行的流程：

1. 从 DEAP 的 `sXX.mat` 直接提取五频段微分熵特征；
2. 使用 1 秒窗口和 50% 重叠，将每个试次的序列长度扩展到 `T=119`；
3. 每个被试直接保存为一个独立 NPZ 文件；
4. 使用单被试分层训练/测试，比较普通单向 Mamba、通道级条件 Mamba 和两级条件 Mamba。

旧的“先拼接所有被试，再拆分缓存”的流程已经废弃。`split_by_subject.py` 仅保留为兼容入口，实际逻辑位于 `dataset.py`。

---

## 1. 目录结构

```text
D:\PythonProject\
├── .venv\
└── EEGMamba\
    ├── dataset.py
    ├── split_by_subject.py
    ├── readme.md
    ├── data\
    │   └── processed\
    │       ├── deap_de_C32_T60_d5.npz       # 旧合并缓存，不再默认使用
    │       └── overlap50\
    │           ├── s01.npz
    │           ├── s02.npz
    │           └── ... s32.npz
    ├── simple_modle\
    │   ├── model.py
    │   ├── train.py
    │   ├── readme.md
    │   └── runs\
    ├── adjmamba\
    └── forward_scan_benchmark.py
```

---

## 2. 原始数据

默认原始数据目录：

```text
D:\PythonProject\元路径\情绪data\extracted
```

其中包含 `s01.mat` 至 `s32.mat`。每个文件的基本结构为：

```text
data:   (40 trials, 40 channels, 8064 samples)
labels: (40 trials, 4 ratings)
```

当前只使用前 32 个 EEG 通道和标签中的 Valence、Arousal。DEAP 采样率为 128 Hz，原始 63 秒数据中：

- 前 3 秒共 384 点用于基线校正；
- 后 60 秒共 7680 点用于特征提取。

---

## 3. 预处理流程

### 3.1 通道选择与基线校正

```text
(40, 40, 8064) -> (40, 32, 8064)
```

对每个试次、每个通道计算前 3 秒均值，再从后 60 秒信号中逐点减去该均值：

```text
baseline[trial, channel]
    = mean(EEG[trial, channel, 0:384])

EEG_corrected[trial, channel, t]
    = EEG[trial, channel, 384 + t] - baseline[trial, channel]
```

得到：

```text
EEG_corrected: (40, 32, 7680)
```

### 3.2 五频段滤波

在完整 60 秒信号上执行 4 阶 Butterworth 零相位带通滤波：

| 频段 | 范围 |
|---|---:|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–14 Hz |
| Beta | 14–31 Hz |
| Gamma | 31–50 Hz |

全长滤波可以避免逐窗滤波产生边界伪迹。

### 3.3 50% 重叠窗口

当前参数：

```text
采样率 fs     = 128 Hz
窗口长度       = 1 秒 = 128 点
重叠率         = 0.5
窗口步长       = 128 × (1 - 0.5) = 64 点
有效信号长度   = 60 秒 = 7680 点
```

时间窗数量：

```text
T = 1 + floor((7680 - 128) / 64)
  = 1 + floor(7552 / 64)
  = 119
```

因此共享的 60 秒信号会得到 119 个时间步：

```text
(40, 32, 7680) -> (40, 32, 119, 128)
```

这里增加的是单个试次内部的时间序列长度，样本数量仍然是 40 个试次。重叠窗口不是新的独立试次，训练和测试划分必须以试次为单位，不能把同一试次的窗口随机拆到训练集和测试集。

### 3.4 微分熵特征

对每个窗口计算频段功率：

```text
power[trial, channel, window]
    = mean(window_samples²)
```

再计算高斯微分熵：

```text
DE = 0.5 × log(2πe × power + 1e-12)
```

五个频段沿最后一维排列：

```text
(40, 32, 119, 5)
```

对应 `C=32`、`T=119`、`d=5`。

### 3.5 四分类标签

对每个被试分别计算 Valence 和 Arousal 的中位数：

```text
high_valence = valence >= median(valence)
high_arousal = arousal >= median(arousal)
label = high_valence × 2 + high_arousal
```

类别编码：

| 标签 | 含义 |
|---:|---|
| 0 | 低 Valence，低 Arousal |
| 1 | 低 Valence，高 Arousal |
| 2 | 高 Valence，低 Arousal |
| 3 | 高 Valence，高 Arousal |

由于 Valence 和 Arousal 可能相关，各被试的类别数量不一定都等于 10。

---

## 4. 每个被试独立保存 NPZ

默认输出目录：

```text
D:\PythonProject\EEGMamba\data\processed\overlap50
```

每个被试一个文件：

```text
s01.npz
s02.npz
...
s32.npz
```

每个 NPZ 包含：

```text
x:         float32, (40, 32, 119, 5)
y:         int64,   (40,)
subjects:  int64,   (40,)
metadata:  标量字符串，保存为 JSON
```

`metadata` 记录：

- `subject_id`
- `fs`
- `n_channels`
- `baseline_samples`
- `signal_samples`
- `window_size`
- `window_seconds`
- `overlap`
- `hop_size`
- `num_windows`
- 频段边界
- 类别名称和标签方案
- 特征形状

文件采用临时文件加原子替换方式写入。重复运行时，如果已有文件的元数据与当前参数一致，会直接复用；参数不一致时会拒绝错误复用，需要使用 `--force` 重新生成。

---

## 5. 运行预处理

在 `D:\PythonProject` 下运行：

```powershell
# 处理全部 s01.mat - s32.mat
.\.venv\Scripts\python.exe EEGMamba\dataset.py

# 只处理 s25
.\.venv\Scripts\python.exe EEGMamba\dataset.py --subjects s25

# 处理多个被试或范围
.\.venv\Scripts\python.exe EEGMamba\dataset.py --subjects "s01,s02,5-7"

# 强制重新生成
.\.venv\Scripts\python.exe EEGMamba\dataset.py --force
```

可以覆盖默认参数：

```powershell
.\.venv\Scripts\python.exe EEGMamba\dataset.py `
  --data-dir "D:\PythonProject\元路径\情绪data\extracted" `
  --output-dir "EEGMamba\data\processed\overlap50" `
  --overlap 0.5 `
  --window-seconds 1.0
```

`split_by_subject.py` 仍可运行，但只是转发到相同流程：

```powershell
.\.venv\Scripts\python.exe EEGMamba\split_by_subject.py --subjects s25
```

当前 `overlap50` 目录已经生成全部 32 个被试文件，每个文件形状为 `(40, 32, 119, 5)`。

---

## 6. 使用 DEAPDataset

```python
from dataset import DEAPDataset

dataset = DEAPDataset(subject_ids=[25])

print(len(dataset))                 # 40
print(tuple(dataset.x.shape))       # (40, 32, 119, 5)
print(dataset.sequence_length)      # 119
print(dataset.input_dim)            # 5
print(dataset.unique_subject_ids)   # [25]
```

单个试次输入模型时的形状：

```text
dataset.x[index]              -> (32, 119, 5)
dataset.x[index][None]        -> (1, 32, 119, 5)
```

`DEAPDataset` 默认从 `data/processed/overlap50` 加载。指定的被试文件不存在时，会根据原始 `.mat` 自动补齐。

---

## 7. 简单单向 Mamba 模型

模型位于 `simple_modle/model.py`，输入输出为：

```text
输入:  (B, 32, 119, 5)
输出:  (B, 4) 分类 logits
```

处理流程：

```text
五频段 DE
  -> 输入线性投影
  -> 计算通道级和样本级全局描述
  -> 单向 Mamba
  -> 时间均值池化
  -> 通道均值池化
  -> MLP 分类
```

Mamba 分支包含标准的时间一维卷积：

```text
input projection
  -> signal / gate split
  -> causal depthwise Conv1d
  -> SiLU
  -> Δ, B, C generation
  -> selective scan
```

默认卷积参数：

```text
kernel_size = 4
groups      = d_inner
padding     = kernel_size - 1，之后裁剪到原序列长度
```

卷积是因果的，只使用当前和过去的时间步。

### 7.1 两级全局条件

从 Mamba 输入投影特征 `X_p` 计算：

```text
G_c = mean_T(X_p)       # 每个通道一个描述
G_g = mean_C(G_c)       # 整个样本一个描述
```

原始时间步：

```text
Δ_0 = softplus(dt_proj(delta_rank))
```

条件缩放：

```text
s_c = α_c · tanh(W_c · LayerNorm(G_c))
s_g = α_g · tanh(W_g · LayerNorm(G_g))
```

调制后的时间步：

```text
Δ' = Δ_0 · exp(s_c + s_g)
```

`Δ'` 在 selective scan 之前生成。

### 7.2 三个对照模型

| 模式 | 说明 |
|---|---|
| `plain` | 普通单向 Mamba，不使用全局条件 |
| `channel` | 只用 `G_c` 调制 `Δ` |
| `two_level` | 同时使用 `G_c` 和 `G_g` 调制 `Δ` |

三个变体共享相同的输入投影、时间一维卷积、selective scan、池化和分类头结构。

---

## 8. 单被试训练与对比

默认采用一次 80/20 分层训练/测试，不进行交叉验证：

```powershell
.\.venv\Scripts\python.exe EEGMamba\simple_modle\train.py --subject 25
```

默认设置：

| 参数 | 默认值 |
|---|---:|
| 训练/测试比例 | 80% / 20% |
| 随机种子 | 42 |
| epochs | 60 |
| batch size | 8 |
| hidden dim | 32 |
| d_state | 8 |
| dt_rank | 4 |
| 一维卷积核长度 | 4 |
| dropout | 0.1 |
| 条件缩放初值 | 0.1 |
| 优化器 | AdamW |
| 类别权重 | balanced |

三个模型使用完全相同的：

- 训练/测试样本索引；
- 特征标准化参数；
- 优化器参数；
- 训练轮数；
- 批量大小；
- 类别权重。

标准化只在训练集上拟合，测试集不参与标准化、模型选择或最佳轮次选择。

结果默认保存到：

```text
simple_modle/runs/s25_simple_mamba.json
simple_modle/runs/s25_simple_mamba.csv
```

JSON 包含每个模型、每次训练/测试的准确率、平衡准确率、Macro-F1、Log Loss、预测标签和概率。

### 8.1 可选交叉验证

只有显式设置 `--split-mode cv` 才执行交叉验证：

```powershell
.\.venv\Scripts\python.exe EEGMamba\simple_modle\train.py `
  --subject 25 `
  --split-mode cv `
  --folds 5
```

### 8.2 常用参数

```powershell
.\.venv\Scripts\python.exe EEGMamba\simple_modle\train.py `
  --subject 25 `
  --epochs 100 `
  --batch-size 8 `
  --hidden-dim 32 `
  --d-conv 4 `
  --variants plain channel two_level `
  --device cuda `
  --seed 42
```

`--device auto` 会自动选择 CUDA 或 CPU。

---

## 9. 实验注意事项

1. 每个被试只有 40 个试次，单次 80/20 划分通常只有 8 个测试样本，单次结果波动较大。
2. 50% 重叠增加的是序列长度 `T`，不是独立样本数量。
3. 同一试次的重叠窗口高度相关，不能作为独立试次跨训练集和测试集划分。
4. 四分类标签由每个被试自身的中位数生成，因此类别边界具有被试特异性。
5. 当前只有 40 个样本，模型准确率接近随机水平并不必然说明实现错误；需要结合数据可分性、多个随机种子或被试级统计判断。
6. 旧的 `data/processed/deap_de_C32_T60_d5.npz` 不参与当前训练，默认训练数据是 `data/processed/overlap50/sXX.npz`。

---

## 10. 其他模块

`adjmamba/` 中保留了双向动态图 Mamba 的实验代码，包括：

- 动态通道图；
- 图 Mamba；
- 图状态条件化；
- 双向 scan；
- 注意力池化。

这些模块独立于 `simple_modle`。当前单被试回归实验默认只使用 `simple_modle/model.py` 中的单向时间 Mamba。
