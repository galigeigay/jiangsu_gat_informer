# GAT + Informer 混合模型替换方案

> **目标**：将现有 `jiangsu_power_predict_gat_infomer` 项目中的 XGBoost 模型替换为 GAT + Informer 混合模型，保持对外输入输出接口不变。  
> **日期**：2026/06/05  
> **适用分支/目录**：`jiangsu_power_predict_gat_infomer/`

---

## 一、现有架构回顾

| 层级 | 关键文件 | 职责 |
|------|---------|------|
| 入口 | `jiangsu_power.py` | 指定 `model_type`，调用 `daily_energy_job` |
| 任务调度 | `jobs.py` | 初始化 `ModelPipeline`，执行 `pipeline.run()` |
| 流程编排 | `api/model/pipeline.py` | 获取数据 → 特征工程 → 实例化模型 → 调用 `run_model()` |
| 模型基类 | `api/model/baseclass.py` | 定义 `BaseModel`，要求实现 `train/predict/save/load` |
| XGB 模型 | `api/model/xgbmodel.py` | 基于 `BaseModel` 的 XGBoost 实现 |
| 数据读取 | `api/model/data.py` | 封装气象数据、边界条件数据的读取与后处理 |

**当前 `run_model` 调用契约**（需保持不变）：

```python
model.train(feature=pdf_feature, label=pdf_label)  # feature/label 均为 pd.DataFrame
model.predict(feature=pdf_feature)                  # 返回 np.ndarray
```

---

## 二、GAT + Informer 设计定位

### 2.1 为什么引入 GAT + Informer

| 维度 | XGBoost 局限 | GAT + Informer 优势 |
|------|-------------|-------------------|
| **空间关系** | 将各城市气象视为独立扁平特征，无法建模“盐城大风 → 南通风电受影响”的空间传导 | GAT 显式构建城市图，注意力机制自动学习空间影响权重 |
| **长序列依赖** | 树模型无显式时序建模能力，依赖手工滞后特征 | Informer 的 ProbSparse Self-Attention 原生支持长序列（96+ 时间步） |
| **多步时序模式** | 需手工构造 `lag_1`, `diff_3h` 等特征 | 模型端到端学习时序模式，减少特征工程负担 |
| **可解释性** | 全局特征重要性，无法区分“哪个城市、哪个时刻”更重要 | GAT 注意力系数可输出每个城市对预测的贡献；Informer 可分析长程依赖 |

### 2.2 与现有数据的无缝衔接

本项目已有丰富的市级气象与装机数据，GAT + Informer 可直接复用：

| 现有数据/特征 | 在本方案中的用途 |
|-------------|---------------|
| `{feat}_capa_w_{city}`（如 `windspeed_100m_capa_w_盐城市`） | **GAT 节点输入特征**。已融合 `气象值 × 装机容量`，直接作为每个城市的节点特征向量 |
| `JIANGSU_CITY_WIND_CAPACITY` / `JIANGSU_CITY_SOLAR_CAPACITY` | **图边先验权重**。两城市间边权重与各自装机容量耦合，装机大市之间的关联对全省出力影响更大 |
| `time_idx` | **时序排序基准**。构造 `(batch, seq_len, num_nodes, feat_dim)` 的关键索引 |
| `data_hour_sin/cos`, `data_day` | **时间编码**。作为所有城市节点共享的全局时序特征拼接至节点特征 |

---

## 三、图结构设计方案

### 3.1 节点定义

以 **城市** 为图节点：

- **风电预测**：节点集合 = `JIANGSU_CITY2_WIND`（13 个地级市）
- **光伏预测**：节点集合 = `JIANGSU_CITY2_SOLAR`（13 个地级市）

每个城市在一个时间步的节点特征包括：
- 风电：`windspeed_10m_capa_w`, `windspeed_100m_capa_w`, `windspeed_200m_capa_w`, `pressure_surface_capa_w`
- 光伏：`temperature_2m_capa_w`, `apparent_temperature_capa_w`, `cloudcover_capa_w`, `shortwave_radiation_instant_capa_w`
- 全局时间编码：`data_hour_sin`, `data_hour_cos`, `data_day`

### 3.2 边构建策略（气象相关性 + 装机耦合）

由于无经纬度信息，采用 **历史气象相关性** 建图，并融入装机容量先验。

#### 3.2.1 相关性计算

选取主气象要素（风电用 `windspeed_100m_capa_w`，光伏用 `shortwave_radiation_instant_capa_w`），计算城市两两之间的皮尔逊相关系数：

```python
ρ_ij = |corr(feat_i, feat_j)|
```

#### 3.2.2 装机耦合

两城市间边权重不仅取决于气象相关性，还与其装机容量的几何平均成正比：

```
W_ij = ReLU(ρ_ij - threshold) × √(capa_i × capa_j) / max_capa
```

| 符号 | 含义 |
|------|------|
| `ρ_ij` | 城市 i 与 j 的历史气象绝对相关系数 |
| `threshold` | 截断阈值（建议 0.3），低于此值的边置零 |
| `capa_i`, `capa_j` | 城市 i/j 的风电/光伏装机容量 |
| `max_capa` | 所有城市中最大装机容量（归一化用） |

> **物理意义**：盐城（1121 MW）与南通（706 MW）之间的风速相关性，对全省风电出力影响远大于两个小装机城市（如无锡 9.2 MW、南京 12.5 MW）之间的相关性。

#### 3.2.3 图结构类型

- **无向图**：`edge_index` 双向存储（i→j 和 j→i）
- **自环**：每个节点保留自环边（i→i），权重为 1.0
- **边权重维度**：`(num_edges, 1)`，作为 `GATConv` 的 `edge_attr` 输入

### 3.3 图构建时机

- **训练时**：在 `train()` 中基于当前训练数据动态计算相关性矩阵，构建 `edge_index` 和 `edge_weight`
- **预测时**：复用训练阶段保存的图结构（随模型权重一同 `save/load`）

---

## 四、模型架构设计

### 4.1 整体数据流

```
输入 Tabular DataFrame (N, F)
           │
           ▼
┌─────────────────────────────────────────────┐
│   _tabular_to_spatiotemporal()              │
│  (N, F) → (B, T, num_nodes, D_node)        │
└─────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  GAT Layer (空间编码)                        │
│  输入: (B×T, num_nodes, D_node)             │
│  输出: (B×T, num_nodes, d_gat)              │
│  图结构: edge_index + edge_weight           │
└─────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Reshape: (B, T, num_nodes × d_gat)         │
│  + 线性投影 → (B, T, d_model)               │
│  + 位置编码 (Temporal Embedding)            │
└─────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  Informer Encoder (时序编码)                 │
│  ProbSparse Self-Attention × N_layers       │
│  输出: (B, T, d_model)                      │
└─────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│  时间维度全局池化 / 取最后一个时间步          │
│  + Fully Connected                          │
│  输出: (B, 1) —— 当前时刻的省级总出力预测     │
└─────────────────────────────────────────────┘
```

### 4.2 核心模块

#### 4.2.1 GAT 空间编码层

使用 PyTorch Geometric 的 `GATConv`，支持边权重输入：

```python
from torch_geometric.nn import GATConv

self.gat = GATConv(
    in_channels=D_node,
    out_channels=d_gat // num_heads,
    heads=num_heads,
    concat=True,          # 多头拼接
    edge_dim=1,           # 边权重作为 edge_attr
    dropout=dropout
)
```

**前向过程**：
1. 将 `(B, T, N, D)` reshape 为 `(B×T, N, D)`
2. 对 `B×T` 个时间片分别执行 GAT：每个时间片内，城市为节点，通过 `edge_index` 聚合邻居信息
3. 输出 `(B×T, N, d_gat)`，再 reshape 回 `(B, T, N, d_gat)`

#### 4.2.2 Informer 时序编码层

基于 Informer 的 **Encoder-Only** 结构（本任务为单步回归，无需 Seq2Seq Decoder）：

| 组件 | 配置建议 | 说明 |
|------|---------|------|
| 输入投影 | `Linear(num_nodes × d_gat, d_model)` | 将 GAT 输出展平并映射到 d_model |
| 位置编码 | 可学习或可正弦 | 注入时序位置信息 |
| Encoder Layer | `ProbSparse Self-Attention` + `FeedForward` + `LayerNorm` | Informer 核心，复杂度 O(L log L) |
| 层数 | 2~4 层 | 根据数据量调整 |
| 预测头 | `AdaptiveAvgPool1d(T→1)` + `Linear(d_model, 1)` | 对时间维度全局池化后输出 |

**ProbSparse Self-Attention 核心**：
- 原生 Self-Attention 中，每个 Query 关注所有 Key，复杂度 O(L²)
- ProbSparse 通过采样估计，只保留“重要”的 Query（对 Key 分布差异大的 Query），将复杂度降至 O(L log L)
- 这使得输入序列长度可以达到 **96 ~ 192 个时间步**（24h ~ 48h 的 15min 粒度数据）而不显著增加计算负担

### 4.3 超参数建议

| 超参数 | 风电 | 光伏 | 说明 |
|--------|------|------|------|
| `seq_len` | 96 (24h) | 96 (24h) | 15min 粒度，覆盖完整日前周期 |
| `num_nodes` | 13 | 13 | 江苏省地级市数量 |
| `d_gat` | 64 | 64 | GAT 输出维度 |
| `gat_heads` | 4 | 4 | GAT 注意力头数 |
| `d_model` | 128 | 128 | Informer 隐层维度 |
| `informer_layers` | 2 | 2 | Informer Encoder 层数 |
| `dropout` | 0.2 | 0.2 | 正则化 |
| `epochs` | 100 | 100 | 训练轮数 |
| `batch_size` | 32 | 32 | 根据 GPU 显存调整 |
| `lr` | 1e-3 | 1e-3 | AdamW 初始学习率 |
| `corr_threshold` | 0.3 | 0.3 | 图边相关性截断阈值 |

---

## 五、数据流转详细设计

### 5.1 核心转换：`_tabular_to_spatiotemporal()`

在 `GATInformerModel` 内部实现，将 `run_model()` 传入的 `pd.DataFrame` 转换为时空张量。

#### 输入

- `pdf_data`: 经过 `pipeline.py` 处理后的 `pdf_data_valid`，包含列：
  - `time_idx`（时间排序列）
  - `{feat}_capa_w_{city}`（各城市加权气象特征）
  - `data_hour_sin`, `data_hour_cos`, `data_day`（全局时间编码）
  - `{label}`（目标列，训练时存在）

#### 输出

- `X`: `(B, T, N, D_node)` 的 `np.float32` 数组
- `y`: `(B,)` 的 `np.float32` 数组（训练时），预测时为 `None`

#### 转换步骤

```python
def _tabular_to_spatiotemporal(self, pdf_data, label=None):
    # 1. 按时间严格排序
    pdf = pdf_data.sort_values('time_idx').reset_index(drop=True)
    T_total = len(pdf)

    # 2. 构建每个城市的气象特征向量
    city_list = self._city_list
    weather_cols = self._get_weather_cols()  # wind/solar 区分
    node_feats = []
    for city in city_list:
        city_feat = []
        for col_base in weather_cols:
            col = f'{col_base}_capa_w_{city}'
            city_feat.append(pdf[col].values if col in pdf.columns else np.zeros(T_total))
        node_feats.append(np.stack(city_feat, axis=1))  # (T_total, D_weather)

    # (T_total, N, D_weather)
    node_data = np.stack(node_feats, axis=1).astype(np.float32)

    # 3. 拼接全局时间编码（广播到所有节点）
    time_cols = ['data_hour_sin', 'data_hour_cos', 'data_day']
    time_data = pdf[time_cols].values  # (T_total, D_time)
    time_data = np.repeat(time_data[:, None, :], len(city_list), axis=1)
    node_data = np.concatenate([node_data, time_data], axis=-1)  # (T_total, N, D_node)

    # 4. 滑动窗口构造样本
    X, y = [], []
    for i in range(T_total - self.seq_len + 1):
        X.append(node_data[i : i + self.seq_len])
        if label is not None:
            y.append(label.iloc[i + self.seq_len - 1].values[0])

    return np.array(X), (np.array(y, dtype=np.float32) if y else None)
```

### 5.2 预测长度对齐

序列模型 `seq_len=96` 会导致前 `seq_len - 1`（即 95）个样本无法预测（历史不足）。

**处理方式**：在 `predict()` 中返回与输入行数对齐的数组，头部用 `np.nan` 填充：

```python
def predict(self, **kwargs):
    feature = self._parse_feature(kwargs)
    X, _ = self._tabular_to_spatiotemporal(feature, label=None)

    # 推理
    self._model.eval()
    with torch.no_grad():
        preds = self._model(torch.from_numpy(X).to(self.device)).cpu().numpy()

    # 对齐原始长度
    full_preds = np.full(len(feature), np.nan, dtype=np.float32)
    full_preds[self.seq_len - 1 :] = preds
    return full_preds
```

在 `pipeline.py` 的后处理阶段，可选择对 `nan` 进行前向填充或直接保留（训练时 `dropna` 不受影响）。

---

## 六、代码改动清单

### 6.1 新增文件

#### `api/model/gat_informer_model.py`

- `GATInformerModel` 类：继承 `BaseModel`，实现 `train/predict/save/load`
- `GATInformerNet` 类：PyTorch `nn.Module`，包含 `GATLayer` + `InformerEncoder` + `PredictionHead`
- `_tabular_to_spatiotemporal()`：表格 → 时空张量转换
- `_build_graph()`：基于训练数据计算气象相关性，构建 `edge_index` + `edge_weight`

#### `api/model/informer_layers.py`（可选）

若环境中未安装 Informer 官方库，需自行实现以下模块：
- `ProbAttention`：ProbSparse Self-Attention 核心
- `InformerEncoderLayer`：单层 Encoder（Attention + FFN + LayerNorm）
- `InformerEncoder`：多层堆叠

> 若可安装外部库，可直接 `pip install informer-pytorch` 或引用 GitHub 源码。

### 6.2 修改文件

#### `api/model/pipeline.py`

| 位置 | 修改内容 |
|------|---------|
| `model_type` 校验（~line 68） | `assert model_type in ['xgb', 'xgboost', 'gat_informer'], ...` |
| 模型实例化（~line 210） | 增加 `elif model_type == 'gat_informer':` 分支，实例化 `GATInformerModel`，传入 `city_list` 和 `capacity_dict` |
| 模型文件后缀（~line 214-216） | XGB 用 `.json`，GATInformer 用 `.pt` |
| `get_feature_list()`（可选） | 对于 `gat_informer`，可精简特征列表（GAT 自动学习空间聚合，不再需要 `southern_mean`、`_diff` 等手工聚合特征） |
| `run_model()`（~line 321-347） | 当模型为 `GATInformerModel` 时，保留 `time_idx` 列供时序排序使用 |

#### `jiangsu_power.py`

将 `model_type='xgboost'` 改为 `model_type='gat_informer'`（train 和 predict 两处）。

---

## 七、对外接口契约（保持不变）

```python
# pipeline.py 中的调用方式完全不变

# 训练
model.train(feature=pdf_feature, label=pdf_label)

# 预测
predict_rst = model.predict(feature=pdf_feature)
```

`GATInformerModel` 内部完成：
1. 接收 `pd.DataFrame`（含/不含 `time_idx`）
2. 自动排序、构造序列、图卷积、时序编码
3. 返回 `np.ndarray`（长度与输入对齐，头部可能含 `nan`）

---

## 八、依赖项

```bash
# 基础依赖
pip install torch

# GAT（PyTorch Geometric）
pip install torch-geometric torch-scatter torch-sparse

# 若无法安装 PyG，可在 gat_informer_model.py 中手写简化版 GAT
# （仅保留 multi-head attention 核心逻辑，约 80 行）
```

---

## 九、关键优势总结

1. **气象相关性建图，无需经纬度**：直接从历史 `capa_w` 数据计算城市间皮尔逊相关系数，数据驱动建图。
2. **装机容量双重融入**：
   - 节点特征层：`capa_w = 气象值 × 装机容量`
   - 图边权重层：`W_ij ∝ √(capa_i × capa_j)`
3. **Informer 长序列建模**：支持 96+ 个 15min 步长的历史输入，ProbSparse Attention 保证计算效率。
4. **端到端减少特征工程**：GAT 自动替代 `southern_mean`、城市间 `diff` 等手工空间特征；Informer 自动替代多阶 `lag` 手工时序特征。
5. **结果可解释**：可输出 GAT 注意力热图（每个城市对预测的贡献度）和 Informer 的 ProbSparse 注意力分布（关键历史时间步）。
