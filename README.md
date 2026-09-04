# 江苏新能源功率预测说明

本项目用于江苏省新能源功率预测，当前主流程以 **GAT + Informer** 时空深度学习模型预测日前风电出力，保留了原有 XGBoost 模型接口，可通过 `model_type` 切换。

> 当前入口脚本：[jiangsu_power.py](jiangsu_power.py)  
> 当前默认模型：`gat_informer`  
> 当前默认预测目标：`new_energy_wind_ahead`（江苏日前风电出力）

---

## 1. 项目目录

```text
jiangsu_power_predict_gat_informer/
├── jiangsu_power.py                 # 主入口：训练 / 预测任务启动
├── jobs.py                          # 任务封装：日期窗口、结果截断、评估与输出
├── run.sh                           # 顺序执行训练和预测的脚本
├── requirements.txt                 # Python 依赖列表
├── gat_informer.md                  # GAT + Informer 模型替换设计文档
├── configs/
│   └── Data_Info.yaml               # 数据源配置（数据库、MinIO、接口等）
├── api/
│   ├── io/                          # 数据库、日志、消息、插值等 IO 工具
│   ├── model/
│   │   ├── pipeline.py              # 模型流程编排：取数、特征、训练、预测、后处理
│   │   ├── gat_informer_model.py    # GAT + Informer 模型实现
│   │   ├── xgbmodel.py              # XGBoost 模型实现
│   │   ├── data.py                  # 气象数据 / 标签数据读取与特征加工
│   │   ├── constants.py             # 江苏城市、装机容量、数据源常量
│   │   └── anomaly_detector.py      # 训练标签异常检测与清洗
│   └── utils/                       # 通用工具
├── local/
│   └── checkpoints/                 # 模型权重与可视化结果目录
└── output/
    └── pdf_rst_jiangsu.csv          # 预测结果输出文件
```

---

## 2. 功能概述

当前流程主要完成以下工作：

1. 从数据源读取江苏省气象预报数据和新能源出力标签；
2. 按城市构造风速、温度、云量、辐照度、时间编码等特征；
3. 训练阶段清洗异常功率样本；
4. 使用 GAT 学习城市间空间关联；
5. 使用 Informer 风格时序编码器学习 15 分钟粒度的历史序列模式；
6. 输出江苏省新能源风电预测结果；
7. 计算 Bias、MAE、按功率分箱 MAE、业务准确率等指标；
8. 保存预测 CSV、模型权重、训练曲线和注意力可视化结果。

---

## 3. 模型说明

### 3.1 可选模型

模型类型由 [jiangsu_power.py](jiangsu_power.py) 中的 `model_type` 控制：

```python
model_type = 'gat_informer'
```

当前 [api/model/pipeline.py](api/model/pipeline.py) 支持：

| `model_type` | 说明 | 模型文件后缀 |
| --- | --- | --- |
| `xgb` / `xgboost` | XGBoost 回归模型 | `.json` |
| `gat_informer` | GAT + Informer 时空序列模型 | `.pt` |

### 3.2 GAT + Informer 数据流

```text
气象 / 标签数据
      │
      ▼
数据读取与清洗：api/model/data.py
      │
      ▼
特征选择与异常样本剔除：api/model/pipeline.py
      │
      ▼
表格数据转时空张量：GATInformerModel._tabular_to_spatiotemporal()
      │
      ▼
GAT 空间编码：学习城市之间的风况关联
      │
      ▼
Informer 风格时序编码：学习 96 个 15 分钟点的日内序列变化
      │
      ▼
预测头输出省级新能源功率
      │
      ▼
物理约束：功率非负、装机容量上限裁剪
      │
      ▼
结果平滑与 CSV 输出
```

### 3.3 当前 GAT + Informer 关键参数

参数在 [api/model/pipeline.py](api/model/pipeline.py) 实例化 `GATInformerModel` 时设置：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `seq_len` | `96` | 输入历史窗口长度，96 点 = 24 小时（15 分钟粒度） |
| `d_gat` | `32` | GAT 空间编码维度 |
| `gat_heads` | `4` | GAT 注意力头数 |
| `d_model` | `64` | 时序编码隐藏维度 |
| `informer_layers` | `2` | 时序编码层数 |
| `informer_heads` | `4` | 时序注意力头数 |
| `epochs` | `100` | 最大训练轮数 |
| `batch_size` | `32` | 批大小 |
| `lr` | `1e-4` | 学习率 |
| `dropout` | `0.25` | Dropout 正则 |
| `corr_threshold` | `0.45` | 城市图相关性建边阈值 |
| `patience` | `16` | 早停轮数 |
| `capacity_clip_ratio` | `0.98` | 预测值按装机容量上限裁剪比例 |

---

## 4. 数据与特征

### 4.1 数据来源

主要数据读取逻辑位于 [api/model/data.py](api/model/data.py)，配置项位于 [configs/Data_Info.yaml](configs/Data_Info.yaml)。

数据类型包括：

- 江苏省气象数据：Doris 气象表；
- 日前新能源风电 / 光伏出力标签：交易中心数据；
- 装机容量常量：位于 [api/model/constants.py](api/model/constants.py)；
- 预测后结果：写入本地 CSV，部分提交逻辑当前在代码中保留为注释。

> 注意：配置文件中包含数据库和对象存储连接信息，编写部署文档或提交代码时不要外泄敏感信息。

### 4.2 当前风电主要特征

当 `meteo_scope='city'` 且预测 `new_energy_wind_ahead` 时，主要使用：

- 城市级风速：`windspeed_100m_*`、`windspeed_200m_*`；
- 时间编码：`data_hour_sin`、`data_hour_cos`、`data_day`、`data_month_sin`、`data_month_cos`、`data_minute_sin`、`data_minute_cos`；
- 空间差分特征：如盐城-扬州、南通-扬州、盐城-淮安、连云港-徐州百米风速差；
- 区域均值特征：南部 / 北部百米风速均值；
- 滞后风速特征：1h、3h、6h、9h、12h、18h、24h 等滞后项。

GAT + Informer 会将带城市名的特征分配到对应城市节点，并将全局时间特征广播到所有节点。

### 4.3 异常数据清洗

训练时会调用 [api/model/anomaly_detector.py](api/model/anomaly_detector.py) 做异常样本检测，主要包括：

- 负值检测；
- 15 分钟突变检测；
- 数据冻结检测；
- 风电功率与风速三次方关系检测；
- 光伏夜间出力异常检测（光伏任务时）。

检测到的异常样本会在训练前剔除，避免污染模型。

---

## 5. 运行方式

### 5.1 安装依赖

```bash
pip install -r requirements.txt
```

GAT + Informer 依赖 PyTorch。若运行环境中没有安装 `torch`，需要按服务器 CUDA / CPU 环境单独安装，例如：

```bash
pip install torch
```

如使用 GPU，请确认 CUDA、显卡驱动和 PyTorch 版本匹配。

### 5.2 训练模型

在项目根目录执行：

```bash
python jiangsu_power.py train
```

当前训练窗口在 [jiangsu_power.py](jiangsu_power.py) 中配置为：

```python
past_days = datetime.now() + timedelta(days=-160)
past_two_days = datetime.now() + timedelta(days=-15)
```

即默认使用当前日期前约 160 天到前 15 天的数据训练。

训练完成后模型默认保存到：

```text
local/checkpoints/new_energy_wind_ahead-ahead-by_day-noa-unweighted.pt
```

### 5.3 执行预测

```bash
python jiangsu_power.py predict
```

当前预测配置为：

```python
start_span=-12
end_span=-1
```

预测任务会额外向前拼接 4 天历史窗口，以满足时序模型 `seq_len=96` 的输入要求；最终输出时会截断掉历史窗口，只保留业务预测区间。

### 5.4 一键运行脚本

也可以执行：

```bash
bash run.sh
```

`run.sh` 会依次执行：

```bash
python jiangsu_power.py train
python jiangsu_power.py predict
```

> 若部署目录不是脚本中的 `PROJECT_DIR`，需要先修改 [run.sh](run.sh) 中的项目路径。

---

## 6. 输出文件

### 6.1 预测结果

预测结果默认输出到：

```text
output/pdf_rst_jiangsu.csv
```

主要字段：

| 字段 | 说明 |
| --- | --- |
| `date_time` | 预测时间点 |
| `version` | 气象版本时间 |
| `new_energy_wind_pred` | 风电预测值 |
| `new_energy_wind_ahead` | 真实日前风电标签（`get_label=True` 时存在） |

预测结果会经过高斯平滑：

```python
pdf_rst_sorted['new_energy_wind_pred'] = smooth_gaussian(pdf_rst_sorted['new_energy_wind_pred'])
```

### 6.2 模型与可视化结果

默认目录：

```text
local/checkpoints/
```

可能生成：

| 文件 / 目录 | 说明 |
| --- | --- |
| `*.pt` | GAT + Informer 模型权重、图结构、归一化参数、超参数 |
| `visualizations/gat_informer_loss_*.png` | 训练 / 验证损失曲线 |
| `visualizations/gat_attention_*.png` | GAT 城市空间注意力热图 |
| `visualizations/gat_attention_*_temporal.png` | Informer 时序注意力图 |
| `visualizations/rolling_backtest_predictions.csv` | 滚动回测逐点预测结果 |
| `visualizations/rolling_backtest_predictions.json` | 滚动回测 JSON 结果 |

---

## 7. 评估指标

预测结束后，[jobs.py](jobs.py) 会输出：

- `Bias`：预测误差均值；
- `MAE`：平均绝对误差；
- 按真实功率分箱后的 MAE；
- `calculate_accuracy` 计算的业务准确率。

业务准确率当前使用阈值：

```python
threshold = 1000
```

即在计算相对误差时，对低功率段分母做下限保护，避免真实功率过小时 MAPE 失真。

---

## 8. 常用改动位置

| 需求 | 修改位置 |
| --- | --- |
| 切换模型类型 | [jiangsu_power.py](jiangsu_power.py) 的 `model_type` |
| 修改训练 / 预测目标 | [jiangsu_power.py](jiangsu_power.py) 的 `label_list` |
| 修改训练日期范围 | [jiangsu_power.py](jiangsu_power.py) 的 `past_days`、`past_two_days` |
| 修改预测日期范围 | [jiangsu_power.py](jiangsu_power.py) 的 `start_span`、`end_span` |
| 修改 GAT + Informer 超参数 | [api/model/pipeline.py](api/model/pipeline.py) 中 `GATInformerModel(...)` 参数 |
| 修改风电 / 光伏城市列表 | [api/model/constants.py](api/model/constants.py) |
| 修改装机容量 | [api/model/constants.py](api/model/constants.py) |
| 修改气象取数 SQL | [api/model/data.py](api/model/data.py) 的 `_get_mysql_query()` |
| 修改异常检测规则 | [api/model/anomaly_detector.py](api/model/anomaly_detector.py) |
| 修改输出文件路径 | [jobs.py](jobs.py) 的 `pdf_file` |

---

## 9. 注意事项

1. **预测前必须有可加载模型**  
   如果本地不存在 `.pt` 模型文件，需要先执行训练。

2. **GAT + Informer 需要足够长的输入窗口**  
   当前 `seq_len=96`，预测数据至少需要 96 个 15 分钟点。预测任务已在 [jobs.py](jobs.py) 中向前拼接历史窗口。

3. **特征列变化后需要重新训练**  
   模型保存了训练时的特征维度与归一化参数。如果新增 / 删除特征后直接加载旧模型，可能报特征维度不一致，需要删除旧模型并重新训练。

4. **注意数据源权限与网络**  
   项目依赖 Doris、Mongo、MinIO 或内部接口，离线环境或权限不足时无法正常取数。

5. **不要提交敏感配置**  
   数据库账号、密码、对象存储密钥等不应写入公开文档或公开仓库。

---

## 10. 快速命令

```bash
# 进入项目目录
cd jiangsu_power_predict_gat_informer

# 安装依赖
pip install -r requirements.txt

# 训练
python jiangsu_power.py train

# 预测
python jiangsu_power.py predict

# 查看预测结果
python - <<'PY'
import pandas as pd
pdf = pd.read_csv('output/pdf_rst_jiangsu.csv')
print(pdf.head())
print(pdf.tail())
PY
```
