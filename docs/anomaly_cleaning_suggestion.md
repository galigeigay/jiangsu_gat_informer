# 出力数据异常清洗流程设计建议

## 一、背景分析

### 当前数据处理现状

根据对项目代码的分析，当前数据处理流程中存在以下简单过滤逻辑：

**位置**：`api/model/pipeline.py`

| 代码位置 | 处理逻辑 | 说明 |
|---------|---------|-----|
| 第239-241行 | `pdf_data = pdf_data[~pdf_data[label].isna()]` | 过滤 NaN 标签 |
| 第239-241行 | `pdf_data = pdf_data[pdf_data[label] >= 50]` | 过滤低功率样本（<50MW） |
| 第249-273行 | 线性插值 + 前后填充 | 处理特征列 NaN 值 |
| 第338-359行 | 夜间置零 + 负值截断 | 预测结果后处理 |

### 缺失的异常检测机制

当前流程**缺乏以下异常检测**：

1. ❌ 出力与物理关系的异常检测（功率与风速³关系）
2. ❌ 出力突变异常检测
3. ❌ 数据冻结异常检测（长时间恒定值）
4. ❌ 光伏夜间出力异常检测

---

## 二、异常检测方案设计

### 核心思路

**利用物理关系而非装机容量数据**：
- **风电**：功率 P 与风速 v 的三次方成正比，即 P ∝ v³
- **光伏**：功率与辐照度成正比，夜间辐照度接近 0 时功率应为 0

### 检测规则设计

#### 1. 风电功率曲线异常检测

**物理原理**：
```
风电功率公式：P = 0.5 × ρ × A × Cp × v³
简化为：P = k × v³（k 为综合系数）
```

**关键参数**：
- 切入风速：约 3 m/s（低于此风速风机不发电）
- 切出风速：约 25 m/s（高于此风速风机停机保护）
- 额定风速：约 12-15 m/s

**检测方法**：
1. 拟合功率系数 k：从历史数据计算 k = P / v³ 的中位数（鲁棒估计）
2. 计算理论功率：P_theoretical = k × v³
3. 计算偏差比：ratio = P_actual / P_theoretical
4. 判定异常：
   - ratio > 2.0：功率远超理论值（异常）
   - ratio < 0.5 且 v > 5m/s 且 P > 100MW：功率远低于理论值（异常）

#### 2. 光伏夜间异常检测

**物理原理**：
- 夜间辐照度接近 0 W/m²，光伏功率应为 0
- 辐照度 < 1 W/m² 时，功率应接近 0

**检测方法**：
- 辐照度 < 1 W/m² 且 功率 > 50MW → 异常

#### 3. 突变异常检测

**规则**：
- 15分钟内功率变化 > 500MW → 异常
- 计算：`|功率_t - 功率_{t-1}| > 500MW`

#### 4. 数据冻结检测

**规则**：
- 连续 12 个点（3小时）标准差为 0 且 功率 > 100MW → 异常
- 表明数据测量系统故障或数据传输问题

#### 5. 负值异常检测

**规则**：
- 功率 < 0 → 异常（物理上不可能）

---

## 三、实现方案

### 1. 新建异常检测模块

**文件路径**：`api/model/anomaly_detector.py`

```python
"""
出力数据异常检测器
基于物理关系检测训练数据中的异常样本
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from api.io.logger import setup_logger

logger = setup_logger(__name__)


class PowerAnomalyDetector:
    """出力数据异常检测器 - 基于物理关系"""

    def __init__(self,
                 windspeed_col: Optional[str] = 'windspeed_100m_盐城市',
                 radiation_col: Optional[str] = 'shortwave_radiation_instant',
                 wind_power_coefficient: Optional[float] = None,
                 tolerance_ratio: float = 2.0,
                 max_change_per_quarter: float = 500,
                 frozen_window: int = 12,
                 radiation_threshold: float = 1.0,
                 ):
        """
        Args:
            windspeed_col: 用于检测的风速特征列（默认盐城百米风速）
            radiation_col: 用于检测的辐照度特征列
            wind_power_coefficient: 功率/风速³系数，若为None则自动拟合
            tolerance_ratio: 功率曲线异常容忍倍数
            max_change_per_quarter: 突变检测阈值（MW/15min）
            frozen_window: 数据冻结检测窗口（点数）
            radiation_threshold: 辐照度阈值（W/m²）
        """
        self.windspeed_col = windspeed_col
        self.radiation_col = radiation_col
        self.wind_power_coefficient = wind_power_coefficient
        self.tolerance_ratio = tolerance_ratio
        self.max_change_per_quarter = max_change_per_quarter
        self.frozen_window = frozen_window
        self.radiation_threshold = radiation_threshold

    def fit_wind_power_curve(self, df: pd.DataFrame, 
                             power_col: str, 
                             windspeed_col: str) -> float:
        """
        拟合功率-风速³关系系数 P = k × v³
        
        Args:
            df: 数据DataFrame
            power_col: 功率列名
            windspeed_col: 风速列名
            
        Returns:
            k: 功率系数（P/v³的中位数）
        """
        # 过滤有效范围（切入风速3m/s到切出风速25m/s）
        valid_mask = (
            (df[windspeed_col] >= 3) & 
            (df[windspeed_col] <= 25) & 
            (df[power_col] > 0)
        )
        valid_df = df[valid_mask]
        
        if len(valid_df) < 100:
            logger.warning(f"有效样本数不足（{len(valid_df)}），无法拟合功率系数")
            return None
        
        # 计算理论功率系数 k = P / v³
        v3 = valid_df[windspeed_col] ** 3
        k_values = valid_df[power_col] / v3
        
        # 取中位数作为代表系数（鲁棒性）
        k = np.median(k_values)
        logger.info(f"拟合功率系数 k = P/v³ = {k:.4f} (样本数: {len(valid_df)})")
        
        return k

    def detect_wind_power_anomaly(self, 
                                  df: pd.DataFrame, 
                                  power_col: str, 
                                  windspeed_col: str) -> pd.Index:
        """
        检测风电功率与风速³关系异常
        
        异常条件：
        1. ratio > tolerance_ratio: 功率远超理论值
        2. ratio < 1/tolerance_ratio 且风速>5m/s且功率>100: 功率远低于理论值
        
        Returns:
            异常样本的索引
        """
        # 拟合或使用已有系数
        if self.wind_power_coefficient is None:
            k = self.fit_wind_power_curve(df, power_col, windspeed_col)
            if k is None:
                return pd.Index([])
        else:
            k = self.wind_power_coefficient
        
        # 计算理论功率
        theoretical_power = k * df[windspeed_col] ** 3
        
        # 计算偏差比（避免除零）
        ratio = np.where(theoretical_power > 0, 
                         df[power_col] / theoretical_power, 
                         np.inf)
        
        # 异常条件
        anomaly_high = ratio > self.tolerance_ratio
        anomaly_low = (
            (ratio < 1/self.tolerance_ratio) & 
            (df[windspeed_col] > 5) & 
            (df[power_col] > 100)
        )
        
        anomaly_mask = anomaly_high | anomaly_low
        anomaly_indices = df[anomaly_mask].index
        
        logger.info(f"[功率曲线异常] 发现 {len(anomaly_indices)} 个异常点")
        
        return anomaly_indices

    def detect_solar_night_anomaly(self, 
                                   df: pd.DataFrame, 
                                   power_col: str, 
                                   radiation_col: str) -> pd.Index:
        """
        检测光伏夜间出力异常
        
        条件：辐照度 < threshold 且 功率 > 50MW
        
        Returns:
            异常样本的索引
        """
        anomaly_mask = (
            (df[radiation_col] < self.radiation_threshold) & 
            (df[power_col] > 50)
        )
        anomaly_indices = df[anomaly_mask].index
        
        logger.info(f"[光伏夜间异常] 发现 {len(anomaly_indices)} 个异常点")
        
        return anomaly_indices

    def detect_negative_values(self, df: pd.DataFrame, power_col: str) -> pd.Index:
        """
        检测负值异常
        
        Returns:
            异常样本的索引
        """
        anomaly_indices = df[df[power_col] < 0].index
        logger.info(f"[负值异常] 发现 {len(anomaly_indices)} 个异常点")
        return anomaly_indices

    def detect_sudden_jumps(self, df: pd.DataFrame, power_col: str) -> pd.Index:
        """
        检测突变异常
        
        条件：15分钟功率变化 > max_change_per_quarter
        
        Returns:
            异常样本的索引
        """
        power_diff = df[power_col].diff().abs()
        anomaly_mask = power_diff > self.max_change_per_quarter
        anomaly_indices = df[anomaly_mask].index
        
        logger.info(f"[突变异常] 发现 {len(anomaly_indices)} 个异常点")
        
        return anomaly_indices

    def detect_frozen_values(self, df: pd.DataFrame, power_col: str) -> pd.Index:
        """
        检测数据冻结
        
        条件：连续 frozen_window 点标准差为 0 且 功率 > 100MW
        
        Returns:
            异常样本的索引
        """
        rolling_std = df[power_col].rolling(window=self.frozen_window).std()
        anomaly_mask = (rolling_std == 0) & (df[power_col] > 100)
        anomaly_indices = df[anomaly_mask].index
        
        logger.info(f"[数据冻结异常] 发现 {len(anomaly_indices)} 个异常点")
        
        return anomaly_indices

    def detect_all(self, 
                   df: pd.DataFrame, 
                   power_col: str, 
                   label_type: str) -> Dict[str, pd.Index]:
        """
        综合检测，返回各类异常索引
        
        Args:
            df: 数据DataFrame
            power_col: 功率/出力列名
            label_type: 标签类型（'wind' 或 'solar'）
            
        Returns:
            dict: {异常类型: 异常索引集合}
        """
        anomalies = {}
        
        # 通用检测
        anomalies['negative'] = self.detect_negative_values(df, power_col)
        anomalies['sudden_jump'] = self.detect_sudden_jumps(df, power_col)
        anomalies['frozen'] = self.detect_frozen_values(df, power_col)
        
        # 风电专用检测
        if 'wind' in label_type and self.windspeed_col:
            if self.windspeed_col in df.columns:
                anomalies['wind_power_curve'] = self.detect_wind_power_anomaly(
                    df, power_col, self.windspeed_col
                )
        
        # 光伏专用检测
        if 'solar' in label_type and self.radiation_col:
            if self.radiation_col in df.columns:
                anomalies['solar_night'] = self.detect_solar_night_anomaly(
                    df, power_col, self.radiation_col
                )
        
        return anomalies

    def get_total_anomaly_indices(self, anomalies: Dict[str, pd.Index]) -> pd.Index:
        """
        合并所有异常索引
        
        Args:
            anomalies: 检测结果字典
            
        Returns:
            合并后的异常索引
        """
        all_indices = set()
        for anomaly_type, indices in anomalies.items():
            all_indices.update(indices.tolist())
        
        return pd.Index(list(all_indices))


def clean_anomalies(df: pd.DataFrame, 
                    power_col: str, 
                    label_type: str,
                    detector: Optional[PowerAnomalyDetector] = None,
                    windspeed_col: Optional[str] = None) -> pd.DataFrame:
    """
    清洗异常数据的便捷函数
    
    Args:
        df: 数据DataFrame
        power_col: 功率列名
        label_type: 标签类型
        detector: 检测器实例（若为None则自动创建）
        windspeed_col: 风速列名（用于风电检测）
        
    Returns:
        清洗后的DataFrame
    """
    # 创建检测器
    if detector is None:
        if 'wind' in label_type:
            ws_col = windspeed_col or 'windspeed_100m_盐城市'
        else:
            ws_col = None
        
        detector = PowerAnomalyDetector(
            windspeed_col=ws_col,
            radiation_col='shortwave_radiation_instant' if 'solar' in label_type else None
        )
    
    # 检测异常
    anomalies = detector.detect_all(df, power_col, label_type)
    
    # 汇总并剔除
    all_anomaly_indices = detector.get_total_anomaly_indices(anomalies)
    
    if len(all_anomaly_indices) > 0:
        before_count = len(df)
        df_clean = df.drop(index=all_anomaly_indices)
        after_count = len(df_clean)
        
        logger.info(f"[异常清洗] 共剔除 {before_count - after_count} 个异常样本，"
                    f"剩余 {after_count} 个样本")
        
        return df_clean
    
    return df
```

### 2. 集成到 Pipeline

**修改文件**：`api/model/pipeline.py`

**修改位置**：约第242-247行（在现有过滤逻辑之后，构建 pdf_data_valid 时）

```python
# === 原有代码（第239-247行） ===
if self._get_label and self._train:
    pdf_data = pdf_data[~pdf_data[label].isna()]
    pdf_data = pdf_data[pdf_data[label] >= 50]
    pdf_data_valid = pdf_data[['date_time', 'time_idx'] + feature_list + [label]].copy()
elif self._get_label:
    pdf_data = pdf_data[~pdf_data[label].isna()]
    pdf_data_valid = pdf_data[['date_time', 'time_idx'] + feature_list + [label]].copy()
else:
    pdf_data_valid = pdf_data[['date_time', 'time_idx'] + feature_list].copy()

# === 新增：异常数据清洗（仅训练时执行） ===
if self._train and self._get_label:
    from .anomaly_detector import PowerAnomalyDetector
    
    # 根据标签类型选择风速列
    windspeed_col = None
    if 'wind' in label:
        # 尝试使用盐城风速（江苏最大风电基地）
        for col in pdf_data_valid.columns:
            if 'windspeed_100m' in col and '盐城' in col:
                windspeed_col = col
                break
        if windspeed_col is None:
            # 回退：使用任意风速列
            windspeed_cols = [c for c in pdf_data_valid.columns if 'windspeed_100m' in c]
            if windspeed_cols:
                windspeed_col = windspeed_cols[0]
    
    # 创建检测器
    detector = PowerAnomalyDetector(
        windspeed_col=windspeed_col,
        radiation_col='shortwave_radiation_instant' if 'solar' in label else None
    )
    
    # 执行异常检测
    anomalies = detector.detect_all(
        df=pdf_data_valid,
        power_col=label,
        label_type=label_type
    )
    
    # 汇总异常索引
    all_anomaly_indices = detector.get_total_anomaly_indices(anomalies)
    
    # 剔除异常样本
    if len(all_anomaly_indices) > 0:
        before_count = len(pdf_data_valid)
        pdf_data_valid = pdf_data_valid.drop(index=all_anomaly_indices).copy()
        after_count = len(pdf_data_valid)
        logger.info(f"[异常清洗] {label}: 剔除 {before_count - after_count} 个异常样本")
```

---

## 四、检测阈值参数

| 参数名 | 默认值 | 说明 |
|--------|-------|-----|
| `tolerance_ratio` | 2.0 | 功率曲线异常容忍倍数 |
| `max_change_per_quarter` | 500 MW | 突变检测阈值（15分钟变化） |
| `frozen_window` | 12 | 数据冻结检测窗口（点数，=3小时） |
| `radiation_threshold` | 1.0 W/m² | 光伏夜间辐照度阈值 |

---

## 五、验证建议

### 1. 日志检查

运行训练流程后，观察日志输出：
```
[异常检测] 功率曲线异常: 发现 X 个异常点
[异常检测] 突变异常: 发现 Y 个异常点
[异常清洗] 共剔除 Z 个异常样本
```

### 2. 数据统计对比

对比清洗前后的数据分布：
- 均值、中位数、分位数变化
- 最大值、最小值变化
- 缺失率变化

### 3. 功率曲线可视化

绘制清洗前后的 P-v³ 散点图：
- 检查异常点是否被剔除
- 验证物理关系是否更加清晰

### 4. 模型效果对比

对比清洗前后模型验证集指标：
- MAE（平均绝对误差）
- RMSE（均方根误差）
- 准确率

---

## 六、扩展建议

### 1. 多风速列融合

可使用加权平均风速计算功率系数：
```python
# 盐城、南通、连云港加权（基于装机容量）
windspeed_weighted = (
    1121 * windspeed_盐城 + 
    550 * windspeed_南通 + 
    350 * windspeed_连云港
) / (1121 + 550 + 350)
```

### 2. 异常报告生成

可选生成异常数据统计报告：
- 异常类型分布
- 异常时段分布
- 异常样本详情（CSV/Excel）

### 3. 光伏温度衰减

光伏功率受温度影响，可扩展检测：
- 高温时段（温度 > 25°C）功率衰减关系
- 低温时段功率异常

### 4. 分时段检测

不同时段可使用不同阈值：
- 白天/夜间
- 高峰时段/低谷时段
- 季节性差异