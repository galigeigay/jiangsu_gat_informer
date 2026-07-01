"""
出力数据异常检测器
基于物理关系检测训练数据中的异常样本

核心原理：
- 风电：功率 P 与风速 v 的三次方成正比，即 P ∝ v³
- 光伏：功率与辐照度成正比，夜间辐照度接近 0 时功率应为 0
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Set
from api.io.logger import setup_logger

logger = setup_logger(__name__)


class PowerAnomalyDetector:
    """出力数据异常检测器 - 基于物理关系"""

    def __init__(self,
                 windspeed_col: Optional[str] = 'windspeed_100m_盐城市',
                 radiation_col: Optional[str] = None,
                 wind_power_coefficient: Optional[float] = None,
                 tolerance_ratio: float = 3.5,
                 max_change_per_quarter: float = 3000.0,
                 frozen_window: int = 12,
                 radiation_threshold: float = 1.0,
                 min_power_for_frozen: float = 100.0,
                 min_windspeed_for_low_power: float = 5.0,
                 min_power_for_low_power_check: float = 100.0,
                 min_windspeed_for_high_check: float = 6.0,
                 capacity: Optional[float] = None,
                 ):
        """
        Args:
            windspeed_col: 用于检测的风速特征列（默认盐城百米风速）
            radiation_col: 用于检测的辐照度特征列
            wind_power_coefficient: 功率/风速³系数，若为None则自动拟合
            tolerance_ratio: 功率曲线异常容忍倍数（实测/理论 > 此值或 < 1/此值视为异常）。
                由于单城风速代表全省、且单一立方模型在额定速以上失效，建议设较宽松值（3.0~4.0）。
            max_change_per_quarter: 突变检测阈值（MW/15min）
            frozen_window: 数据冻结检测窗口（点数，默认12=3小时）
            radiation_threshold: 光伏夜间辐照度阈值（W/m²）
            min_power_for_frozen: 数据冻结检测的最小功率阈值（MW）
            min_windspeed_for_low_power: 低功率异常检测的最小风速阈值（m/s）
            min_power_for_low_power_check: 低功率异常检测的最小功率阈值（MW）
            min_windspeed_for_high_check: anomaly_high 检测的最小风速阈值（m/s）。
                在 v 很小时 k·v³ 极小，ratio>tol 极易触发假阳，故对低风速跳过 anomaly_high。
            capacity: 装机容量上限（MW），用于把理论功率封顶为 min(k·v³, capacity)，避免
                额定风速以上 k·v³ 远超装机导致 anomaly_low 假阳。None 表示不封顶（保留原行为）。
        """
        self.windspeed_col = windspeed_col
        self.radiation_col = radiation_col
        self.wind_power_coefficient = wind_power_coefficient
        self.tolerance_ratio = tolerance_ratio
        self.max_change_per_quarter = max_change_per_quarter
        self.frozen_window = frozen_window
        self.radiation_threshold = radiation_threshold
        self.min_power_for_frozen = min_power_for_frozen
        self.min_windspeed_for_low_power = min_windspeed_for_low_power
        self.min_power_for_low_power_check = min_power_for_low_power_check
        self.min_windspeed_for_high_check = min_windspeed_for_high_check
        self.capacity = capacity

    def fit_wind_power_curve(self, df: pd.DataFrame,
                             power_col: str,
                             windspeed_col: str) -> Optional[float]:
        """
        拟合功率-风速³关系系数 P = k × v³

        Args:
            df: 数据DataFrame
            power_col: 功率列名
            windspeed_col: 风速列名

        Returns:
            k: 功率系数（P/v³的中位数），若样本不足则返回None
        """
        # 重置索引以避免重复索引问题
        df_reset = df.reset_index(drop=True)

        # 过滤有效范围（切入风速3m/s到切出风速25m/s）
        windspeed = df_reset[windspeed_col].values
        power = df_reset[power_col].values

        valid_mask = (
            (windspeed >= 3) &
            (windspeed <= 25) &
            (power > 0)
        )
        valid_windspeed = windspeed[valid_mask]
        valid_power = power[valid_mask]

        if len(valid_power) < 100:
            logger.warning(f"[拟合功率系数] 有效样本数不足（{len(valid_power)}），无法拟合")
            return None

        # 计算理论功率系数 k = P / v³
        v3 = valid_windspeed ** 3
        k_values = valid_power / v3

        # 取中位数作为代表系数（鲁棒性）
        k = np.median(k_values)
        logger.info(f"[拟合功率系数] k = P/v³ = {k:.4f} (样本数: {len(valid_power)})")

        return k

    def detect_wind_power_anomaly(self,
                                   df: pd.DataFrame,
                                   power_col: str,
                                   windspeed_col: str) -> pd.Index:
        """
        检测风电功率与风速³关系异常

        异常条件：
        1. ratio > tolerance_ratio: 功率远超理论值
        2. ratio < 1/tolerance_ratio 且风速>阈值 且功率>阈值: 功率远低于理论值

        Returns:
            异常样本的索引（基于原始df的索引）
        """
        # 拟合或使用已有系数
        if self.wind_power_coefficient is None:
            k = self.fit_wind_power_curve(df, power_col, windspeed_col)
            if k is None:
                return pd.Index([])
        else:
            k = self.wind_power_coefficient

        # 使用 numpy 数值计算，避免 DataFrame 索引对齐问题
        windspeed = df[windspeed_col].values
        power = df[power_col].values

        # 计算理论功率: P = k * v^3, 但封顶到装机容量, 避免额定速以上 k·v³ 远超装机
        theoretical_power = k * windspeed ** 3
        if self.capacity is not None and self.capacity > 0:
            theoretical_power = np.minimum(theoretical_power, float(self.capacity))

        # 计算偏差比（避免除零）
        ratio = np.where(
            theoretical_power > 0,
            power / theoretical_power,
            np.inf
        )

        # 异常条件: anomaly_high 在低风速下 k·v³ 很小, ratio 失真, 故加守门
        anomaly_high = (ratio > self.tolerance_ratio) & (windspeed > self.min_windspeed_for_high_check)
        anomaly_low = (
            (ratio < 1 / self.tolerance_ratio) &
            (windspeed > self.min_windspeed_for_low_power) &
            (power > self.min_power_for_low_power_check)
        )

        anomaly_mask = anomaly_high | anomaly_low
        anomaly_indices = df.index[anomaly_mask]

        cap_info = f", capacity_cap={self.capacity}" if (self.capacity is not None and self.capacity > 0) else ""
        logger.info(f"[功率曲线异常] 发现 {len(anomaly_indices)} 个异常点 "
                    f"(高于理论: {np.sum(anomaly_high)}, 低于理论: {np.sum(anomaly_low)}; "
                    f"tol={self.tolerance_ratio}, v_high_gate={self.min_windspeed_for_high_check}{cap_info})")

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
        # 使用 numpy 数值计算，避免 DataFrame 索引对齐问题
        radiation = df[radiation_col].values
        power = df[power_col].values

        anomaly_mask = (radiation < self.radiation_threshold) & (power > 50)
        anomaly_indices = df.index[anomaly_mask]

        logger.info(f"[光伏夜间异常] 发现 {len(anomaly_indices)} 个异常点")

        return anomaly_indices

    def detect_negative_values(self, df: pd.DataFrame, power_col: str) -> pd.Index:
        """
        检测负值异常

        Returns:
            异常样本的索引
        """
        anomaly_mask = df[power_col].values < 0
        anomaly_indices = df.index[anomaly_mask]
        logger.info(f"[负值异常] 发现 {len(anomaly_indices)} 个异常点")
        return anomaly_indices

    def detect_sudden_jumps(self, df: pd.DataFrame, power_col: str) -> pd.Index:
        """
        检测突变异常

        条件：15分钟功率变化 > max_change_per_quarter

        Returns:
            异常样本的索引
        """
        power = df[power_col].values
        power_diff = np.abs(np.diff(power, prepend=power[0]))
        anomaly_mask = power_diff > self.max_change_per_quarter
        anomaly_indices = df.index[anomaly_mask]

        logger.info(f"[突变异常] 发现 {len(anomaly_indices)} 个异常点 "
                    f"(阈值: {self.max_change_per_quarter} MW/15min)")

        return anomaly_indices

    def detect_frozen_values(self, df: pd.DataFrame, power_col: str) -> pd.Index:
        """
        检测数据冻结

        条件：连续 frozen_window 点标准差为 0 且 功率 > min_power_for_frozen

        Returns:
            异常样本的索引
        """
        power = df[power_col].values

        # 计算滑动窗口标准差（手动实现，避免 pandas rolling 的索引问题）
        rolling_std = np.zeros(len(power))
        for i in range(len(power)):
            if i < self.frozen_window - 1:
                rolling_std[i] = np.std(power[:i+1]) if i > 0 else 0
            else:
                rolling_std[i] = np.std(power[i-self.frozen_window+1:i+1])

        anomaly_mask = (rolling_std == 0) & (power > self.min_power_for_frozen)
        anomaly_indices = df.index[anomaly_mask]

        logger.info(f"[数据冻结异常] 发现 {len(anomaly_indices)} 个异常点 "
                    f"(窗口: {self.frozen_window}点={self.frozen_window/4}小时)")

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
        anomalies: Dict[str, pd.Index] = {}

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
            else:
                logger.warning(f"[风电检测] 风速列 '{self.windspeed_col}' 不存在于数据中")

        # 光伏专用检测
        if 'solar' in label_type and self.radiation_col:
            if self.radiation_col in df.columns:
                anomalies['solar_night'] = self.detect_solar_night_anomaly(
                    df, power_col, self.radiation_col
                )
            else:
                logger.warning(f"[光伏检测] 辐照度列 '{self.radiation_col}' 不存在于数据中")

        return anomalies

    def get_total_anomaly_indices(self, anomalies: Dict[str, pd.Index]) -> pd.Index:
        """
        合并所有异常索引（去重）

        Args:
            anomalies: 检测结果字典

        Returns:
            合并后的异常索引
        """
        all_indices: Set[int] = set()
        for anomaly_type, indices in anomalies.items():
            all_indices.update(indices.tolist())

        return pd.Index(list(all_indices))

    def summary(self, anomalies: Dict[str, pd.Index]) -> str:
        """
        生成异常检测摘要报告

        Args:
            anomalies: 检测结果字典

        Returns:
            摘要文本
        """
        lines = ["=== 异常检测摘要 ==="]
        total = 0
        for anomaly_type, indices in anomalies.items():
            count = len(indices)
            total += count
            lines.append(f"  {anomaly_type}: {count} 个异常点")
        lines.append(f"  总计（去重前）: {total} 个异常点")
        lines.append(f"  总计（去重后）: {len(self.get_total_anomaly_indices(anomalies))} 个异常点")
        return "\n".join(lines)


def clean_anomalies(df: pd.DataFrame,
                    power_col: str,
                    label_type: str,
                    detector: Optional[PowerAnomalyDetector] = None,
                    windspeed_col: Optional[str] = None,
                    radiation_col: Optional[str] = None) -> pd.DataFrame:
    """
    清洗异常数据的便捷函数

    Args:
        df: 数据DataFrame
        power_col: 功率列名
        label_type: 标签类型 ('wind' 或 'solar')
        detector: 检测器实例（若为None则自动创建）
        windspeed_col: 风速列名（用于风电检测）
        radiation_col: 辐照度列名（用于光伏检测）

    Returns:
        清洗后的DataFrame
    """
    # 创建检测器
    if detector is None:
        if 'wind' in label_type:
            ws_col = windspeed_col or 'windspeed_100m_盐城市'
        else:
            ws_col = None

        rad_col = radiation_col if 'solar' in label_type else None

        detector = PowerAnomalyDetector(
            windspeed_col=ws_col,
            radiation_col=rad_col
        )

    # 检测异常
    anomalies = detector.detect_all(df, power_col, label_type)

    # 汇总并剔除
    all_anomaly_indices = detector.get_total_anomaly_indices(anomalies)

    if len(all_anomaly_indices) > 0:
        before_count = len(df)
        df_clean = df.drop(index=all_anomaly_indices).copy()
        after_count = len(df_clean)

        logger.info(f"[异常清洗] 共剔除 {before_count - after_count} 个异常样本，"
                    f"剩余 {after_count} 个样本")

        return df_clean

    logger.info("[异常清洗] 未发现异常样本")
    return df