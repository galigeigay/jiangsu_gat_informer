from copy import deepcopy
from datetime import datetime,timedelta
import os
from typing import (List, Sequence)
import pandas as pd
from .baseclass import BaseModel, BasePipeline
from .constants import (
    REALTIME_FEATURE_LIST,
    WEATHER_FEATURE_LIST,
    WIND_WEATHER_FEATURE_LIST,
    SOLAR_WEATHER_FEATURE_LIST,
    JIANGSU_CITY,
    JIANGSU_CITY2_WIND,
    JIANGSU_CITY2_SOLAR,
    JIANGSU_SEA,
    WEATHER_FEATURE_SEA_LIST,
    MINIO_SERVER,
    get_latest_province_capacity
)
import numpy as np
from .data import get_data
from .utils import get_available_model_filename, get_sun_times, set_solar_zero
from .xgbmodel import XGBModel
from api.io.logger import setup_logger
from anydb.miniolib import upload_file
from sklearn.model_selection import train_test_split
# from df_to_mongo import resample_to_15min
from astral import LocationInfo
from astral.sun import sun
logger = setup_logger(__name__)

class ModelPipeline(BasePipeline):
    def __init__(self,
                 model_type: str,
                 train: bool,
                 label_list: Sequence[str],
                 feature_list: Sequence[str],
                 trade_type: str,
                 start_date: str,
                 end_date: str,
                 pipeline_version = '',
                 local_dir = 'local',
                 load_strictly = False,
                 get_label = True,
                 **kwargs
                 ):
        """核心模型流程, 支持供需/价格的训练/预测

        Args:
            model_type (str): 模型类型, 支持: 'xgboost'
            train (str): 是否训练
            label_list (Sequence[str]): 模型标签, 支持: 'new_energy_wind', 'new_energy_solar', 
                'new_energy_wind_ahead', 'new_energy_solar_ahead',
            feature_list (Sequence[str]): 模型标签, 支持: 'provincial_load', 'tie_line',
                'new_energy_wind', 'new_energy_solar', 'non_market' 及 以上标签加'_ahead' 
            trade_type (str): 交易类型, 支持: 'ahead', 'realtime'
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
            pipeline_version (str, optional): 指定模型的版本号, 习惯格式: '20241023-134746', 即年月日-时分秒
            local_dir (str, optional): 本地模型文件保存路径
            load_strictly (bool, optional): 如果设为True, 只会在local_dir读取模型版本号严格为pipeline_version的模型文件;
                如果设为False, 优先读取模型版本号为pipeline_version的模型文件, 如果读不到则读取时间最近的模型文件
            get_label (bool, optional): 是否获取模型标签, 一般训练和测试时为True, 对未来预测时为False
            use_predict (bool, optional): 是否用预测值代替交易中心披露值, 比如用预测的供需代替边界条件
        """
        super().__init__(class_name='ModelPipeline')
        assert model_type in ['xgb', 'xgboost', 'gat_informer'], "传入参数model_type不支持: {}".format(model_type)
        self._model_type = model_type
        self._models = {}
        self._train = train
        if 'clear_price' in label_list and len(label_list) > 1:
            raise ValueError("传入参数label_list含有clear_price时, 不应包含其他值")
        self._label_list = label_list
        self._feature_list = feature_list
        assert trade_type in ['ahead', 'realtime'], "传入参数trade_type不支持: {}".format(trade_type)
        self._trade_type = trade_type
        self._start_date = start_date
        self._end_date = end_date
        if not pipeline_version:
            cur_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            pipeline_version = cur_timestamp
        self._pipeline_version = pipeline_version
        self._local_dir = local_dir
        self._load_strictly = load_strictly
        self._get_label = get_label
        self._predict_results = {}

        self._meteo_scope = kwargs.get('meteo_scope', 'station')
        self._meteo_weighted = kwargs.get('meteo_weighted', False)
        # self._version_num = kwargs.get('version_num', 0)
        # self._meteo_source = kwargs.get('meteo_source', 'ENS_EXTEND')
        # self._capa_rev = kwargs.get('capa_rev', True)
        # self._avg_data = kwargs.get('avg_data', False)
        self._kwargs = kwargs

    @property
    def predict_results(self):
        return self._predict_results
    
    def get_feature_list(self, label: str, meteo_scope: str, pdf_data):
        wind_weather_features = deepcopy(WIND_WEATHER_FEATURE_LIST)
        solar_weather_features = deepcopy(SOLAR_WEATHER_FEATURE_LIST)

        wind_weather_sea_features = deepcopy(WEATHER_FEATURE_SEA_LIST)
        solar_weather_sea_features = deepcopy(WEATHER_FEATURE_SEA_LIST)

        if self._meteo_weighted:
            wind_weather_features.remove("windspeed_200m")
            solar_weather_features.remove("windspeed_200m")

        feature_list = []
        feature_list2 = []
        feature_list_sea = []
        if meteo_scope == 'station':
            if 'wind' in label:
                feature_list += [f'{feat}_{stc}' for feat in wind_weather_features for stc in WIND_POWER_STATION_CODE_LIST]
            elif 'solar' in label:
                feature_list += [f'{feat}_{stc}' for feat in solar_weather_features for stc in SOLAR_POWER_STATION_CODE_LIST]
            else:
                feature_list += WEATHER_FEATURE_LIST
        elif meteo_scope == 'city':
            city_list = JIANGSU_CITY2_WIND if 'wind' in label else JIANGSU_CITY2_SOLAR if 'solar' in label else JIANGSU_CITY
            suffix = '_wind' if 'wind' in label and self._meteo_weighted else '_solar' if 'solar' in label and self._meteo_weighted else ''

            # 市级特征：使用原始气象城市拼接列，不再使用装机容量加权后的 capa_w 列
            main_features = wind_weather_features if 'wind' in label else solar_weather_features
            for feat in main_features:
                for city in city_list:
                    raw_col = f'{feat}_{city}{suffix}'
                    if raw_col in pdf_data.columns:
                        feature_list.append(raw_col)

            if 'wind' in label:
                feature_list2 = ['data_hour_sin','data_hour_cos','data_day','data_month_sin','data_month_cos','data_minute_sin','data_minute_cos', \
                                'windspeed_100m_盐城_扬州_diff', 'windspeed_100m_南通_扬州_diff', \
                                'windspeed_100m_盐城_淮安_diff','windspeed_100m_连云港_徐州_diff', \
                                'windspeed_100m_southern_mean','windspeed_100m_northern_mean']#,'holiday_tag'

                # feature_list3 = [col for col in pdf_data.columns
                #                  if ('_lag_' in col)
                #                  and ('windspeed_' in col)
                #                  and ('_capa_w_' not in col)]  # 只保留原始气象列的滞后特征

                
                
                feature_list.extend(feature_list2)
                # feature_list.extend(feature_list3)
                

            else:
                # feature_list2 = ['data_hour_sin','data_hour_cos','data_day', ]

                feature_list2 = ['data_hour_sin','data_hour_cos','data_day',
                                 'cloudcover_x_shortwave',
                                 'temp_above_25',
                                 'shortwave_rad_lag_1',
                                 'shortwave_rad_lag_3',
                                 'shortwave_rad_diff_1h',
                                 'shortwave_rad_diff_3h',
                                 ] #'data_minute_sin','data_minute_cos',

                feature_list.extend(feature_list2)

        elif meteo_scope == 'province' and self._meteo_weighted:
            suffix = '_wind' if 'wind' in label else '_solar' if 'solar' in label else ''
            feature_list += [f'{feat}{suffix}' for feat in (wind_weather_features if 'wind' in label else solar_weather_features)]
        else:
            feature_list += WEATHER_FEATURE_LIST
        return feature_list

    def run(self, **kwargs):
        # 获取交易中心数据和气象数据
        data_feat_list = self._feature_list + self._label_list if self._get_label else self._feature_list
        pdf_data_src = get_data(
            feature_list=data_feat_list,
            start_date=self._start_date,
            end_date=self._end_date,
            train_flag=self._train,
            **self._kwargs
        )
        # print(pdf_data_src)
        # cond = pdf_data_src['new_energy_wind_ahead'] < 100
        # print(pdf_data_src.loc[cond])
        # exit()
        # 依次处理每个标签
        for label in self._label_list:
            pdf_data = pdf_data_src.copy()
            label_type = 'wind' if 'wind' in label else 'solar' if 'solar' in label else 'default'
            if self._model_type in ['xgb', 'xgboost']:
                self._models[label] = XGBModel(classification=False, label_type=label_type)
            elif self._model_type == 'gat_informer':
                from .gat_informer_model import GATInformerModel
                self._models[label] = GATInformerModel(
                    label_type=label_type,
                    seq_len=96,
                    d_gat=32,
                    gat_heads=4,
                    d_model=64, #64,128
                    informer_layers=2,
                    informer_heads=4,
                    d_ff=128, #128,256
                    epochs=100,
                    batch_size=32,
                    lr=1e-4, #5e-5,
                    # Round-2: 调轻正则 (dropout 0.4->0.25, drop_edge 0.3->0.15,
                    # temporal_noise 0.1->0.03, weight_decay 1e-3->5e-4, corr_threshold 0.6->0.45)
                    # 用于缓解 Val/Fold3 幅度压缩 (pred_std/y_std=0.75-0.84 -> ~0.9).
                    dropout=self._kwargs.get('gat_dropout', 0.25),
                    corr_threshold=self._kwargs.get('gat_corr_threshold', 0.45),
                    val_ratio=0.2,
                    patience=16, #3,
                    rolling_backtest=self._kwargs.get('gat_rolling_backtest', True),
                    rb_n_folds=self._kwargs.get('gat_rb_n_folds', 3),
                    rb_val_ratio=self._kwargs.get('gat_rb_val_ratio', 0.15),
                    loss_delta=self._kwargs.get('gat_loss_delta', 1000.0),
                    low_power_threshold=self._kwargs.get('gat_low_power_threshold', 2500.0),
                    high_power_threshold=self._kwargs.get('gat_high_power_threshold', 8000.0),
                    low_power_weight=self._kwargs.get('gat_low_power_weight', 2),
                    high_power_weight=self._kwargs.get('gat_high_power_weight', 4), #2.5, 1.8
                    max_sample_weight=self._kwargs.get('gat_max_sample_weight', 3.0),
                    mape_weight=self._kwargs.get('gat_mape_weight', 0.25),
                    drop_edge=self._kwargs.get('gat_drop_edge', 0.15),
                    temporal_noise=self._kwargs.get('gat_temporal_noise', 0.03),
                    weight_decay=self._kwargs.get('gat_weight_decay', 5e-4),
                    clip_grad=self._kwargs.get('gat_clip_grad', 1.0),
                    # Round-2: 幅度损失 + 近期加权 + 早停按 val_accuracy + cap 0.98
                    std_loss_weight=self._kwargs.get('gat_std_loss_weight', 0.03),
                    corr_loss_weight=self._kwargs.get('gat_corr_loss_weight', 0.02),
                    recent_weight=self._kwargs.get('gat_recent_weight', 2.0),
                    recent_tail_ratio=self._kwargs.get('gat_recent_tail_ratio', 0.30),
                    early_stop_metric=self._kwargs.get('gat_early_stop_metric', 'val_accuracy'),
                    capacity_clip_ratio=self._kwargs.get('gat_capacity_clip_ratio', 0.98),
                )

            with_ahead_tag = 'noa' if not self._feature_list else 'wta'
            meteo_tag = 'weighted' if self._meteo_weighted else 'unweighted'
            ext = 'json' if self._model_type in ['xgb', 'xgboost'] else 'pt'
            model_filename = os.path.join(self._local_dir,
                                          '{}-{}-{}-{}-{}.{}'.format(label, self._trade_type, self._kwargs.get('task_name', 'by_day'), with_ahead_tag,
                                                                    meteo_tag, ext))
            if self._trade_type == 'ahead':
                non_weather_feat = []
            else:
                non_weather_feat = self._feature_list
            
            feature_list = deepcopy(non_weather_feat)
            meteo_scopes = self._meteo_scope.split("_")
            for meteo_scope in meteo_scopes:
                feature_list += self.get_feature_list(label, meteo_scope,pdf_data)

            if self._get_label and self._train:
                pdf_data = pdf_data[~pdf_data[label].isna()]
                pdf_data = pdf_data[pdf_data[label] >= 50]
                pdf_data_valid = pdf_data[['date_time', 'time_idx'] + feature_list + [label]].copy()
            elif self._get_label:
                pdf_data = pdf_data[~pdf_data[label].isna()]
                pdf_data_valid = pdf_data[['date_time', 'time_idx'] + feature_list + [label]].copy()
            else:
                pdf_data_valid = pdf_data[['date_time', 'time_idx'] + feature_list].copy()
            # print(pdf_data_valid['windspeed_100m_盐城市'])
            # exit(0)
            # 空值处理：数值型特征优先线性插值，避免整行删除造成数据浪费
            # pdf_data_valid = pdf_data_valid.sort_values(by='time_idx').reset_index(drop=True)
            nan_columns = pdf_data_valid.columns[pdf_data_valid.isna().any()].to_list()
            if nan_columns:
                logger.info(f"包含NaN值列:{nan_columns}")
                numeric_cols = pdf_data_valid.select_dtypes(include=['number']).columns
                for col in set(nan_columns) & set(numeric_cols):
                    if col == label:
                        continue  
                    pdf_data_valid[col] = pdf_data_valid[col].interpolate(method='linear')

                feat_cols_for_fill = [c for c in pdf_data_valid.columns if c != label]
                pdf_data_valid[feat_cols_for_fill] = pdf_data_valid[feat_cols_for_fill].ffill().bfill()

                if self._train and label in pdf_data_valid.columns:
                    before = len(pdf_data_valid)
                    pdf_data_valid = pdf_data_valid[pdf_data_valid[label].notna()].copy()
                    dropped = before - len(pdf_data_valid)
                    if dropped > 0:
                        logger.info(f"训练时丢弃 label({label}) 缺失的 {dropped} 个样本（避免插值污染）")

                still_nan = pdf_data_valid.columns[pdf_data_valid.isna().any()].to_list()
                if still_nan:
                    logger.warning(f"插值后仍有NaN列:{still_nan}，执行dropna")
                    pdf_data_valid.dropna(inplace=True)

            # === 异常数据清洗（仅训练时执行） ===
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
                            logger.info(f"[异常检测] 使用风速列: {windspeed_col}")

                # 辐照度列（光伏检测）
                radiation_col = None
                if 'solar' in label:
                    radiation_cols = [c for c in pdf_data_valid.columns if 'shortwave_radiation' in c]
                    if radiation_cols:
                        radiation_col = radiation_cols[0]
                        logger.info(f"[异常检测] 使用辐照度列: {radiation_col}")

                # 创建检测器并执行异常检测: 把省级装机传入用于理论功率封顶, 避免额定速以上假阳
                province_capacity = get_latest_province_capacity(label_type)
                detector = PowerAnomalyDetector(
                    windspeed_col=windspeed_col,
                    radiation_col=radiation_col,
                    capacity=province_capacity if province_capacity > 0 else None,
                )

                anomalies = detector.detect_all(
                    df=pdf_data_valid,
                    power_col=label,
                    label_type=label_type
                )

                # 汇总异常索引并剔除
                all_anomaly_indices = detector.get_total_anomaly_indices(anomalies)

                if len(all_anomaly_indices) > 0:
                    before_count = len(pdf_data_valid)
                    pdf_data_valid = pdf_data_valid.drop(index=all_anomaly_indices).copy()
                    after_count = len(pdf_data_valid)
                    logger.info(f"[异常清洗] {label}: 剔除 {before_count - after_count} 个异常样本")

            # pdf_data_valid.to_excel('./pdf_weather_pre.xlsx')
            logger.info(f"样本数量:{pdf_data_valid.shape[0]}")
            if self._train:
                assert self._get_label, "训练模型时参数get_label应为True"
                logger.info("训练模型: {}".format(label))

                # 执行模型训练
                run_model(
                    model=self._models[label],
                    train=True,
                    pdf_data=pdf_data_valid,
                    label=label,
                    # weights=weights
                )
                if self._model_type == 'gat_informer':
                    test_feature = pdf_data_valid.drop(columns=[label]).copy()
                    pred_before = self._models[label].predict(feature=test_feature)
                    logger.info(
                        f"[自检] 加载前预测 mean={np.mean(pred_before):.1f}, "
                        f"std={np.std(pred_before):.1f}, min={np.min(pred_before):.1f}, max={np.max(pred_before):.1f}"
                    )

                logger.info(f"保存模型到: {model_filename}")
                self._models[label].save(model_name=model_filename)
                # logger.info(f"上传模型到minio: models/tree_for_power/wjq/mengxi/")
                # upload_file(MINIO_SERVER, model_filename, f'models/tree_for_power/wjq/mengxi/{os.path.basename(model_filename)}')
            logger.info(f"读取模型文件: {model_filename}")
            self._models[label].load(model_name=model_filename)

            if self._train and self._model_type == 'gat_informer':
                pred_after = self._models[label].predict(feature=test_feature)
                logger.info(
                    f"[自检] 加载后预测 mean={np.mean(pred_after):.1f}, "
                    f"std={np.std(pred_after):.1f}, min={np.min(pred_after):.1f}, max={np.max(pred_after):.1f}"
                )
                # 训练完成后自动可视化 GAT 注意力，检测盐城是否垄断
                vis_dir = os.path.join(self._local_dir, 'visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(
                    vis_dir,
                    f'gat_attention_{label}_{self._trade_type}_{self._kwargs.get("task_name", "by_day")}.png'
                )
                try:
                    self._models[label].visualize_attention(
                        feature=pdf_data_valid,
                        n_samples=4,
                        sample_mode="high_power",
                        save_path=vis_path,
                    )
                except Exception as e:
                    logger.warning(f"[GAT Attention] 可视化失败（非关键错误）: {e}")

            pdf_rst = run_model(
                model=self._models[label],
                train=False,
                pdf_data=pdf_data_valid,
                label=label,
            )

            pdf_rst = pd.concat([pdf_data[['date_time', 'version', 'version_diff', 'meteo_source']], pdf_rst], axis=1)
            logger.info(f"Finish predicting {label}")
            pdf_rst = pdf_rst.loc[:, ~pdf_rst.columns.duplicated()]

            if 'solar' in label:
                pdf_rst['date_time'] = pdf_rst['date_time'].astype(str)
                pdf_rst['_date'] = pd.to_datetime(pdf_rst['date_time']).dt.date
                unique_dates = pdf_rst['_date'].unique()
                date_sun_map = {}
                for d in unique_dates:
                    sunrise, sunset = get_sun_times(d)
                    date_sun_map[d] = (sunrise, sunset)

                pdf_rst['_sunrise'] = pdf_rst['_date'].map(lambda d: date_sun_map[d][0])
                pdf_rst['_sunset'] = pdf_rst['_date'].map(lambda d: date_sun_map[d][1])
                pdf_rst['_hour_min'] = pdf_rst['date_time'].str[-8:-3]  # 'HH:MM'

                pred_col = f'{label}_pred'
                mask_night = (pdf_rst['_hour_min'] <= pdf_rst['_sunrise']) | (pdf_rst['_hour_min'] >= pdf_rst['_sunset'])
                pdf_rst.loc[mask_night, pred_col] = 0
                pdf_rst.loc[pdf_rst[pred_col] < 0, pred_col] = 0

                pdf_rst = pdf_rst.drop(columns=['_date', '_sunrise', '_sunset', '_hour_min'])
            else:
                pred_col = '{}_pred'.format(label)
                pdf_rst[pred_col] = pdf_rst[pred_col].clip(lower=0)
                province_capacity = get_latest_province_capacity(label_type)
                # Round-2: cap ratio 0.98 与模型侧 _capacity_cap 保持一致, 覆盖历史峰值(~22138)又不越装机
                cap_ratio = self._kwargs.get('gat_capacity_clip_ratio', 0.98)
                if province_capacity > 0:
                    pdf_rst[pred_col] = pdf_rst[pred_col].clip(upper=province_capacity * cap_ratio)

            if self._train:
                self._predict_results[label] = pdf_rst
            else:
                self._predict_results[label] = pdf_rst
        return


def run_model(model: BaseModel,
              train: bool,
              pdf_data: pd.DataFrame,
              label: str,
              ):
    if label in pdf_data:
        pdf_feature = pdf_data.drop(columns=[label])
    else:
        pdf_feature = pdf_data.copy()

    # gat_informer 需要 time_idx 做时序排序，xgboost 不需要
    if isinstance(model, XGBModel):
        feature_columns = [col for col in pdf_feature.columns if col not in ['time_idx']]
        pdf_feature = pdf_feature[feature_columns]
    else:
        feature_columns = [col for col in pdf_feature.columns]
        pdf_feature = pdf_feature[feature_columns]

    if train:
        pdf_label = pdf_data[[label]]
        model.train(feature=pdf_feature, label=pdf_label)
        return None

    else:
        predict_rst = model.predict(feature=pdf_feature)
        pdf_predict = pdf_data.copy()

        pdf_predict['{}_pred'.format(label)] = predict_rst.tolist()

        return pdf_predict