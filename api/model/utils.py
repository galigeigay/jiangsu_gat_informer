import datetime
import os

from astral.sun import sun
from astral import LocationInfo
import numpy as np
import pandas as pd

from typing import Union 
from ..io.utils import str_timestamp_to_str


def calculate_time_idx(info_time):
    """
    将'00:15'-'24:00'转换为1-96, 作为模型样本的特征
    """
    if info_time == '24:00':
        return 96
    else:
        minutes = (datetime.datetime.strptime(info_time, '%H:%M') - datetime.datetime.strptime('00:00', '%H:%M')).total_seconds() / 60.0
        return int(minutes / 15)


def calculate_accuracy(val_pred : Union[np.ndarray, pd.DataFrame, pd.Series],
                       val_true : Union[np.ndarray, pd.DataFrame, pd.Series],
                       threshold=10):
    """
    计算准确率, 即1-MAPE
    """
    if isinstance(val_pred, pd.DataFrame) or isinstance(val_pred, pd.Series):
        val_pred = np.squeeze(val_pred.to_numpy())
        assert val_pred.ndim == 1

    if isinstance(val_true, pd.DataFrame) or isinstance(val_true, pd.Series):
        val_true = np.squeeze(val_true.to_numpy())
        assert val_true.ndim == 1

    assert isinstance(val_pred, np.ndarray) and isinstance(val_true, np.ndarray), \
        "传入参数val_pred, val_true类型不支持: {} {}".format(type(val_pred), type(val_true))

    val_pred = val_pred.copy()
    val_true = val_true.copy()

    mask = ~np.isnan(val_pred) & ~np.isnan(val_true)
    val_pred = val_pred[mask]
    val_true = val_true[mask]

    val_pred[val_pred < threshold] = threshold
    val_true[val_true < threshold] = threshold
    
    accuracy = 1.0 - np.average(np.abs(val_pred - val_true) / np.abs(val_true))
    return accuracy


def get_available_model_filename(filename: str,
                                 load_strictly: bool,
                                 ):
    """获取有效的模型文件名

    Args:
        filename (str): 模型文件名, 格式: '20241023-134746-provincial_load-ahead.json',
            即'年月日-时分秒-标签-交易类型.json'
        load_strictly (bool): 如果设为True, 只会读取名为filename的模型文件;
            如果设为False, 优先读取名为filename的模型文件, 如果读不到则读取时间最近的模型文件

    Returns:
        str: 有效的模型文件名
    """
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        return filename
    elif load_strictly:
        raise ValueError("未找到文件: {}".format(filename))
    else:
        postfix = os.path.basename(filename)[23:]
        folder = os.path.dirname(filename)
        file_list = [f for f in os.listdir(folder) if f.endswith(postfix)]
        if not file_list:
            raise ValueError("文件筛选出错, 未找到相似文件: {}".format(filename))
        file_list.sort()
        new_file = file_list[-1]
        new_filepath =  os.path.join(folder, new_file)
        return new_filepath


def get_sun_times(date: datetime.date):
    # 创建 LocationInfo 对象 - 使用江苏南京坐标（江苏省省会）
    # 南京坐标: latitude=32.06, longitude=118.78，江苏全省纬度范围约31-35°N
    location = LocationInfo(name='nanjing', region="Jiangsu, China", timezone="Asia/Shanghai",
                            latitude=32.06, longitude=118.78)
    # 获取当前日期的日出和日落时间
    s = sun(location.observer, date=date, tzinfo=location.timezone)
    sunrise = s['sunrise'].strftime('%H:%M')
    sunset = s['sunset'].strftime('%H:%M')
    return sunrise, sunset


def set_solar_zero(pdf_data:pd.DataFrame, sunrise:str='05:30', sunset:str='19:30', label:str=None):
    """_summary_
    Args:
        pdf_data (pd.DataFrame): 预处理数据
        low_time (int): 低于(不包含)该时刻设置为0
        up_time (int): 高于(不包含)该时刻设置为0
        label (str): 预处理数据标签
    Returns:
        _type_: _description_
    """ 
    pdf_data['date_time'] = pdf_data['date_time'].astype(str)
    pdf_data.set_index((pd.DatetimeIndex(pdf_data['date_time'])), inplace=True)
    pdf_data['hour'] = pdf_data.apply(lambda x: str_timestamp_to_str(x['date_time'], output_format='%H:%M'), axis=1)
    pdf_data[label] = pdf_data.apply(lambda x: 0 if (x['hour'] <= sunrise) else x[label], axis=1)
    pdf_data[label] = pdf_data.apply(lambda x: 0 if (x['hour'] >= sunset) else x[label], axis=1)
    pdf_data.drop(columns=['hour'], inplace=True)
    pdf_data.reset_index(inplace=True, drop=True)
    return pdf_data


def process_average_data_by_versions(pdf_data: pd.DataFrame, start_plus_hour: int, window_hours: int):
    # 获取每个版本对应预测时间
    pdf_version_windows = None
    for version_date, pdf_version_data in pdf_data.groupby('version'):
        pdf_version_data.sort_values(by='date_time', inplace=True)
        version_date_time_str = version_date.strftime('%Y-%m-%d %H:%M:%S')
        start_date_time = str_timestamp_to_str(version_date_time_str, plus_hours=start_plus_hour)
        end_date_time = str_timestamp_to_str(start_date_time, plus_hours=window_hours)

        pdf_version_data_window = \
            pdf_version_data[(pdf_version_data['date_time']>=start_date_time) & (pdf_version_data['date_time']<end_date_time)].copy()
        # 当一个版本中数据不全时不使用
        if pdf_version_data_window.shape[0] < window_hours:
            continue
        
        pdf_version_data_window['version'] = version_date_time_str
        pdf_version_data_window['info_date'] = start_date_time.split(' ')[0]

        if pdf_version_windows is None:
            pdf_version_windows = pdf_version_data_window.copy()
        else:
            pdf_version_windows = pd.concat([pdf_version_windows, pdf_version_data_window], axis=0)
    pdf_version_windows.reset_index(drop=True, inplace=True)

    # 计算每个版本对应分时平均值
    pdf_version_windows['info_time'] = pdf_version_windows['date_time'].str[-8:]
    compute_cols = [col for col in pdf_version_windows.columns if col not in ['date_time', 'info_date', 'info_time', 'version', 'version_diff', 'meteo_source']]
    agg_dict = dict.fromkeys(compute_cols, 'mean')
    pdf_version_mean = pdf_version_windows.groupby(['version', 'info_date', 'info_time', 'meteo_source']).agg(agg_dict)
    pdf_version_mean.reset_index(inplace=True)
    pdf_version_mean['date_time'] = pdf_version_mean['info_date'] + ' ' + pdf_version_mean['info_time']
    pdf_version_mean['version_diff'] = None
    pdf_version_mean = pdf_version_mean.dropna(how='all')

    return pdf_version_mean


def process_average_data_by_date(pdf_data: pd.DataFrame, start_date: str, end_date: str):
    pdf_data.sort_values(by='date_time', inplace=True)
    pdf_data_window = pdf_data[(pdf_data['info_date']>=start_date) & (pdf_data['info_date']<=end_date)].copy()
    compute_cols = [col for col in pdf_data_window.columns if col not in ['date_time', 'info_date', 'info_time', 'version', 'version_diff', 'meteo_source']]
    pdf_data_window['info_date'] = start_date
    pdf_data_window['info_time'] = pdf_data_window['date_time'].str[-8:]
    agg_dict = dict.fromkeys(compute_cols, 'mean')
    pdf_version_mean = pdf_data_window.groupby(['version', 'info_date', 'info_time', 'meteo_source']).agg(agg_dict)
    pdf_version_mean.reset_index(inplace=True)
    pdf_version_mean['info_date'] = start_date
    pdf_version_mean['date_time'] = pdf_version_mean['info_date'] + ' ' + pdf_version_mean['info_time']
    pdf_version_mean['version_diff'] = None

    return pdf_version_mean

from scipy.ndimage import gaussian_filter1d
def smooth_gaussian(data, sigma=4):
    P_MIN = 0.0
    P_MAX = np.max(data) * 1.05  
    smooth_data = gaussian_filter1d(data, sigma=sigma)
    return np.clip(smooth_data, P_MIN, P_MAX)

