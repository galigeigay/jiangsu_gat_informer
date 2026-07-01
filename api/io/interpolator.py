from copy import deepcopy
from datetime import datetime, timedelta, timezone
from dateutil import tz
import requests
from requests import RequestException
import io
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from suntime import Sun


__all__ = ["interpolate_by_service", "to_parquet_string", "from_parquet_string"]


def is_daylight(coordinate: str, now_time: np.datetime64, offset=0) -> bool:
    """
    判断该点目前时间是否是白天，注意参数date一定为时间时间
    :param coordinate: 经纬度坐标，格式为"lat,lon"
    :param now_time: 当前时间 (UTC世界时)，类型为datetime.datetime 或者 np.datetime64"
    :param offset: 时区偏移量
    """
    lat, lon = coordinate.split(",")
    sun = Sun(float(lat), float(lon))
    # convert np.datetime64 to datetime
    if not isinstance(now_time, datetime):
        now_time = _to_datetime(now_time)
    abd_sr = sun.get_sunrise_time(now_time, time_zone=tz.tzutc())
    abd_ss = sun.get_sunset_time(now_time, time_zone=tz.tzutc())
    # 考虑落日前后30分钟依然有辐照度的存在
    abd_sr += timedelta(hours=8)
    abd_ss += timedelta(hours=8)
    now_time += timedelta(hours=8 - offset)
    return abd_sr.strftime('%H%M%S') <= now_time.strftime('%H%M%S') <= abd_ss.strftime('%H%M%S')


def _to_datetime(date) -> datetime:
    """
    Converts a numpy datetime64 object to a python datetime object
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def to_parquet_string(df: pd.DataFrame) -> str:
    """
    用于HTTP json传递的数据压缩与序列化, 其数据结构采用parquet

    通过pyarrow引擎将DataFrame转换为parquet格式的字节流，
    并返回字节流的字符串表示

    :return: str
    """
    fb = io.BytesIO()
    df.to_parquet(fb, engine='pyarrow')
    fb.seek(0)
    return str(fb.read()).encode().decode('unicode-escape')


def from_parquet_string(parquet_string: str) -> pd.DataFrame:
    """
    用于HTTP json传递的Parquet数据反序列化，将字符串表示的parquet数据转换为DataFrame

    :param parquet_string: str
    :return: pd.DataFrame
    """
    fb = parquet_string.encode('ISO-8859-1')[2:-1]
    return pd.read_parquet(io.BytesIO(fb))


def interpolate_by_service(df: pd.DataFrame, interpolate_column: list[str], meta_column: list[str]):
    """
    对气象数据进行逐小时深度模型逐小时插值, 顶层接口函数

    1. 每次批量请求25个点
    2. 对`interpolate_column`列进行深度模型插值
    3. 对非`interpolate_column`列进行线性插值

    Args:
        df: pd.DataFrame 原始气象数据，目前只支持ECMWF产品系列的原始数据
        interpolate_column: list[str] 需要进行深度模型插值的列

    Returns:
        pd.DataFrame 插值后的逐小时气象数据
    """
    batch_size = 25
    station_codes = df['coordinate'].unique()
    batches = [station_codes[i:i + batch_size] for i in range(0, len(station_codes), batch_size)]
    
    results = pd.DataFrame()

    for batch in batches:
        batch_df = df[df['coordinate'].isin(batch)]
        interpolated_batch = batch_interpolate_by_service(batch_df, interpolate_column, meta_column=meta_column)
        results = pd.concat([results, interpolated_batch], ignore_index=True)

    return results.round(2)


def batch_interpolate_by_service(df: pd.DataFrame, interpolate_column: list[str], meta_column: list[str]):
    def process_group(group):
        return request_interpolate_api(group, interpolate_column=interpolate_column, meta_column=meta_column)

    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = [executor.submit(process_group, group) for _, group in df.groupby("coordinate")]
        results = []
        for future in as_completed(futures):
            results.append(future.result())

    df = pd.concat(results).reset_index(drop=True)

    return df


def request_interpolate_api(group: pd.DataFrame, interpolate_column: list[str], meta_column: list[str]):
    group.sort_values('date_time', inplace=True)
    array_lq = group.loc[:, ["date_time"] + interpolate_column].copy()

    group.drop(columns=interpolate_column, inplace=True)

    array_hq = interpolate(array_lq, service_endpoint='https://meteo-model.aienertech.cn/api/interpolate/weather')

    array_hq["date_time"] = pd.to_datetime(array_hq["date_time"])
    array_hq = ssrd_accumulate_to_instant(array_hq, loc=group.loc[group.index[0], "coordinate"])
    if "shortwave_radiation_instant" in group.columns:
        group.drop(columns="shortwave_radiation_instant", inplace=True)

    # 辐照与降雨强度
    array_hq["date_time"] = array_hq["date_time"].astype(str)
    final_df = pd.merge(array_hq, group, on="date_time", how="outer")

    fill_columns = deepcopy(meta_column)
    fill_columns.remove("date_time")
    for fill_column in fill_columns:
        final_df[fill_column] = final_df[fill_column].bfill()


    # 处理没有模型插值的列，采用线性插值方法
    non_interpolate_column = get_non_interpolate_column(final_df, interpolate_column, meta_column=meta_column)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        final_df[non_interpolate_column] = (final_df[non_interpolate_column].
                                            infer_objects(copy=False).interpolate(method="linear"))
    final_df = precipitation_accumulate_to_instant(final_df)

    return final_df


def ssrd_accumulate_to_instant(df: pd.DataFrame,
                               loc: str) -> pd.DataFrame:
    ssrd = "shortwave_radiation"
    ssrd_instant = "shortwave_radiation_instant"

    if ssrd not in df.columns:
        return df

    ssrd_serial = df[ssrd].diff() / 3600
    ssrd_serial.fillna(df[ssrd] / 3600, inplace=True)

    df[ssrd_instant] = ssrd_serial

    # 对于date_time 处理21:00 - 06:00之间的辐照度为0
    df[ssrd_instant] = df.apply(
        lambda x: 0 if not is_daylight(loc, x['date_time'], offset=8) else max(0, x[ssrd_instant]), axis=1)

    # 最后一个值
    _tmp = df[ssrd_instant].iloc[-2] - (df[ssrd_instant].iloc[-3]) + df[ssrd_instant].iloc[-2]
    df.loc[len(df) - 1, ssrd_instant] = _tmp

    return df


factors = {
    "shortwave_radiation": 3600,
    "total_precipitation": 1,
    "snow_fall": 1
}


def acc_2_inst(group: pd.DataFrame, column_names: list[str]):
    group.reset_index(drop=True, inplace=True)
    
    for column_name in column_names:
        inst_column = f"{column_name}_inst"
        group[inst_column] = group[column_name].copy().astype('float64')
        group.loc[0, inst_column] = group.loc[0, column_name] / factors[column_name]
        group.loc[1:, inst_column] = (group[column_name] - group[inst_column].shift(1))[1:] / factors[column_name]
    
    return group


def precipitation_accumulate_to_instant(df: pd.DataFrame) -> pd.DataFrame:
    tp = "total_precipitation"
    tp_instant = "precipitation_instant"

    if tp not in df.columns:
        return df

    tp_serial = df[tp].diff()
    tp_serial.fillna(df[tp], inplace=True)

    df[tp_instant] = tp_serial

    # 最后一个值
    _tmp = df[tp_instant].iloc[-2] - (df[tp_instant].iloc[-3]) + df[tp_instant].iloc[-2]
    df.loc[len(df) - 1, tp_instant] = _tmp

    # 降雨量不能为负数
    df[tp_instant] = df[tp_instant].clip(lower=0)

    return df


def get_non_interpolate_column(df: pd.DataFrame,
                               interpolate_column: list,
                               meta_column: list[str]) -> list:
    """
    获取插值的列
    """
    interp_set = set(interpolate_column)
    
    meta_set = set(meta_column)

    non_interp_set = set(df.columns) - interp_set - meta_set

    return list(non_interp_set)


def interpolate(time_serial_df: pd.DataFrame,
                service_endpoint: str = 'https://meteo-model.aienertech.cn/api/interpolate/weather'):
    """
    调用气象模型的时序插值微服务
    :param time_serial_df: pd.DataFrame
    :param service_endpoint: str 插值服务后端地址
    :return: pd.DataFrame
    """
    try:
        df_str = to_parquet_string(time_serial_df)
    except Exception as e:
        raise ValueError(f'failed to serialize DataFrame: {e}')

    payload = {
        "forecast_lq": df_str
    }

    resp = requests.post(service_endpoint, json=payload)

    resp_json = resp.json()

    if resp.status_code != 200 and "data" not in resp_json:
        raise RequestException(f'failed to interpolate via micro service: {resp.text}')

    if resp_json['code'] is None or resp_json['code'] != 200:
        raise RequestException(f'failed to interpolate via micro service: {resp_json["msg"]}')

    df_hq_str = resp.json()['data']

    try:
        df_hq = from_parquet_string(df_hq_str)
    except Exception as e:
        raise ValueError(f'failed to deserialize DataFrame: {e}')

    return df_hq
