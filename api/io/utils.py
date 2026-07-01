import calendar
from datetime import datetime, timedelta
from typing import List

import pandas as pd


def str_timestamp_to_str(time_stamp, input_format='%Y-%m-%d %H:%M:%S', output_format='%Y-%m-%d %H:%M:%S', plus_hours=0, plus_minutes=0):
    """对时间字符串进行转换

    Args:
        time_stamp (_type_): 输入的时间字符串
        input_format (str, optional): 输入时间的格式, 比如: '%Y-%m-%d %H:%M:%S'.
        output_format (str, optional): 输出时间的格式, 比如: '%Y-%m-%d %H:%M:%S'.
        plus_hours (int, optional): 在转换时增加的小时数
        plus_minutes (int, optional): 在转换时增加的分钟数

    Returns:
        str: 转换后的时间字符串
    """
    if '+' in time_stamp:
        offset_hours = 0
        time_stamp = time_stamp.split('+')[0]
    else:
        offset_hours = 0
    rst = (datetime.strptime(time_stamp, input_format) + timedelta(hours=plus_hours) + timedelta(hours=offset_hours) + timedelta(minutes=plus_minutes)).strftime(output_format)
    return rst


def parse_datetime_with_24(dt_str, input_format='%Y-%m-%d %H:%M:%S', output_format='%Y-%m-%d %H:%M:%S', offset_minutes=-15):
    """
    将格式为'2023-05-01 24:00'字符串转为含义为'2023-05-02 00:00'的字符串

    当offset_minutes=-15时, 作用是将'2023-05-01 24:00'转换为'2023-05-01 23:45'
    """
    if ' 24:' in dt_str:
        dt_str = dt_str.replace(' 24:', ' 23:')
        plus_hours = 1
    else:
        plus_hours = 0
    rst = str_timestamp_to_str(dt_str, input_format, output_format, plus_hours=plus_hours, plus_minutes=offset_minutes)
    return rst


def format_float(val, float_point=2):
    """
    格式化浮点数据, 小数点后保留float_point位
    """
    assert float_point in (2,3,4)
    if float_point == 2:
        rst = float("{:.2f}".format(val)) if not pd.isna(val) else None
    elif float_point == 3:
        rst = float("{:.3f}".format(val)) if not pd.isna(val) else None
    elif float_point == 4:
        rst = float("{:.4f}".format(val)) if not pd.isna(val) else None
    else:
        rst = val
    return rst


def calculate_datetime_version_diff(date_time, version, treat_future_as_d1=False):
    """
    计算version相对于date_time是'D-?'版本
    """
    dt = str_timestamp_to_str(date_time, input_format='%Y-%m-%d %H:%M:%S', output_format='%Y-%m-%d')
    cur_dt = datetime.now().strftime("%Y-%m-%d")
    next_dt = str_timestamp_to_str(cur_dt, input_format='%Y-%m-%d', output_format='%Y-%m-%d', plus_hours=24)

    if treat_future_as_d1 and dt > next_dt:
        diff = datetime.strptime(next_dt, "%Y-%m-%d") - datetime.strptime(version, "%Y-%m-%d")
    else:
        diff = datetime.strptime(dt, "%Y-%m-%d") - datetime.strptime(version, "%Y-%m-%d")
    day_diff = diff.days

    return day_diff


def convert_pdf_to_96p(pdf_data: pd.DataFrame, 
                       interp_cols: List[str]=[], 
                       ffill_list: List[str]=[], 
                       version_count: int=0, 
                       add_last: bool=True):
    """
    将Pandas DataFrame从24点转换为96点, 并用二次函数插值interp_cols所给的列
    """
    if version_count == 0:
        pdf_data['date_time'] = pd.to_datetime(pdf_data['date_time'])
        pdf_data = pdf_data.sort_values(by='date_time')

        if add_last:
            last_index = len(pdf_data)
            pdf_data.loc[last_index] = pdf_data.iloc[-1]
            pdf_data.loc[last_index, 'date_time'] += pd.Timedelta(hours=1)

        pdf_data.set_index("date_time", inplace=True)
        pdf_data = pdf_data.resample("15min").asfreq()
        for col in interp_cols:
            pdf_data[col] = pdf_data[col].interpolate(method='linear').ffill()
        if ffill_list:
            pdf_data[ffill_list] = pdf_data[ffill_list].ffill()
        pdf_data = pdf_data.reset_index()
        pdf_data['date_time'] = pdf_data['date_time'].astype(str)
        return pdf_data
    else:
        pdf_data['date_time'] = pd.to_datetime(pdf_data['date_time'])
        pdf_all = pd.DataFrame()
        for num in range(version_count+1):
            pdf_one_version = pdf_data[pdf_data['version_diff']==num].copy()
            pdf_one_version = pdf_one_version.sort_values(by='date_time')
            if add_last:
                last_index = len(pdf_data)
                pdf_one_version.loc[last_index] = pdf_one_version.iloc[-1]
                pdf_one_version.loc[last_index, 'date_time'] += pd.Timedelta(hours=1)
            pdf_one_version.set_index("date_time", inplace=True)
            pdf_one_version = pdf_one_version.resample("15min").asfreq()
            for col in interp_cols:
                pdf_one_version[col] = pdf_one_version[col].interpolate(method='linear').ffill()
            if ffill_list:
                pdf_one_version[ffill_list] = pdf_one_version[ffill_list].ffill()
            pdf_one_version = pdf_one_version.reset_index()
            if pdf_all.empty:
                pdf_all = pdf_one_version.copy()
            else:
                pdf_all = pd.concat([pdf_all, pdf_one_version.copy()], axis=0)
        pdf_all['date_time'] = pdf_all['date_time'].astype(str)
        return pdf_all



def get_next_ten_days_period(set_date: str=None):
    if set_date is not None:
        set_date = datetime.strptime(set_date, "%Y-%m-%d")
    else:
        set_date = datetime.today()
    year = set_date.year
    month = set_date.month
    day = set_date.day

    if day < 10:
        start_day = 11
        end_day = 20
    elif day < 20:
        start_day = 21
        next_month = datetime(year, month%12+1, 1)
        end_day = (next_month - timedelta(days=1)).day
    else:
        start_day =1
        end_day = 10 
        month = month % 12 + 1
        year = year + 1 if month == 1 else year
    start_date = datetime(year, month, start_day).strftime("%Y-%m-%d")
    end_date = datetime(year, month, end_day).strftime("%Y-%m-%d")
    
    return start_date, end_date


def get_next_month_period(set_date: str):
    if set_date is not None:
        set_date = datetime.strptime(set_date, "%Y-%m-%d")
    else:
        set_date = datetime.today()
    year = set_date.year
    month = set_date.month
    if month == 12:
        month = 1
        year = year + 1
    else:
        month = month + 1
    start_date = datetime(year, month, 1).strftime("%Y-%m-%d")
    _, last_day = calendar.monthrange(year, month)
    end_date = datetime(year, month, last_day).strftime("%Y-%m-%d")

    return start_date, end_date


def get_next_span_days(task_name: str, set_date: str=None):
    if task_name == 'ten_days':
        start_date, end_date = get_next_ten_days_period(set_date)
    else:
        start_date, end_date = get_next_month_period(set_date)
    return start_date, end_date
