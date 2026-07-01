import datetime 
import sys

import chinese_calendar as cc
import pandas as pd

# sys.path.append('../../..')
# from src.priceapi.datasets.databases import ClusterMySQL


def date_to_holiday_tag(date: str) -> int:
    """ 将日期映射到相应的整数标签 
    
    Inputs:
        date (str): 日期字符串, 形式为'YYYY-mm-dd', 例如'2022-01-01'

    Returns:
        tag (int): 日期的整数标签, 含义如下:
            -1  工作日
            0   普通周六日
            1   元旦
            2   春节
            3   清明
            4   劳动节
            5   端午
            6   中秋
            7   国庆节
    """

    y, m, d = [int(x) for x in date.split('-')]
    date = datetime.date(y, m, d)

    if cc.is_holiday(date):
        holiday_name = cc.get_holiday_detail(date)[1]
        
        if holiday_name:
            if holiday_name == cc.Holiday.new_years_day.value:
                tag = 1  # 元旦
            elif holiday_name == cc.Holiday.spring_festival.value:
                tag = 2  # 春节
            elif holiday_name == cc.Holiday.tomb_sweeping_day.value:
                tag = 3  # 清明
            elif holiday_name == cc.Holiday.labour_day.value:
                tag = 4  # 劳动节
            elif holiday_name == cc.Holiday.dragon_boat_festival.value:
                tag = 5  # 端午
            elif holiday_name == cc.Holiday.mid_autumn_festival.value:
                tag = 6  # 中秋
            elif holiday_name == cc.Holiday.national_day.value:
                tag = 7  # 国庆节
            else:
                raise ValueError(f"Holiday '{holiday_name}' has not been assigned a tag yet!")
        else:
            tag = 0  # 普通周六日
    else:  
        tag = -1  # 工作日
    
    return tag




# 设置日期范围
start_date = '2025-01-01'
end_date = '2025-12-31'

# 给日期打Tag
df = pd.date_range(start_date, end_date).strftime("%Y-%m-%d").to_frame(name='info_date').reset_index(drop=True)
df['holiday_tag'] = df['info_date'].map(lambda date: date_to_holiday_tag(date))
