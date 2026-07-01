from datetime import datetime
import time
import numpy as np
import pandas as pd
import os
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Sequence,
)

import sys
script_path = os.path.dirname(os.path.abspath(__file__))
script_path = script_path.split('/api')[0]
sys.path.append(f'{script_path}/api')


import anydb.doris as cusdoris
# from .. anydb import doris as cusdoris
from ..io.mongo import MongoInfo, read_mongo, insert_mongo
from ..io.mysql import MysqlInfo, read_mysql, upload_mysql, read_mysql2
from ..io.utils import (
    convert_pdf_to_96p,
    format_float,
    parse_datetime_with_24,
    str_timestamp_to_str,
    get_next_span_days
)
from .baseclass import BaseData
from .constants import (
    METEO_DORIS_CONST,
    MONGO_CONST,
    DORIS_SERVER_CONST,
    MONGO_SERVE_CONST,
    WEATHER_FEATURE_LIST,
    WEATHER_FEATURE_SEA_LIST,
    JIANGSU_SEA,
    JIANGSU_CITY,
    JIANGSU_CITY2_WIND,
    JIANGSU_CITY2_SOLAR,
    JIANGSU_SOLAR_CAPACITY,
    JIANGSU_WIND_CAPACITY,
    JIANGSU_CITY_WIND_CAPACITY,
    JIANGSU_CITY_SOLAR_CAPACITY,
    METEO_SOURCE,
    MONTH_PLUS_HOUR,
    TEN_DAYS_PLUS_HOUR,
    METEO_TEST_DORIS_CONST,
    METEO_SEA_SOURCE,
    SEA_WIND_CONST,
)
from .utils import calculate_time_idx, process_average_data_by_versions, process_average_data_by_date
from ..io.logger import setup_logger
from api.io.interpolator import interpolate_by_service

from .mark_holiday import date_to_holiday_tag
logger = setup_logger(__name__)


class PdData(BaseData):
    def __init__(self,
                 data_source: MongoInfo,
                 query_conditions: Mapping[str, Any],
                 query_fields: Mapping[str, Any],
                 process_func: Callable = None,
                 process_func_args: Mapping[str, Any] = {},
                 class_name='PdData',
                 ):
        """Pandas DataFrame数据类, 包含数据库读取和后处理流程

        Args:
            data_source (MongoInfo): Mongo数据源
            query_conditions (Mapping[str, Any]): Mongo查询条件
            query_fields (Mapping[str, Any]): Mongo获取数据字段, 若为空则获取全部字段
            process_func (Callable, optional): 数据后处理函数
            process_func_args (Mapping[str, Any], optional): 数据后处理函数的传入参数
            class_name (str, optional): 类名称, 方便打印debug信息
        """
        super().__init__(class_name=class_name)
        if isinstance(data_source, MongoInfo):
            self._data = read_mongo(
                mongo_info=data_source,
                conditions=query_conditions,
                fields=query_fields,
            )
        else:
            raise TypeError('传入参数data_source类型不支持: {}'.format(type(data_source)))

        if process_func is not None:
            if process_func_args:
                self._data = process_func(data=self._data, **process_func_args)
            else:
                self._data = process_func(data=self._data)

    @property
    def data(self) -> pd.DataFrame:
        return self._data


class BaseConditionData(PdData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 mongo_collection: str,
                 column_name: str,
                 trade_type: str,
                 class_name='BaseConditionData',
                 ):
        """边界条件数据基类, 基于PdData基类

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
            mongo_collection (str): Mongo数据表名
            column_name (str): 边界条件列名称, 支持: 'provincial_load', 'tie_line',
                'new_energy_wind', 'new_energy_solar', 'hydropower', 'nuclear', 'local_coal',
                'non_market', 'exe_hydro', 'total_output', 'clear_price'
            trade_type (str): 交易类型, 支持: 'ahead', 'realtime'
            class_name (str, optional): 类名称, 方便打印debug信息
        """
        super().__init__(
            data_source=MongoInfo(
                mongo_url=MONGO_CONST.mongo_url,
                mongo_user=MONGO_CONST.mongo_user,
                mongo_password=MONGO_CONST.mongo_password,
                mongo_auth_db=MONGO_CONST.mongo_auth_db,
                mongo_db=MONGO_CONST.mongo_db,
                mongo_collection=mongo_collection,
            ),
            query_conditions={
                'info_date': {'$gte': start_date, '$lte': end_date},
                'area': '汇总',
            },
            query_fields={},
            process_func=process_trade_values,
            process_func_args={
                'new_name': column_name,
            },
            class_name=class_name,
        )


class BaseRevPowerData(PdData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 mongo_collection: str,
                 column_name: str,
                 rev_flag: bool = True,
                 class_name='BaseRevPowerData',
                 ):
        """边界条件数据基类, 基于PdData基类

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
            mongo_collection (str): Mongo数据表名
            column_name (str): 边界条件列名称
            rev_flag (bool): 是否使用装机容量进行修正
            class_name (str, optional): 类名称, 方便打印debug信息
        """

        super().__init__(
            data_source=MongoInfo(
                mongo_url=MONGO_CONST.mongo_url,
                mongo_user=MONGO_CONST.mongo_user,
                mongo_password=MONGO_CONST.mongo_password,
                mongo_auth_db=MONGO_CONST.mongo_auth_db,
                mongo_db=MONGO_CONST.mongo_db,
                mongo_collection=mongo_collection,
            ),
            # query_conditions=query_conditions,
            query_conditions={
                'info_date': {'$gte': start_date, '$lte': end_date},
                'area': '汇总',
            },
            query_fields={},
            process_func=process_trade_values,
            process_func_args={
                'new_name': column_name,
            },
            class_name=class_name,
        )

        if rev_flag:
            self._revise_with_capacity(column_name)
        self._data[column_name] = self._data.apply(lambda x: 0 if x[column_name] < 0 else x[column_name], axis=1)

    def _revise_with_capacity(self, column_name: str):
        """
        根据装机容量修正出力数据
        """
        org_pdf = self._data.copy()
        org_pdf["date_time"] = pd.to_datetime(org_pdf["date_time"])

        if "wind" in column_name:
            capacity = JIANGSU_WIND_CAPACITY
            # capacity = MENGXI_WIND_CAPACITY
        elif "solar" in column_name:
            capacity = JIANGSU_SOLAR_CAPACITY
            # capacity = MENGXI_SOLAR_CAPACITY

        def get_capacity(row):
            year = row["date_time"].year
            month = row["date_time"].month
            return capacity.get(year, {}).get(month, 0)

        org_pdf[f"{column_name}_capacity"] = org_pdf.apply(get_capacity, axis=1)

        std_value = capacity[max(capacity)]
        std_value = std_value[max(std_value)]

        org_pdf.loc[org_pdf[f"{column_name}_capacity"] != 0, column_name] = round(
            std_value / org_pdf[f"{column_name}_capacity"] * org_pdf[column_name], 4
        )
        org_pdf = org_pdf.drop(f"{column_name}_capacity", axis=1)
        org_pdf["date_time"] = org_pdf["date_time"].astype(str)

        self._data = org_pdf


def process_trade_values(data: pd.DataFrame,
                         new_name: str,
                         ):
    """边界条件数据后处理

    Args:
        data (pd.DataFrame): 原始数据DataFrame
        new_name (str): 边界条件列名称, 支持: 'provincial_load', 'tie_line',
            'new_energy_wind', 'new_energy_solar', 'hydropower', 'nuclear', 'local_coal',
            'non_market', 'exe_hydro', 'total_output', 'clear_price'
        trade_type (str): 交易类型, 支持: 'ahead', 'realtime'

    Returns:
        pd.DataFrame: 处理后的数据DataFrame
    """

    pdf_data = data[['info_date', 'info_time', 'value', 'update_time']].copy()

    pdf_data.rename(columns={'value': new_name}, inplace=True)
    pdf_data[new_name] = pdf_data.apply(lambda x: format_float(float(str(x[new_name])), float_point=2), axis=1)

    info_time = pdf_data['info_time'].astype(str)
    is_24 = info_time.isin(['24:00'])
    pdf_data['date_time'] = pdf_data['info_date'] + ' ' + info_time.str.replace('24:00', '00:00')
    # pdf_data.loc[~pdf_data['date_time'].str.endswith(':00'), 'date_time'] += ':00'
    pdf_data['date_time'] = pd.to_datetime(pdf_data['date_time'])
    pdf_data.loc[is_24, 'date_time'] = pdf_data.loc[is_24, 'date_time'] + pd.Timedelta(days=1)
    pdf_data['date_time'] = pdf_data['date_time'].dt.strftime('%Y-%m-%d %H:%M')

    pdf_data.sort_values(['date_time', 'update_time'], ascending=True, inplace=True)
    pdf_data.drop_duplicates(['date_time'], keep='last', inplace=True)
    pdf_data.drop(['update_time'], axis=1, inplace=True)

    return pdf_data


class ProvincialLoadAheadData(BaseConditionData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 ):
        """日前省调负荷数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
        """
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            mongo_collection=MONGO_CONST.mongo_collection_provincial_load_ahead,
            column_name='provincial_load_ahead',
            trade_type='ahead',
            class_name='ProvincialLoadAheadData',
        )


class TieLineAheadData(BaseConditionData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 ):
        """日前联络线数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
        """
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            mongo_collection=MONGO_CONST.mongo_collection_tie_line_ahead,
            column_name='tie_line_ahead',
            trade_type='ahead',
            class_name='TieLineAheadData',
        )


class NewEnergyTotalAheadData(BaseRevPowerData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 rev_flag: bool = True
                 ):
        """日前新能源总出力数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
        """
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            mongo_collection=MONGO_CONST.mongo_collection_new_energy_ahead,
            column_name='new_energy_total_ahead',
            rev_flag=rev_flag,
            class_name='NewEnergyTotalAheadData',
        )


class NewEnergyWindAheadData(BaseRevPowerData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 rev_flag: bool = True
                 ):
        """日前新能源-风出力数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
        """
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            mongo_collection=MONGO_CONST.mongo_collection_new_energy_wind_ahead,
            column_name='new_energy_wind_ahead',
            rev_flag=rev_flag,
            class_name='NewEnergyWindAheadData',
        )


class NewEnergySolarAheadData(BaseRevPowerData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 rev_flag: bool = True
                 ):
        """日前新能源-光出力数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
        """
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            mongo_collection=MONGO_CONST.mongo_collection_new_energy_solar_ahead,
            column_name='new_energy_solar_ahead',
            rev_flag=rev_flag,
            class_name='NewEnergySolarAheadData',
        )


class NonMarketAheadData(BaseConditionData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 ):
        """日前非市场化出力数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
        """
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            mongo_collection=MONGO_CONST.mongo_collection_non_market_ahead,
            column_name='non_market_ahead',
            trade_type='ahead',
            class_name='NonMarketAheadData',
        )


class AheadData(BaseData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 column_list: Sequence[str],
                 rev_flag: bool = True
                 ):
        """日前数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
            column_list (Sequence[str]): 边界条件列名称, 支持: 'provincial_load', 'tie_line',
                'new_energy_wind', 'new_energy_solar', 'hydropower', 'nuclear', 'local_coal',
                'non_market', 'exe_hydro', 'total_output', 'clear_price'
        """
        super().__init__(class_name='AheadData')
        self._data : pd.DataFrame = None
        self.load_data(start_date=start_date, end_date=end_date, column_list=column_list, rev_flag=rev_flag)

    def load_data(self,
                  start_date: str,
                  end_date: str,
                  column_list: Sequence[str],
                  rev_flag: bool = True
                  ):
        pdf_data_ahead = None

        if 'provincial_load_ahead' in column_list:
            data_provincial_load_ahead = ProvincialLoadAheadData(start_date=start_date, end_date=end_date)
            if pdf_data_ahead is None:
                pdf_data_ahead = data_provincial_load_ahead.data.drop(columns=['info_date', 'info_time'])
            else:
                pdf_data_ahead = pdf_data_ahead.merge(data_provincial_load_ahead.data.drop(columns=['info_date', 'info_time']), on='date_time', how='outer')
            # print(pdf_data_ahead.head())

        if 'tie_line_ahead' in column_list:
            data_tie_line_ahead = TieLineAheadData(start_date=start_date, end_date=end_date)
            if pdf_data_ahead is None:
                pdf_data_ahead = data_tie_line_ahead.data.drop(columns=['info_date', 'info_time'])
            else:
                pdf_data_ahead = pdf_data_ahead.merge(data_tie_line_ahead.data.drop(columns=['info_date', 'info_time']), on='date_time', how='outer')
            # print(pdf_data_ahead.head())

        if 'new_energy_total_ahead' in column_list:
            data_new_energy_ahead = NewEnergyTotalAheadData(start_date=start_date, end_date=end_date, rev_flag=rev_flag)
            if pdf_data_ahead is None:
                pdf_data_ahead = data_new_energy_ahead.data.drop(columns=['info_date', 'info_time'])
            else:
                pdf_data_ahead = pdf_data_ahead.merge(data_new_energy_ahead.data.drop(columns=['info_date', 'info_time']), on='date_time', how='outer')
            # print(pdf_data_ahead.head())

        if 'new_energy_wind_ahead' in column_list:
            data_new_energy_wind_ahead = NewEnergyWindAheadData(start_date=start_date, end_date=end_date, rev_flag=rev_flag)
            if pdf_data_ahead is None:
                pdf_data_ahead = data_new_energy_wind_ahead.data.drop(columns=['info_date', 'info_time'])
            else:
                pdf_data_ahead = pdf_data_ahead.merge(data_new_energy_wind_ahead.data.drop(columns=['info_date', 'info_time']), on='date_time', how='outer')
            # print(pdf_data_ahead.head())

        if 'new_energy_solar_ahead' in column_list:
            data_new_energy_solar_ahead = NewEnergySolarAheadData(start_date=start_date, end_date=end_date, rev_flag=rev_flag)
            if pdf_data_ahead is None:
                pdf_data_ahead = data_new_energy_solar_ahead.data.drop(columns=['info_date', 'info_time'])
            else:
                pdf_data_ahead = pdf_data_ahead.merge(data_new_energy_solar_ahead.data.drop(columns=['info_date', 'info_time']), on='date_time', how='outer')
            # print(pdf_data_ahead.head())

        if 'non_market_ahead' in column_list:
            data_non_market_ahead = NonMarketAheadData(start_date=start_date, end_date=end_date)
            if pdf_data_ahead is None:
                pdf_data_ahead = data_non_market_ahead.data.drop(columns=['info_date', 'info_time'])
            else:
                pdf_data_ahead = pdf_data_ahead.merge(data_non_market_ahead.data.drop(columns=['info_date', 'info_time']), on='date_time', how='outer')
            # print(pdf_data_ahead.head())

        self._data = pdf_data_ahead

    @property
    def data(self) -> pd.DataFrame:
        return self._data


class ProvincialLoadRealtimeData(BaseConditionData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 ):
        """实时省调负荷数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
        """
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            mongo_collection=MONGO_CONST.mongo_collection_provincial_load_realtime,
            column_name='provincial_load',
            trade_type='realtime',
            class_name='ProvincialLoadRealtimeData',
        )


class TieLineRealtimeData(BaseConditionData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 ):
        """实时联络线数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
        """
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            mongo_collection=MONGO_CONST.mongo_collection_tie_line_realtime,
            column_name='tie_line',
            trade_type='realtime',
            class_name='TieLineRealtimeData',
        )


class NewEnergyTotalRealtimeData(BaseRevPowerData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 rev_flag: bool = True
                 ):
        """实时新能源总出力数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
        """
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            mongo_collection=MONGO_CONST.mongo_collection_new_energy_realtime,
            column_name='new_energy_total',
            rev_flag=rev_flag,
            class_name='NewEnergyTotalRealtimeData',
        )


class NewEnergyWindRealtimeData(BaseRevPowerData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 rev_flag: bool = True
                 ):
        """实时新能源-风出力数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
        """
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            mongo_collection=MONGO_CONST.mongo_collection_new_energy_wind_realtime,
            column_name='new_energy_wind',
            rev_flag=rev_flag,
            class_name='NewEnergyWindRealtimeData',
        )


class NewEnergySolarRealtimeData(BaseRevPowerData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 rev_flag: bool = True
                 ):
        """实时新能源-光出力数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
        """
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            mongo_collection=MONGO_CONST.mongo_collection_new_energy_solar_realtime,
            column_name='new_energy_solar',
            rev_flag=rev_flag,
            class_name='NewEnergySolarRealtimeData',
        )


class NonMarketRealtimeData(BaseConditionData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 ):
        """实时非市场化出力数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
        """
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            mongo_collection=MONGO_CONST.mongo_collection_non_market_realtime,
            column_name='non_market',
            trade_type='realtime',
            class_name='NonMarketRealtimeData',
        )


class RealtimeData(BaseData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 column_list: Sequence[str],
                 rev_flag: bool = True
                 ):
        """实时数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
            column_list (Sequence[str]): 边界条件列名称, 支持: 'provincial_load', 'tie_line',
                'new_energy_wind', 'new_energy_solar', 'hydropower', 'local_coal',
                'non_market', 'total_output', 'clear_price'
        """
        super().__init__(class_name='RealtimeData')
        self._data : pd.DataFrame = None
        self.load_data(start_date=start_date, end_date=end_date, column_list=column_list, rev_flag=rev_flag)

    def load_data(self,
                  start_date: str,
                  end_date: str,
                  column_list: Sequence[str],
                  rev_flag: bool = True
                  ):
        pdf_data_realtime = None

        if 'provincial_load' in column_list:
            data_provincial_load_realtime = ProvincialLoadRealtimeData(start_date=start_date, end_date=end_date)
            if pdf_data_realtime is None:
                pdf_data_realtime = data_provincial_load_realtime.data.drop(columns=['info_date', 'info_time'])
            else:
                pdf_data_realtime = pdf_data_realtime.merge(data_provincial_load_realtime.data.drop(columns=['info_date', 'info_time']), on='date_time', how='outer')
            # print(pdf_data_realtime.head())

        if 'tie_line' in column_list:
            data_tie_line_realtime = TieLineRealtimeData(start_date=start_date, end_date=end_date)
            if pdf_data_realtime is None:
                pdf_data_realtime = data_tie_line_realtime.data.drop(columns=['info_date', 'info_time'])
            else:
                pdf_data_realtime = pdf_data_realtime.merge(data_tie_line_realtime.data.drop(columns=['info_date', 'info_time']), on='date_time', how='outer')
            # print(pdf_data_realtime.head())

        if 'new_energy_total' in column_list:
            data_new_energy_realtime = NewEnergyTotalRealtimeData(start_date=start_date, end_date=end_date, rev_flag=rev_flag)
            if pdf_data_realtime is None:
                pdf_data_realtime = data_new_energy_realtime.data.drop(columns=['info_date', 'info_time'])
            else:
                pdf_data_realtime = pdf_data_realtime.merge(data_new_energy_realtime.data.drop(columns=['info_date', 'info_time']), on='date_time', how='outer')
            # print(pdf_data_realtime.head())
        
        if 'new_energy_wind' in column_list:
            data_new_energy_wind_realtime = NewEnergyWindRealtimeData(start_date=start_date, end_date=end_date, rev_flag=rev_flag)
            if pdf_data_realtime is None:
                pdf_data_realtime = data_new_energy_wind_realtime.data.drop(columns=['info_date', 'info_time'])
            else:
                pdf_data_realtime = pdf_data_realtime.merge(data_new_energy_wind_realtime.data.drop(columns=['info_date', 'info_time']), on='date_time', how='outer')
            # print(pdf_data_realtime.head())

        if 'new_energy_solar' in column_list:
            data_new_energy_solar_realtime = NewEnergySolarRealtimeData(start_date=start_date, end_date=end_date, rev_flag=rev_flag)
            if pdf_data_realtime is None:
                pdf_data_realtime = data_new_energy_solar_realtime.data.drop(columns=['info_date', 'info_time'])
            else:
                pdf_data_realtime = pdf_data_realtime.merge(data_new_energy_solar_realtime.data.drop(columns=['info_date', 'info_time']), on='date_time', how='outer')
            # print(pdf_data_realtime.head())

        if 'non_market' in column_list:
            data_non_market_realtime = NonMarketRealtimeData(start_date=start_date, end_date=end_date)
            if pdf_data_realtime is None:
                pdf_data_realtime = data_non_market_realtime.data.drop(columns=['info_date', 'info_time'])
            else:
                pdf_data_realtime = pdf_data_realtime.merge(data_non_market_realtime.data.drop(columns=['info_date', 'info_time']), on='date_time', how='outer')
            # print(pdf_data_realtime.head())

        self._data = pdf_data_realtime

    @property
    def data(self) -> pd.DataFrame:
        return self._data


class MeteoDorisData(BaseData):
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 meteo_scope: str = 'station',
                 version_num: int = 0,
                 train_flag: bool = False,
                 data_source: str = 'ENS_EXTEND',
                 task_name: str = 'by_day',
                 version_start_date: str = None,
                 version_end_date: str = None,
                 meteo_weighted: bool = False,
                 use_interpolation: bool = False
                 ):
        """气象数据

        Args:
            start_date (str): 样本开始时间, 格式: '2024-10-01'
            end_date (str): 样本结束时间, 格式: '2024-10-31'
        """
        super().__init__(class_name='MeteoDorisData')
        self._data : pd.DataFrame = None
        self._train_flag = train_flag
        self._data_source = data_source
        self._task_name = task_name
        self._org_weather_feat = ['temperature_2m',
                                  'apparent_temperature',
                                  'humidity',
                                  'precipitation_instant',
                                  'cloudcover',
                                  'windspeed_10m',
                                  'windspeed_100m',
                                  'windspeed_200m',
                                  'shortwave_radiation_instant']
        self.meteo_scope = meteo_scope
        self.meteo_weighted = meteo_weighted # 当前只有meteo_scope='city'支持加权气象参数
        self.version_num = version_num
        self.use_interpolation = use_interpolation
        self.get_meteo_data(start_date, end_date, version_start_date, version_end_date)

    def get_meteo_data(self, 
                       start_date: str,
                       end_date: str,
                       version_start_date: str,
                       version_end_date: str):
        meteo_scopes = self.meteo_scope.split('_')
        meteo_data = pd.DataFrame()
        for meteo_scope in meteo_scopes:
            load_func = self._get_load_func(meteo_scope)
            if version_start_date and version_end_date:
                load_func(start_date=version_start_date, end_date=version_end_date)
            else:
                load_func(start_date=start_date, end_date=end_date)
            if meteo_data.empty:
                meteo_data = self._data.copy()
            else:
                meteo_data = pd.merge(meteo_data, self._data.copy(), on=['date_time', 'version', 'version_diff', 'meteo_source'], how='inner')
        self._data = meteo_data

    def _get_load_func(self, meteo_scope: str):
        if self.meteo_weighted:
            assert meteo_scope in ['city', 'province']
            if meteo_scope == 'province':
                return self.load_weighted_province_data
            elif meteo_scope == 'city':
                return self.load_weighted_city_data
        else:
            if meteo_scope == 'province':
                return self.load_province_data
            elif meteo_scope == 'city':
                return self.load_city_data
            elif meteo_scope == 'station':
                return self.load_station_data
            
        return self.load_station_data

    def _get_mysql_query(self,
                         meteo_scope: str,
                         start_date: str,
                         end_date: str,
                         table_name: str,
                         table_name_sea: str,
                         ):
        # 保证在逐日训练时使用单版本，预测时可以设置最新版本（0）或版本号区间；分时训练时按版本时间获取数据，预测时使用最新版本（0）或者按版本时间区间
        # time_span inclue date_time, version, version_diff
        if self.version_num == 0:
            row_number_command = 'ROW_NUMBER() OVER ( PARTITION BY date_time, station_code ORDER BY version DESC, update_time DESC ) AS rank_id'
            time_span = f"date_time BETWEEN '{start_date} 00:00:00' AND '{end_date} 23:59:59'"
        elif self._train_flag:
            if self._task_name == "by_day":
                row_number_command = 'ROW_NUMBER() OVER ( PARTITION BY date_time, station_code ORDER BY version_diff DESC, update_time DESC ) AS rank_id'
                time_span = f"date_time BETWEEN '{start_date} 00:00:00' AND '{end_date} 23:59:59'\nAND version_diff = {self.version_num}"
            else:
                row_number_command = 'ROW_NUMBER() OVER ( PARTITION BY date_time, station_code, version ORDER BY update_time DESC ) AS rank_id'
                time_span = f"version BETWEEN '{start_date} 00:00:00' AND '{end_date} 23:59:59'\nAND version_time = '08:00:00'"
                if self._task_name == "ten_days":
                    time_span += "\nAND version_diff BETWEEN 5 AND 20"
        else:
            if self._task_name == "by_day":
                row_number_command = 'ROW_NUMBER() OVER ( PARTITION BY date_time, station_code, version_diff ORDER BY update_time DESC ) AS rank_id'
                time_span = f"date_time BETWEEN '{start_date} 00:00:00' AND '{end_date} 23:59:59'\nAND version_diff BETWEEN 0 AND {self.version_num}"
            else:
                row_number_command = 'ROW_NUMBER() OVER ( PARTITION BY date_time, station_code, version ORDER BY update_time DESC ) AS rank_id'
                time_span = f"version BETWEEN '{start_date} 00:00:00' AND '{end_date} 23:59:59'\nAND version_time = '08:00:00'"
                if self._task_name == "ten_days":
                    time_span += "\nAND version_diff BETWEEN 5 AND 20"

        if meteo_scope == 'province':
            if self.meteo_weighted:
                if self.version_num == 0:
                    row_number_command = 'ROW_NUMBER() OVER ( PARTITION BY date_time, scope_name ORDER BY version DESC, update_time DESC ) AS rank_id'
                else:
                    row_number_command = 'ROW_NUMBER() OVER ( PARTITION BY date_time, scope_name, version ORDER BY update_time DESC ) AS rank_id'
                query_msg = f"""
                SELECT
                    date_time,
                    version,
                    version_diff,
                    province,
                    coordinate,
                    temperature_2m,
                    apparent_temperature,
                    humidity,
                    total_precipitation,
                    precipitation_instant,
                    cloudcover,
                    windspeed_10m,
                    windspeed_100m,
                    windspeed_200m,
                    shortwave_radiation,
                    shortwave_radiation_instant
                FROM
                    (
                    SELECT
                        date_time,
                        version,
                        version_diff,
                        coordinate,
                        scope_name as province,
                        temperature_2m,
                        apparent_temperature,
                        humidity,
                        total_precipitation,
                        precipitation_instant,
                        cloudcover,
                        windspeed_10m,
                        windspeed_100m,
                        windspeed_200m,
                        shortwave_radiation,
                        shortwave_radiation_instant,
                        update_time,
                        {row_number_command}
                    FROM
                        {table_name} 
                    WHERE
                        {time_span}
                        AND meteo_scope = 'province'
                    ) subquery 
                WHERE
                    rank_id = 1;
                """
            else:
                query_msg = f"""
                    select
                        date_time,
                        AVG(temperature_2m) as temperature_2m,
                        AVG(apparent_temperature) as apparent_temperature,
                        AVG(humidity) as humidity,
                        AVG(precipitation_instant) as precipitation_instant,
                        AVG(cloudcover) as cloudcover,
                        AVG(windspeed_10m) as windspeed_10m,
                        AVG(windspeed_100m) as windspeed_100m,
                        AVG(windspeed_200m) as windspeed_200m,
                        AVG(shortwave_radiation_instant) as shortwave_radiation_instant
                    from
                        (
                        select
                            date_time,
                            info_date,
                            info_time,
                            version,
                            version_date,
                            version_time,
                            version_diff,
                            province,
                            city
                            district,
                            station_code,
                            temperature_2m,
                            apparent_temperature,
                            humidity,
                            precipitation_instant,
                            cloudcover,
                            windspeed_10m,
                            windspeed_100m,
                            windspeed_200m,
                            shortwave_radiation_instant,
                            row_number() over (partition by date_time, station_code order by version desc, update_time desc) as rank_id
                        from
                            {table_name}
                        where
                            date_time between '{start_date} 00:00:00' and '{end_date} 23:59:59'
                            AND province = '江苏省'
                            AND city IN ('南京市','无锡市','徐州市','常州市','苏州市','南通市','连云港市','淮安市','盐城市','扬州市','镇江市','泰州市','宿迁市') 
                        ) t1
                    where
                        rank_id = 1
                    group by
                        date_time
                    order by
                        date_time
                    """
                            # AND province = '江苏省'
                            # AND city IN ('南京市','无锡市','徐州市','常州市','苏州市','南通市','连云港市','淮安市','盐城市','扬州市','镇江市','泰州市','宿迁市') 
                            # AND province = '安徽省'
                            # AND city IN ('合肥市','黄山市','芜湖市','马鞍山市','安庆市','淮南市','阜阳市','淮北市','铜陵市','亳州市','宣城市','蚌埠市','六安市','滁州市','宿州市') 


        elif meteo_scope == 'station':
            query_msg = f"""
                SELECT
                    date_time,
                    version,
                    version_diff,
                    city,
                    district,
                    station_code,
                    temperature_2m,
                    apparent_temperature,
                    humidity,
                    precipitation_instant,
                    cloudcover,
                    windspeed_10m,
                    windspeed_100m,
                    windspeed_200m,
                    shortwave_radiation_instant
                FROM
                    (
                    SELECT
                        *,
                        {row_number_command}
                    FROM
                        {table_name} 
                    WHERE
                        {time_span}
                        AND province = '内蒙古自治区'
                        AND city IN ('阿拉善盟', '乌海市', '巴彦淖尔市','鄂尔多斯市','包头市','呼和浩特市','乌兰察布市' ,'锡林郭勒盟') 
                    ) subquery 
                WHERE
                    rank_id = 1;
                """
        elif meteo_scope == 'city':

            if self.meteo_weighted:
                if self.version_num == 0:
                    row_number_command = 'ROW_NUMBER() OVER ( PARTITION BY date_time, scope_name ORDER BY version DESC, update_time DESC ) AS rank_id'
                else:
                    row_number_command = 'ROW_NUMBER() OVER ( PARTITION BY date_time, scope_name, version ORDER BY update_time DESC ) AS rank_id'
                query_msg = f"""
                SELECT
                    date_time,
                    version,
                    version_diff,
                    city,
                    coordinate,
                    temperature_2m,
                    apparent_temperature,
                    humidity,
                    total_precipitation,
                    precipitation_instant,
                    cloudcover,
                    windspeed_10m,
                    windspeed_100m,
                    windspeed_200m,
                    shortwave_radiation,
                    shortwave_radiation_instant
                FROM
                    (
                    SELECT
                        date_time,
                        version,
                        version_diff,
                        scope_name as city,
                        coordinate,
                        temperature_2m,
                        apparent_temperature,
                        humidity,
                        total_precipitation,
                        precipitation_instant,
                        cloudcover,
                        windspeed_10m,
                        windspeed_100m,
                        windspeed_200m,
                        shortwave_radiation,
                        shortwave_radiation_instant,
                        update_time,
                        {row_number_command}
                    FROM
                        {table_name} 
                    WHERE
                        {time_span}
                        AND meteo_scope = 'city'
                    ) subquery 
                WHERE
                    rank_id = 1;
                """
            else:

                query_msg = f"""
                    SELECT
                        date_time,
                        city,

                        ANY(version) AS version,
                        ANY(version_diff) AS version_diff,
                        SUM(temperature_2m) / SUM(1) AS temperature_2m,
                        SUM(apparent_temperature) / SUM(1) AS apparent_temperature,
                        SUM(humidity) / SUM(1) AS humidity,
                        SUM(pressure_surface) / SUM(1) AS pressure_surface,
                        SUM(precipitation_instant) / SUM(1) AS precipitation_instant,
                        SUM(cloudcover) / SUM(1) AS cloudcover,
                        SUM(windspeed_10m) / SUM(1) AS windspeed_10m,
                        SUM(windspeed_100m) / SUM(1) AS windspeed_100m,
                        SUM(windspeed_200m) / SUM(1) AS windspeed_200m,
                        SUM(winddir_10m) / SUM(1) AS winddir_10m,
                        SUM(winddir_100m) / SUM(1) AS winddir_100m,
                        SUM(winddir_200m) / SUM(1) AS winddir_200m,
                        SUM(shortwave_radiation_instant) / SUM(1) AS shortwave_radiation_instant
                    FROM
                        (
                        SELECT
                            *,
                            {row_number_command}
                        FROM
                            {table_name} 
                        WHERE
                            {time_span}
                            
                            AND province = '江苏省'
                            AND city IN ('南京市','无锡市','徐州市','常州市','苏州市','南通市','连云港市','淮安市','盐城市','扬州市','镇江市','泰州市','宿迁市') 

                        ) subquery 
                    WHERE
                        rank_id = 1
                    GROUP BY
                        date_time, city
                    ORDER BY
                        date_time
                    """  

                row_number_command = 'ROW_NUMBER() OVER ( PARTITION BY date, key_id ORDER BY update_time DESC ) AS rank_id'
                time_span = f"date BETWEEN '{start_date} 00:00:00' AND '{end_date} 23:59:59'"
                query_msg_sea = f"""
                    SELECT
                        date,
                        key_id,
                        windspeed_100
                    FROM
                        (
                        SELECT
                            *,
                            {row_number_command}
                        FROM
                            {table_name_sea} 
                        WHERE
                            {time_span}
                        ) subquery
                    WHERE
                        rank_id = 1
                    GROUP BY
                        date, key_id
                    ORDER BY
                        date
                    """  

        return query_msg, query_msg_sea

    def load_city_data(self,
                       start_date: str,
                       end_date: str,
                       ):
        table_name = METEO_SOURCE[self._data_source]
        meteo_db = METEO_DORIS_CONST
        
        table_name_sea = METEO_SEA_SOURCE[self._data_source]
        meteo_db_sea = SEA_WIND_CONST

        query_msg,query_msg_sea = self._get_mysql_query(start_date=start_date, end_date=end_date, table_name=table_name,
                                          table_name_sea=table_name_sea, meteo_scope='city')


        # query_msg, query_msg_sea = self._get_mysql_query(start_date=start_date, end_date=end_date, table_name=table_name,
        #                                   meteo_scope='city')
        logger.debug(query_msg)
        pdf_weather_source = read_mysql(
            mysql_info=MysqlInfo(
                mysql_url=meteo_db.mysql_url,
                mysql_user=meteo_db.mysql_user,
                mysql_password=meteo_db.mysql_password,
                mysql_port=meteo_db.mysql_port,
                mysql_db=meteo_db.mysql_db,
            ),
            query=query_msg,
        )
        

        logger.debug("Finish reading mysql")
        pdf_weather_source['date_time'] = pdf_weather_source['date_time'].astype('str')
        pdf_weather_source['shortwave_radiation_instant'] = pdf_weather_source['shortwave_radiation_instant'].interpolate(method='linear')
        pdf_weather_source['precipitation_instant'] = pdf_weather_source['precipitation_instant'].interpolate(method='linear')

        # 气象特殊值后处理
        pdf_weather_source['humidity'] = pdf_weather_source.apply(lambda x: min(100, max(0, x['humidity'])), axis=1)
        pdf_weather_source['cloudcover'] = pdf_weather_source.apply(lambda x: min(100, max(0, x['cloudcover'])), axis=1)
        pdf_weather_source['precipitation_instant'] = pdf_weather_source.apply(lambda x: max(0, x['precipitation_instant']), axis=1)
        pdf_weather_source['shortwave_radiation_instant'] = pdf_weather_source.apply(lambda x: max(0, x['shortwave_radiation_instant']), axis=1)

        logger.info(f"插值前数据行数: {len(pdf_weather_source)}, 时间分辨率: 逐小时")
        non_numeric_cols_before = pdf_weather_source.select_dtypes(exclude=['number']).columns.tolist()
        pdf_weather_source['date_time'] = pd.to_datetime(pdf_weather_source['date_time'])

        meta_cols = [c for c in pdf_weather_source.columns if c not in [
            'date_time', 'city', 'coordinate', 'version', 'version_diff', 'meteo_source']
                     and pdf_weather_source[c].dtype == 'object']
        # 逐城市重采样为15分钟
        resampled_parts = []
        for city, group in pdf_weather_source.groupby('city'):
            group = group.set_index('date_time').sort_index()
            # 对数值列做15分钟重采样+线性插值
            numeric_cols = group.select_dtypes(include=['number']).columns.tolist()
            group_numeric = group[numeric_cols].resample('15min').interpolate(method='linear')
            # 对非数值列（city, version等）做前向填充
            non_numeric_cols = [c for c in group.columns if c not in numeric_cols]
            group_non_numeric = group[non_numeric_cols].resample('15min').ffill()
            # 合并
            group_resampled = pd.concat([group_numeric, group_non_numeric], axis=1)
            group_resampled = group_resampled.reset_index()
            resampled_parts.append(group_resampled)
        pdf_weather_source = pd.concat(resampled_parts, ignore_index=True)

        # 夜间辐照度置零（19:00-06:00 线性插值可能在夜间产生微弱正值）
        if 'shortwave_radiation_instant' in pdf_weather_source.columns:
            hour = pdf_weather_source['date_time'].dt.hour
            night_mask = (hour < 6) | (hour >= 20)
            pdf_weather_source.loc[night_mask, 'shortwave_radiation_instant'] = 0.0
        logger.info(f"插值后数据行数: {len(pdf_weather_source)}, 时间分辨率: 逐15分钟")
        pdf_weather_source['date_time'] = pdf_weather_source['date_time'].astype('str')
        pdf_weather = pd.DataFrame()
        city_list = sorted(pdf_weather_source['city'].unique())

        wind_capa = JIANGSU_CITY_WIND_CAPACITY
        solar_capa = JIANGSU_CITY_SOLAR_CAPACITY
        wind_total = sum(wind_capa.values())  # 风电总装机容量 (MW)
        solar_total = sum(solar_capa.values())  # 光伏总装机容量 (MW)

        for city in city_list:
            city_wind_capa = wind_capa.get(city, 0)  # 该城市风电装机 (MW)
            city_solar_capa = solar_capa.get(city, 0)  # 该城市光伏装机 (MW)

            city_mask = pdf_weather_source['city'] == city
            # 风电相关气象要素：值 × 容量（中间列，用于聚合）
            for wind_feat in ['windspeed_10m', 'windspeed_100m', 'windspeed_200m', 'pressure_surface']:
                if wind_feat in pdf_weather_source.columns:
                    pdf_weather_source.loc[city_mask, f'{wind_feat}_capa_w'] = (
                        pdf_weather_source.loc[city_mask, wind_feat] * city_wind_capa
                    )
            # 光伏相关气象要素：值 × 容量（中间列，用于聚合）
            for solar_feat in ['temperature_2m', 'apparent_temperature', 'cloudcover', 'humidity', 'shortwave_radiation_instant']:
                if solar_feat in pdf_weather_source.columns:
                    pdf_weather_source.loc[city_mask, f'{solar_feat}_capa_w'] = (
                        pdf_weather_source.loc[city_mask, solar_feat] * city_solar_capa
                    )

        logger.info(f"容量加权完成: 风电总装机{wind_total}MW, 光伏总装机{solar_total}MW")

        # 夜间辐照度加权列也置零（capa_w列 = 原始值 × 权重，夜间也应为0）
        if 'shortwave_radiation_instant_capa_w' in pdf_weather_source.columns:
            hour = pd.to_datetime(pdf_weather_source['date_time']).dt.hour
            night_mask = (hour < 6) | (hour >= 20)
            pdf_weather_source.loc[night_mask, 'shortwave_radiation_instant_capa_w'] = 0.0

        rename_stc_list = []
        for city in city_list:
            pdf_stc = pdf_weather_source[pdf_weather_source['city'] == city].copy()

            rename_stc_list = []
            rename_stc_dict = {}
            for feat in WEATHER_FEATURE_LIST:
                weighted_col = f'{feat}_capa_w'
                if weighted_col in pdf_stc.columns:
                    new_name = f'{feat}_capa_w_{city}'
                    rename_stc_dict[weighted_col] = new_name
                    rename_stc_list.append(new_name)

                if feat in pdf_stc.columns:
                    new_name = f'{feat}_{city}'
                    rename_stc_dict[feat] = new_name
                    rename_stc_list.append(new_name)
            city_wind_capa = wind_capa.get(city, 0)
            city_solar_capa = solar_capa.get(city, 0)
            pdf_stc['capacity_wind_' + city] = city_wind_capa  # 该城市风电装机 (MW)
            pdf_stc['capacity_solar_' + city] = city_solar_capa  # 该城市光伏装机 (MW)
            rename_stc_list.append('capacity_wind_' + city)
            rename_stc_list.append('capacity_solar_' + city)

            pdf_stc = pdf_stc.rename(columns=rename_stc_dict)
            if pdf_weather.empty:
                pdf_weather = pdf_stc[['date_time', 'version', 'version_diff'] + rename_stc_list].copy()
            else:
                pdf_weather = pd.merge(pdf_weather, pdf_stc[['date_time', 'version', 'version_diff'] + rename_stc_list].copy(), on=['date_time', 'version', 'version_diff'], how='inner')
        logger.debug(pdf_weather.head())

        feature_list = ['date_time', 'version', 'version_diff','is_strong_wind']
        meteorology_cols = [col for col in pdf_weather.columns
                            if (col not in feature_list)
                            and ('windspeed_' in col)
                            and ('_capa_w_' not in col)  # 使用原始气象列，不再对装机加权列做滞后
                            and ('_lag_' not in col)  # 排除已有的滞后列
                            and ('log' not in col)]
        a_meteorology_cols = []
        for i in meteorology_cols:
            for j in JIANGSU_CITY2_WIND:
                if j in i:
                    a_meteorology_cols.append(i)
        meteorology_cols = a_meteorology_cols.copy()

        lag_orders = [4, 12, 24, 36, 48, 72, 96]  # 对应 1h, 3h, 6h, 9h, 12h, 18h, 24h
        max_lag = max(lag_orders)  # 最大滞后阶数（24小时）
        def add_lag_features(df, cols, lags, train_flag=True):
            """
            添加滞后特征
            train_flag=True: 训练模式，正常添加滞后特征，后续会dropna
            train_flag=False: 预测模式，正常添加滞后特征，NaN值使用前后值填充
            """
            lag_dfs = []
            for col in cols:
                for lag in lags:
                    lag_col_name = f"{col}_lag_{lag}"
                    # 统一使用shift添加滞后特征
                    lag_df = pd.DataFrame({lag_col_name: df[col].shift(periods=lag)}, index=df.index)
                    lag_dfs.append(lag_df)
            df = pd.concat([df] + lag_dfs, axis=1)

            if not train_flag:
                # 预测模式：对滞后特征的NaN值使用前后值填充
                lag_cols = [c for c in df.columns if '_lag_' in c]
                for lag_col in lag_cols:
                    # 先用ffill填充（用前面的值），再用bfill填充（用后面的值）
                    df[lag_col] = df[lag_col].ffill().bfill()
                    # 如果仍有NaN（全空情况），填充为0
                    if df[lag_col].isna().any():
                        df[lag_col] = df[lag_col].fillna(0)
            return df

        # 根据训练/预测模式差异化处理滞后特征
        if self._train_flag:
            pdf_weather = add_lag_features(pdf_weather, meteorology_cols, lag_orders, train_flag=True)
            pdf_weather = pdf_weather.dropna().reset_index(drop=True)
            logger.info(f"[训练] 添加滞后特征后样本数: {len(pdf_weather)}")
        else:
            pdf_weather = add_lag_features(pdf_weather, meteorology_cols, lag_orders, train_flag=False)
            logger.info(f"[预测] 滞后特征填充为0，样本数: {len(pdf_weather)}")

        pdf_weather['data_hour'] = pd.to_datetime(pdf_weather['date_time']).dt.hour
        pdf_weather['data_hour_sin'] = np.sin((2*np.pi*pdf_weather['data_hour'])/24)
        pdf_weather['data_hour_cos'] = np.cos((2*np.pi*pdf_weather['data_hour'])/24)

        pdf_weather['data_month'] = pd.to_datetime(pdf_weather['date_time']).dt.month
        pdf_weather['data_month_sin'] = np.sin((2*np.pi*pdf_weather['data_month'])/12)
        pdf_weather['data_month_cos'] = np.cos((2*np.pi*pdf_weather['data_month'])/12)

        # 15分钟粒度增加分钟级时间编码
        pdf_weather['data_minute'] = pd.to_datetime(pdf_weather['date_time']).dt.minute
        pdf_weather['data_minute_sin'] = np.sin(2 * np.pi * pdf_weather['data_minute'] / 60)
        pdf_weather['data_minute_cos'] = np.cos(2 * np.pi * pdf_weather['data_minute'] / 60)


        pdf_weather['data_day'] = pd.to_datetime(pdf_weather['date_time']).dt.day
        pdf_weather['data_ymd'] = pd.to_datetime(pdf_weather['date_time']).dt.strftime("%Y-%m-%d")
        # pdf_weather['holiday_tag'] = pdf_weather['data_ymd'].map(lambda date: date_to_holiday_tag(date))

        # 差值计算使用原始气象列（不再使用装机加权后的 capa_w 列）
        pdf_weather['windspeed_100m_南通_扬州_diff'] = pdf_weather['windspeed_100m_南通市'] - pdf_weather['windspeed_100m_扬州市']
        pdf_weather['windspeed_100m_盐城_扬州_diff'] = pdf_weather['windspeed_100m_盐城市'] - pdf_weather['windspeed_100m_扬州市']
        pdf_weather['windspeed_100m_盐城_淮安_diff'] = pdf_weather['windspeed_100m_盐城市'] - pdf_weather['windspeed_100m_淮安市']
        pdf_weather['windspeed_100m_连云港_徐州_diff'] = pdf_weather['windspeed_100m_连云港市'] - pdf_weather['windspeed_100m_徐州市']

        southern_cities = ['南通市','常州市','无锡市','苏州市']
        northern_cities = ['盐城市','连云港市','宿迁市','徐州市','淮安市']

        # 区域均值使用原始气象列（不再使用装机加权后的 capa_w 列）
        col_south = pdf_weather[[f'windspeed_100m_{city}' for city in southern_cities if f'windspeed_100m_{city}' in pdf_weather.columns]].mean(axis=1).rename("windspeed_100m_southern_mean")
        col_north = pdf_weather[[f'windspeed_100m_{city}' for city in northern_cities if f'windspeed_100m_{city}' in pdf_weather.columns]].mean(axis=1).rename("windspeed_100m_northern_mean")
        pdf_weather = pd.concat([pdf_weather, col_south, col_north], axis=1)
        # pdf_weather['windspeed_100m_southern_mean'] = pdf_weather[[f'windspeed_100m_{city}' for city in southern_cities if f'windspeed_100m_{city}' in pdf_weather.columns]].mean(axis=1)
        # pdf_weather['windspeed_100m_northern_mean'] = pdf_weather[[f'windspeed_100m_{city}' for city in northern_cities if f'windspeed_100m_{city}' in pdf_weather.columns]].mean(axis=1)

        # 保留 wind_capa / solar_capa 供后续非输入派生列使用；模型输入不再选择装机加权气象列。
        wind_capa = JIANGSU_CITY_WIND_CAPACITY
        solar_capa = JIANGSU_CITY_SOLAR_CAPACITY

        # NOTE: 气象×装机容量交互特征已禁用，避免构造无用的 _x_capa_ 列并减少 DataFrame 碎片化。
        # 若后续业务需要，可取消下方注释恢复。
        """
        interaction_dfs = []

        # 风电：风速 × 装机
        wind_interaction_feats = []
        for city, capa in wind_capa.items():
            if capa <= 0:
                continue
            for wind_feat in ['windspeed_10m', 'windspeed_100m', 'windspeed_200m']:
                raw_col = f'{wind_feat}_{city}'
                if raw_col in pdf_weather.columns:
                    new_col = f'{wind_feat}_x_capa_{city}'
                    interaction_dfs.append(pd.DataFrame({new_col: pdf_weather[raw_col].values * capa}, index=pdf_weather.index))
                    wind_interaction_feats.append(new_col)
        if wind_interaction_feats:
            logger.info(f"[交互特征] 构造风电×装机容量交互特征: {len(wind_interaction_feats)}个")

        # 光伏：辐照、温度 × 装机
        solar_interaction_feats = []
        for city, capa in solar_capa.items():
            if capa <= 0:
                continue
            for solar_feat in ['shortwave_radiation_instant', 'temperature_2m', 'cloudcover']:
                raw_col = f'{solar_feat}_{city}'
                if raw_col in pdf_weather.columns:
                    new_col = f'{solar_feat}_x_capa_{city}'
                    interaction_dfs.append(pd.DataFrame({new_col: pdf_weather[raw_col].values * capa}, index=pdf_weather.index))
                    solar_interaction_feats.append(new_col)
        if solar_interaction_feats:
            logger.info(f"[交互特征] 构造光伏×装机容量交互特征: {len(solar_interaction_feats)}个")

        # 批量添加所有交互特征，避免逐列插入导致碎片化
        if interaction_dfs:
            pdf_weather = pd.concat([pdf_weather] + interaction_dfs, axis=1)
        """

        # 2. 温度衰减特征：组件温度高于25°C时效率下降（约-0.35%/°C）
        solar_cities = JIANGSU_CITY2_SOLAR
        new_cols_dict = {}
        for city in solar_cities:
            sw_col = f'shortwave_radiation_instant_{city}'
            cc_col = f'cloudcover_{city}'
            t2m_col = f'temperature_2m_{city}'
            if sw_col in pdf_weather.columns and cc_col in pdf_weather.columns:
                new_cols_dict[f'cloudcover_x_shortwave_{city}'] = pdf_weather[cc_col] * pdf_weather[sw_col]
                # 统一添加滞后特征
                new_cols_dict[f'shortwave_rad_lag_1_{city}'] = pdf_weather[sw_col].shift(periods=4)   # 1小时前
                new_cols_dict[f'shortwave_rad_lag_3_{city}'] = pdf_weather[sw_col].shift(periods=12)  # 3小时前
                new_cols_dict[f'shortwave_rad_diff_1h_{city}'] = pdf_weather[sw_col].diff(periods=4)   # 1小时变化
                new_cols_dict[f'shortwave_rad_diff_3h_{city}'] = pdf_weather[sw_col].diff(periods=12)  # 3小时变化
            if t2m_col in pdf_weather.columns:
                new_cols_dict[f'temp_above_25_{city}'] = (pdf_weather[t2m_col] - 25).clip(lower=0)

        if new_cols_dict:
            pdf_weather = pd.concat([pdf_weather, pd.DataFrame(new_cols_dict, index=pdf_weather.index)], axis=1)

        # 预测模式下对光伏滞后特征NaN值进行填充
        if not self._train_flag:
            solar_lag_cols = [c for c in pdf_weather.columns if 'shortwave_rad_lag_' in c or 'shortwave_rad_diff_' in c]
            for solar_lag_col in solar_lag_cols:
                pdf_weather[solar_lag_col] = pdf_weather[solar_lag_col].ffill().bfill()
                if pdf_weather[solar_lag_col].isna().any():
                    pdf_weather[solar_lag_col] = pdf_weather[solar_lag_col].fillna(0)

        # 聚合特征：同样批量收集后一次性concat，避免碎片化
        cc_sw_cols = [c for c in pdf_weather.columns if 'cloudcover_x_shortwave_' in c]
        temp25_cols = [c for c in pdf_weather.columns if 'temp_above_25_' in c]
        sw_lag1_cols = [c for c in pdf_weather.columns if 'shortwave_rad_lag_1_' in c]
        sw_lag3_cols = [c for c in pdf_weather.columns if 'shortwave_rad_lag_3_' in c]
        sw_diff1_cols = [c for c in pdf_weather.columns if 'shortwave_rad_diff_1h_' in c]
        sw_diff3_cols = [c for c in pdf_weather.columns if 'shortwave_rad_diff_3h_' in c]

        agg_dict = {}
        if cc_sw_cols:
            agg_dict['cloudcover_x_shortwave'] = pdf_weather[cc_sw_cols].mean(axis=1)
        if temp25_cols:
            agg_dict['temp_above_25'] = pdf_weather[temp25_cols].mean(axis=1)
        if sw_lag1_cols:
            agg_dict['shortwave_rad_lag_1'] = pdf_weather[sw_lag1_cols].mean(axis=1)
        if sw_lag3_cols:
            agg_dict['shortwave_rad_lag_3'] = pdf_weather[sw_lag3_cols].mean(axis=1)
        if sw_diff1_cols:
            agg_dict['shortwave_rad_diff_1h'] = pdf_weather[sw_diff1_cols].mean(axis=1)
        if sw_diff3_cols:
            agg_dict['shortwave_rad_diff_3h'] = pdf_weather[sw_diff3_cols].mean(axis=1)

        if agg_dict:
            pdf_weather = pd.concat([pdf_weather, pd.DataFrame(agg_dict, index=pdf_weather.index)], axis=1)

        pdf_weather['meteo_source'] = self._data_source

        # 消除DataFrame碎片化，提升后续操作性能
        pdf_weather = pdf_weather.copy()

        pdf_weather.to_csv('./jiangsu_weather.csv',index=False)

        self._data = pdf_weather

    def load_weighted_city_data(self,
                           start_date: str,
                           end_date: str):
        table_items = METEO_SOURCE[self._data_source+'_W']
        meteo_db = METEO_TEST_DORIS_CONST

        pdf_weather_source = pd.DataFrame()
        pdf_weather_res = pd.DataFrame()
        for key, table_name in table_items.items():
            query_msg = self._get_mysql_query(start_date=start_date, end_date=end_date, table_name=table_name,
                                              meteo_scope='city')
            logger.debug(query_msg)
            pdf_weather_source = read_mysql(
                mysql_info=MysqlInfo(
                    mysql_url=meteo_db.mysql_url,
                    mysql_user=meteo_db.mysql_user,
                    mysql_password=meteo_db.mysql_password,
                    mysql_port=meteo_db.mysql_port,
                    mysql_db=meteo_db.mysql_db,
                ),
                query=query_msg,
            )
            logger.debug("Finish reading mysql")
            if not self._train_flag and self.use_interpolation:
                pdf_weather_source = interpolate_by_service(pdf_weather_source, 
                                                            interpolate_column=['cloudcover', 'humidity', 'shortwave_radiation', 'temperature_2m', 'windspeed_10m', 'windspeed_100m'],
                                                            meta_column=['date_time', 'version', 'version_diff', 'city', 'coordinate'])
            pdf_weather_source['date_time'] = pdf_weather_source['date_time'].astype('str')
            pdf_weather_source['shortwave_radiation_instant'] = pdf_weather_source['shortwave_radiation_instant'].interpolate(method='linear')
            pdf_weather_source['precipitation_instant'] = pdf_weather_source['precipitation_instant'].interpolate(method='linear')

            # 气象特殊值后处理
            pdf_weather_source['humidity'] = pdf_weather_source.apply(lambda x: min(100, max(0, x['humidity'])), axis=1)
            pdf_weather_source['cloudcover'] = pdf_weather_source.apply(lambda x: min(100, max(0, x['cloudcover'])), axis=1)
            pdf_weather_source['precipitation_instant'] = pdf_weather_source.apply(lambda x: max(0, x['precipitation_instant']), axis=1)
            pdf_weather_source['shortwave_radiation_instant'] = pdf_weather_source.apply(lambda x: max(0, x['shortwave_radiation_instant']), axis=1)

            pdf_weather = pd.DataFrame()
            city_list = sorted(pdf_weather_source['city'].unique())
            rename_stc_list = []
            for city in city_list:
                pdf_stc = pdf_weather_source[pdf_weather_source['city'] == city].copy()
                rename_stc_list = [f'{feat}_{city}_{key}' for feat in WEATHER_FEATURE_LIST]
                rename_stc_dict = dict(zip(WEATHER_FEATURE_LIST, rename_stc_list))
                pdf_stc = pdf_stc.rename(columns=rename_stc_dict)
                if pdf_weather.empty:
                    pdf_weather = pdf_stc[['date_time', 'version', 'version_diff'] + rename_stc_list].copy()
                else:
                    pdf_weather = pd.merge(pdf_weather, pdf_stc[['date_time', 'version', 'version_diff'] + rename_stc_list].copy(), on=['date_time', 'version', 'version_diff'], how='inner')
            logger.debug(pdf_weather.head())

            if pdf_weather_res.empty:
                pdf_weather_res = pdf_weather.copy()
            else:
                pdf_weather_res = pd.merge(pdf_weather_res, pdf_weather, on=['date_time', 'version', 'version_diff'], how='inner')

        pdf_weather_res['meteo_source'] = self._data_source

        self._data = pdf_weather_res

    def load_weighted_province_data(self,
                                    start_date: str,
                                    end_date: str):
        table_items = METEO_SOURCE[self._data_source+'_W']
        meteo_db = METEO_TEST_DORIS_CONST

        pdf_weather_source = pd.DataFrame()
        pdf_weather_res = pd.DataFrame()
        for key, table_name in table_items.items():
            query_msg = self._get_mysql_query(start_date=start_date, end_date=end_date, table_name=table_name,
                                              meteo_scope='province')
            logger.debug(query_msg)
            pdf_weather_source = read_mysql(
                mysql_info=MysqlInfo(
                    mysql_url=meteo_db.mysql_url,
                    mysql_user=meteo_db.mysql_user,
                    mysql_password=meteo_db.mysql_password,
                    mysql_port=meteo_db.mysql_port,
                    mysql_db=meteo_db.mysql_db,
                ),
                query=query_msg,
            )
            logger.debug("Finish reading mysql")
            pdf_weather_source['date_time'] = pdf_weather_source['date_time'].astype('str')
            if not self._train_flag and self.use_interpolation:
                pdf_weather_source = interpolate_by_service(pdf_weather_source, 
                                                            interpolate_column=['cloudcover', 'humidity', 'shortwave_radiation', 'temperature_2m', 'windspeed_10m', 'windspeed_100m'],
                                                            meta_column=['date_time', 'version', 'version_diff', 'province', 'coordinate'])
            pdf_weather_source['date_time'] = pdf_weather_source['date_time'].astype('str')
            pdf_weather_source['shortwave_radiation_instant'] = pdf_weather_source['shortwave_radiation_instant'].interpolate(method='linear')
            pdf_weather_source['precipitation_instant'] = pdf_weather_source['precipitation_instant'].interpolate(method='linear')

            # 气象特殊值后处理
            pdf_weather_source['humidity'] = pdf_weather_source.apply(lambda x: min(100, max(0, x['humidity'])), axis=1)
            pdf_weather_source['cloudcover'] = pdf_weather_source.apply(lambda x: min(100, max(0, x['cloudcover'])), axis=1)
            pdf_weather_source['precipitation_instant'] = pdf_weather_source.apply(lambda x: max(0, x['precipitation_instant']), axis=1)
            pdf_weather_source['shortwave_radiation_instant'] = pdf_weather_source.apply(lambda x: max(0, x['shortwave_radiation_instant']), axis=1)

            pdf_weather = pd.DataFrame()
            rename_stc_list = [f'{feat}_{key}' for feat in WEATHER_FEATURE_LIST]
            rename_stc_dict = dict(zip(WEATHER_FEATURE_LIST, rename_stc_list))
            pdf_weather = pdf_weather_source.rename(columns=rename_stc_dict)
            logger.debug(pdf_weather.head())

            if pdf_weather_res.empty:
                pdf_weather_res = pdf_weather.copy()
            else:
                pdf_weather_res = pd.merge(pdf_weather_res, pdf_weather, on=['date_time', 'version', 'version_diff'], how='inner')

        pdf_weather_res['meteo_source'] = self._data_source

        self._data = pdf_weather_res

    def load_station_data(self,
                          start_date: str,
                          end_date: str,
                          ):
        table_name = METEO_SOURCE[self._data_source]

        query_msg = self._get_mysql_query(start_date=start_date, end_date=end_date, table_name=table_name,
                                          meteo_scope='station')
        logger.debug(query_msg)
        pdf_weather_source = read_mysql(
            mysql_info=MysqlInfo(
                mysql_url=METEO_DORIS_CONST.mysql_url,
                mysql_user=METEO_DORIS_CONST.mysql_user,
                mysql_password=METEO_DORIS_CONST.mysql_password,
                mysql_port=METEO_DORIS_CONST.mysql_port,
                mysql_db=METEO_DORIS_CONST.mysql_db,
            ),
            query=query_msg,
        )
        logger.debug("Finish reading mysql")
        pdf_weather_source['date_time'] = pdf_weather_source['date_time'].astype('str')
        pdf_weather_source['shortwave_radiation_instant'] = pdf_weather_source['shortwave_radiation_instant'].interpolate(method='linear')
        pdf_weather_source['precipitation_instant'] = pdf_weather_source['precipitation_instant'].interpolate(method='linear')

        # 气象特殊值后处理
        pdf_weather_source['humidity'] = pdf_weather_source.apply(lambda x: min(100, max(0, x['humidity'])), axis=1)
        pdf_weather_source['cloudcover'] = pdf_weather_source.apply(lambda x: min(100, max(0, x['cloudcover'])), axis=1)
        pdf_weather_source['precipitation_instant'] = pdf_weather_source.apply(lambda x: max(0, x['precipitation_instant']), axis=1)
        pdf_weather_source['shortwave_radiation_instant'] = pdf_weather_source.apply(lambda x: max(0, x['shortwave_radiation_instant']), axis=1)

        pdf_weather = pd.DataFrame()
        station_list = list(set(WIND_POWER_STATION_CODE_LIST) | set(SOLAR_POWER_STATION_CODE_LIST))
        rename_stc_list = []
        for s_code in station_list:
            pdf_stc = pdf_weather_source[pdf_weather_source['station_code'] == str(s_code)].copy()
            rename_stc_list = [f'{feat}_{s_code}' for feat in WEATHER_FEATURE_LIST]
            rename_stc_dict = dict(zip(WEATHER_FEATURE_LIST, rename_stc_list))
            pdf_stc = pdf_stc.rename(columns=rename_stc_dict)
            if pdf_stc.empty:
                print(s_code)
            # print(pdf_stc.head())
            if pdf_weather.empty:
                pdf_weather = pdf_stc[['date_time', 'version', 'version_diff'] + rename_stc_list].copy()
            else:
                pdf_weather = pd.merge(pdf_weather, pdf_stc[['date_time', 'version', 'version_diff'] + rename_stc_list].copy(), on=['date_time', 'version', 'version_diff'], how='inner')
        logger.debug(pdf_weather.head())

        pdf_weather['meteo_source'] = self._data_source

        self._data = pdf_weather

    def load_province_data(self,
                           start_date: str,
                           end_date: str,
                           ):
        # EC ENS预报的ads表历史数据未回溯完成, 历史数据暂时用ods表
        table_name = METEO_SOURCE[self._data_source]

        query_msg = self._get_mysql_query(start_date=start_date, end_date=end_date, table_name=table_name,
                                          meteo_scope='province')
        pdf_weather_source = read_mysql(
            mysql_info=MysqlInfo(
                mysql_url=METEO_DORIS_CONST.mysql_url,
                mysql_user=METEO_DORIS_CONST.mysql_user,
                mysql_password=METEO_DORIS_CONST.mysql_password,
                mysql_port=METEO_DORIS_CONST.mysql_port,
                mysql_db=METEO_DORIS_CONST.mysql_db,
            ),
            query=query_msg
        )

        pdf_weather = pdf_weather_source.copy()
        pdf_weather['date_time'] = pdf_weather['date_time'].astype('str')
        rename_dict = dict(zip(self._org_weather_feat, WEATHER_FEATURE_LIST))
        pdf_weather = pdf_weather.rename(columns=rename_dict)
        pdf_weather['shortwave_radiation_instant'] = pdf_weather['shortwave_radiation_instant'].interpolate(method='linear')
        pdf_weather['precipitation_instant'] = pdf_weather['precipitation_instant'].interpolate(method='linear')

        # 气象特殊值后处理
        pdf_weather['humidity'] = pdf_weather.apply(lambda x: min(100, max(0, x['humidity'])), axis=1)
        pdf_weather['cloudcover'] = pdf_weather.apply(lambda x: min(100, max(0, x['cloudcover'])), axis=1)
        pdf_weather['precipitation_instant'] = pdf_weather.apply(lambda x: max(0, x['precipitation_instant']), axis=1)
        pdf_weather['shortwave_radiation_instant'] = pdf_weather.apply(lambda x: max(0, x['shortwave_radiation_instant']), axis=1)
        logger.debug(pdf_weather.head())

        pdf_weather['meteo_source'] = self._data_source

        self._data = pdf_weather
        
    @property
    def data(self) -> pd.DataFrame:
        return self._data


def get_data_by_hour(feature_list: Sequence[str],
                    start_date: str,
                    end_date: str,
                    meteo_scope: str = 'station',
                    version_num: int = 0,
                    train_flag: bool = False,
                    meteo_source: str = 'ENS_EXTEND',
                    capa_rev: bool = True,
                    version_start_date: str=None,
                    version_end_date: str=None,
                    task_name: str='by_day',
                    meteo_weighted: bool = False,
                    use_interpolation: bool = False,
                    **kwargs) -> pd.DataFrame:
    """统一获取日前/实时边界条件数据+气象数据, 用于模型训练/预测

    Args:
        feature_list (Sequence[str]): 模型使用的特征及标签, 支持: 'provincial_load', 'tie_line',
            'new_energy_total', 'new_energy_wind', 'new_energy_solar', 'non_market'
        start_date (str): 样本开始时间, 格式: '2024-10-01'
        end_date (str): 样本结束时间, 格式: '2024-10-31'
        use_stations (bool): 是否使用单台站数据
        version_num (int): 使用的气象数据版本, 0为使用最新版本
        meteo_source (str): 使用气象数据源
        capa_rev (bool): 是否使用修正后的标签值
    Returns:
        pd.DataFrame: 处理后的数据DataFrame
    """

    # 完整label列表
    # 'provincial_load', 'tie_line', 'new_energy_wind', 'new_energy_solar', 'hydropower', 'non_market' 
    pdf_data = pd.DataFrame()

    if feature_list:
        ahead_feat_list = [feat for feat in feature_list if 'ahead' in feat]
        logger.info("Start to get ahead data.")
        ahead_data = AheadData(
            start_date=start_date,
            end_date=end_date,
            column_list=ahead_feat_list,
            rev_flag=capa_rev
        ) if ahead_feat_list else None
        if ahead_data is not None:
            pdf_data = ahead_data.data.copy()
        logger.info("Finish getting ahead data.")

        real_feat_list = list(set(feature_list) - set(ahead_feat_list))
        logger.info("Start to get realtime data.")
        realtime_data = RealtimeData(
            start_date=start_date,
            end_date=end_date,
            column_list=real_feat_list,
            rev_flag=capa_rev
        ) if real_feat_list else None
        if pdf_data.empty and realtime_data is not None:
            pdf_data = realtime_data.data.copy()
        # elif realtime_data is not None:
        #     pdf_data = pd.merge(pdf_data, realtime_data.data.copy(), on='date_time', how='outer')
        logger.info("Finish getting realtime data.")
        pdf_data = pdf_data.infer_objects(copy=False)
        numeric_columns = pdf_data.select_dtypes(include=[np.number]).columns

        label_cols_in_feat = {c for c in feature_list if 'ahead' in c or 'realtime' in c}
        interp_cols = [c for c in numeric_columns if c not in label_cols_in_feat]
        if interp_cols:
            pdf_data[interp_cols] = pdf_data[interp_cols].interpolate(method='linear')

    # pdf_data.to_csv('./pdf_data_pre.csv')
    logger.info("Start to get meteo data.")
    meteo_data = MeteoDorisData(
        start_date=start_date,
        end_date=end_date,
        meteo_scope=meteo_scope,
        version_num=version_num,
        train_flag=train_flag,
        data_source=meteo_source,
        task_name=task_name,
        version_start_date=version_start_date,
        version_end_date=version_end_date,
        meteo_weighted=meteo_weighted,
        use_interpolation=use_interpolation
    )
    logger.info("Finish getting meteo data.")

    if feature_list:
        # data1 = meteo_data.data.copy()
        # data2 = pdf_data.copy()
        # data1['date_time'] = pd.to_datetime(data1['date_time'])
        # data2['date_time'] = pd.to_datetime(data2['date_time'])
        # pdf_data = pd.merge(data1, data2, on=['date_time'], how='left')
        pdf_data = pd.merge(meteo_data.data, pdf_data.copy(), on=['date_time'], how='left')
    else:
        pdf_data = meteo_data.data.copy()

    pdf_data['info_date'] = pdf_data['date_time'].map(lambda x: str_timestamp_to_str(x, input_format='%Y-%m-%d %H:%M:%S', output_format='%Y-%m-%d'))
    pdf_data['info_time'] = pdf_data['date_time'].map(lambda x: str_timestamp_to_str(x, input_format='%Y-%m-%d %H:%M:%S', output_format='%H:%M'))
    # pdf_data['info_time'] = pdf_data['date_time'].map(lambda x: str_timestamp_to_str(x, input_format='%Y-%m-%d %H:%M:%S', output_format='%H:%M', plus_minutes=15))
    # pdf_data['info_time'] = pdf_data['info_time'].str.replace('00:00', '24:00')
    pdf_data['time_idx'] = pdf_data['info_time'].map(lambda x: calculate_time_idx(x))


    return pdf_data


def get_data_average(feature_list: Sequence[str],
                    start_date: str=None,
                    end_date: str=None,
                    meteo_scope: str = 'station',
                    version_num: int = 0,
                    train_flag: bool = False,
                    meteo_source: str = 'ENS_EXTEND',
                    capa_rev: bool = True,
                    **kwargs
                    ):
    version_start_date = kwargs.get('version_start_date', None)
    version_end_date = kwargs.get('version_end_date', None)
    task_name = kwargs.get('task_name', 'ten_days')


    pdf_data = get_data_by_hour(feature_list=feature_list,
                                start_date=start_date,
                                end_date=end_date,
                                meteo_scope=meteo_scope,
                                version_num=version_num,
                                train_flag=train_flag,
                                meteo_source=meteo_source,
                                capa_rev=capa_rev,
                                version_start_date=version_start_date,
                                version_end_date=version_end_date,
                                task_name=task_name,
                                meteo_weighted=kwargs.get('meteo_weighted', False))
    
    if train_flag or (version_start_date is not None and version_end_date is not None):
        if task_name == 'ten_days':
            start_plus_hour = TEN_DAYS_PLUS_HOUR
            window_hours = 240
        else:
            start_plus_hour = MONTH_PLUS_HOUR
            window_hours = 720
        pdf_version_mean = process_average_data_by_versions(pdf_data, start_plus_hour, window_hours)
    else:
        # 如何是非训练且不提供版本区间，则对获取最新版本数据自动查找旬/月区间
        pdf_version_mean = process_average_data_by_date(pdf_data, start_date, end_date)
    return pdf_version_mean


def get_data(feature_list: Sequence[str],
             start_date: str,
             end_date: str,
             meteo_scope: str = 'station',
             version_num: int = 0,
             train_flag: bool = False,
             meteo_source: str = 'ENS_EXTEND',
             capa_rev: bool = True,
             avg_data: bool = False,
             **kwargs
             ):
    if not avg_data:
        pdf_data = get_data_by_hour(feature_list=feature_list,
                                    start_date=start_date,
                                    end_date=end_date,
                                    meteo_scope=meteo_scope,
                                    version_num=version_num,
                                    train_flag=train_flag,
                                    meteo_source=meteo_source,
                                    capa_rev=capa_rev,
                                    **kwargs)
    else:
        pdf_data = get_data_average(feature_list=feature_list,
                                    start_date=start_date,
                                    end_date=end_date,
                                    meteo_scope=meteo_scope,
                                    version_num=version_num,
                                    train_flag=train_flag,
                                    meteo_source=meteo_source,
                                    capa_rev=capa_rev,
                                    **kwargs)
        
    return pdf_data


# Deprecated
def submit_power_to_doris_server(pdf_data: pd.DataFrame,
                                 col_list: List[str],
                                 use_ahead: bool = False,
                                 task_name: str='by_day'):
    new_cols = ['realtime_' + col for col in col_list]

    pdf_rst = pdf_data.copy()
    pred_col_list = [col+'_pred' for col in col_list]
    rename_cols = dict(zip(pred_col_list, new_cols))
    pdf_rst.rename(columns=rename_cols, inplace=True)
    pdf_rst['realtime_new_energy'] = pdf_rst['realtime_new_energy_wind'] + pdf_rst['realtime_new_energy_solar']
    for col in ['realtime_new_energy', 'realtime_new_energy_wind', 'realtime_new_energy_solar']:
        pdf_rst[col] = pdf_rst[col].astype('float32').round(4)

    pdf_rst['info_date'] = pdf_rst.apply(lambda x: str_timestamp_to_str(x['date_time'], input_format='%Y-%m-%d %H:%M:%S', output_format='%Y-%m-%d'), axis=1)
    pdf_rst['info_time'] = pdf_rst.apply(lambda x: str_timestamp_to_str(x['date_time'], input_format='%Y-%m-%d %H:%M:%S', output_format='%H:%M', plus_minutes=15), axis=1)
    pdf_rst['info_time'] = pdf_rst['info_time'].str.replace('00:00', '24:00')
    pdf_rst['date_time'] = pdf_rst['date_time'].astype(str)
    pdf_rst['update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if 'version' in pdf_rst.columns:
        pdf_rst.rename(columns={'version': 'data_version'}, inplace=True)
        pdf_rst['data_version'] = pdf_rst['data_version'].astype(str)
    pdf_rst['version'] = datetime.now().strftime('%Y-%m-%d') + ' 00:00:00'
    if use_ahead:
        pdf_rst['comment'] = 'use_ahead'
    pdf_rst['version_diff'] = pd.to_numeric(pdf_rst['version_diff'], errors='coerce').astype('Int64')

    if task_name == 'by_day':
        doris_table = DORIS_SERVER_CONST.mysql_table_power_realtime
    else:
        doris_table = DORIS_SERVER_CONST.mysql_table_power_longterm_realtime
        pdf_rst['task_name'] = task_name
    res = cusdoris.api.stream_load_with_pdf(
    d=cusdoris.DorisInfo(doris_user=DORIS_SERVER_CONST.mysql_user,
                                doris_password=DORIS_SERVER_CONST.mysql_password,
                                doris_host=DORIS_SERVER_CONST.mysql_url,
                                doris_fe_port=31116,
                                doris_database=DORIS_SERVER_CONST.mysql_db,
                                doris_be_port=int(DORIS_SERVER_CONST.mysql_port)),
    table_name=doris_table,
    data=pdf_rst
    )
    logger.info(f"写入Doris:{doris_table}, 数据量:{pdf_rst.shape[0]}")


def submit_power_to_mongo_server(pdf_data: pd.DataFrame,
                                 col_list: List[str],
                                 use_ahead: bool = False,
                                 task_name: str='by_day'):
    new_cols = ['realtime_' + col for col in col_list]

    pdf_rst = pdf_data.copy()
    pred_col_list = [col+'_pred' for col in col_list]
    rename_cols = dict(zip(pred_col_list, new_cols))
    pdf_rst.rename(columns=rename_cols, inplace=True)
    pdf_rst['realtime_new_energy'] = pdf_rst['realtime_new_energy_wind'] + pdf_rst['realtime_new_energy_solar']

    pdf_rst['info_date'] = pdf_rst.apply(lambda x: str_timestamp_to_str(x['date_time'], input_format='%Y-%m-%d %H:%M:%S', output_format='%Y-%m-%d'), axis=1)
    pdf_rst['info_time'] = pdf_rst.apply(lambda x: str_timestamp_to_str(x['date_time'], input_format='%Y-%m-%d %H:%M:%S', output_format='%H:%M', 
                                                                        plus_minutes=15 if task_name == 'by_day' else 0), axis=1)
    pdf_rst['info_time'] = pdf_rst['info_time'].str.replace('00:00', '24:00')
    pdf_rst['date_time'] = pdf_rst['date_time'].astype(str)
    pdf_rst['update_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if 'version' in pdf_rst.columns:
        pdf_rst.rename(columns={'version': 'data_version'}, inplace=True)
    pdf_rst['version'] = datetime.now().strftime('%Y-%m-%d') + ' 00:00:00'

    if use_ahead:
        pdf_rst['comment'] = 'use_ahead'
    
    mongo_info = MongoInfo(mongo_url=MONGO_SERVE_CONST.mongo_url,
                           mongo_user=MONGO_SERVE_CONST.mongo_user,
                           mongo_password=MONGO_SERVE_CONST.mongo_password,
                           mongo_auth_db=MONGO_SERVE_CONST.mongo_auth_db,
                           mongo_db=MONGO_SERVE_CONST.mongo_db,
                           mongo_collection='')
    if task_name == 'by_day':
        mongo_info.mongo_collection = MONGO_SERVE_CONST.mongo_collection_power_realtime_by_day
    else:
        mongo_info.mongo_collection = MONGO_SERVE_CONST.mongo_collection_power_realtime_longterm
        pdf_rst['task_name'] = task_name
    
    mongo_rst = insert_mongo(mongo_info, pdf_rst.to_dict('records'))
    logger.info("写入Mongo: {}, 数据量: {}".format(mongo_info, mongo_rst))

    return pdf_rst


def submit_power_to_server(pdf_data: pd.DataFrame,
                           trade_type: str,
                           col_list: List[str],
                           use_ahead: bool = False,
                           to_doris: bool=True,
                           task_name: str='by_day'):
    if to_doris:
        submit_power_to_doris_server(
            pdf_data=pdf_data,
            col_list=col_list,
            use_ahead=use_ahead,
            task_name=task_name
        )
    else:
        submit_power_to_mongo_server(
            pdf_data=pdf_data,
            col_list=col_list,
            use_ahead=use_ahead,
            task_name=task_name
        )

