from dataclasses import dataclass
from ..anydb.miniolib import MinioInfo
import numpy as np

# 气象要素列表, 用于指定供需预测模型的特征
WEATHER_FEATURE_LIST = [
    'temperature_2m', 'apparent_temperature', 'humidity', 'precipitation_instant',
    'cloudcover', 'windspeed_10m', 'windspeed_100m', 'windspeed_200m','winddir_10m', \
    'winddir_100m','winddir_200m', 'shortwave_radiation_instant','pressure_surface', \
    # 'windspeed_10m_log','windspeed_100m_log','windspeed_200m_log'
]


WEATHER_FEATURE_SEA_LIST = ['windspeed_100m_sea']
# WEATHER_FEATURE_SEA_LIST = ['temperature','solartotal','pressure','totalcloud','humidity', 'windspeed','windspeed_100','rainfall']


WIND_WEATHER_FEATURE_LIST = [
    'windspeed_100m','windspeed_200m']
# WIND_WEATHER_FEATURE_LIST = [
#     'windspeed_10m', 'windspeed_100m','windspeed_200m', 'pressure_surface']#'winddir_10m','winddir_100m','winddir_200m',
# WIND_WEATHER_FEATURE_LIST = WEATHER_FEATURE_LIST


SOLAR_WEATHER_FEATURE_LIST = [
    'temperature_2m',  
    'apparent_temperature',
    'cloudcover', 'humidity', 'shortwave_radiation_instant',
]
# SOLAR_WEATHER_FEATURE_LIST = WEATHER_FEATURE_LIST


REALTIME_FEATURE_LIST = ['provincial_load_ahead', 'tie_line_ahead', 'new_energy_total_ahead',
                         'new_energy_wind_ahead', 'new_energy_solar_ahead', 'non_market_ahead']

JIANGSU_CITY = ['南京市','无锡市','徐州市','常州市','苏州市','南通市','连云港市','淮安市','盐城市','扬州市','镇江市','泰州市','宿迁市']  
JIANGSU_CITY2_WIND = ['盐城市','徐州市','南通市','连云港市','泰州市','宿迁市'] 

# JIANGSU_CITY2_WIND = JIANGSU_CITY
# JIANGSU_CITY2_WIND = ['盐城市','南通市','淮安市','扬州市','泰州市']

# JIANGSU_CITY2_SOLAR = ['连云港市','淮安市','扬州市','宿迁市','镇江市','常州市'] 
# JIANGSU_CITY2_SOLAR = ['无锡市','常州市','苏州市','南通市','连云港市','淮安市','盐城市','扬州市','镇江市','泰州市','宿迁市'] 
JIANGSU_CITY2_SOLAR = JIANGSU_CITY

JIANGSU_CITY_ID = np.arange(2,109)
JIANGSU_SEA = np.arange(113,149)

JIANGSU_SOLAR_CAPACITY = {
    2025: {
        7: 3883.167,
    }
}
JIANGSU_WIND_CAPACITY = {
    2025: {
        7: 22754.28,
    }
}


def get_latest_province_capacity(label_type: str) -> float:
    """Return the latest configured Jiangsu province installed capacity (MW)."""
    capacity_map = JIANGSU_WIND_CAPACITY if 'wind' in label_type else JIANGSU_SOLAR_CAPACITY if 'solar' in label_type else {}
    latest_capacity = 0.0
    for year in sorted(capacity_map):
        for month in sorted(capacity_map[year]):
            value = capacity_map[year][month]
            if value is not None and value > 0:
                latest_capacity = float(value)
    return latest_capacity

JIANGSU_CITY_WIND_CAPACITY = {
    '盐城市': 1121,   # 1121
    '南通市': 550,
    '连云港市': 350,
    '徐州市': 60,
    # '淮安市': 40,
    '宿迁市': 40,
    # '扬州市': 20,
    '泰州市': 20,
    # '南京市': 10,
    # '常州市': 10,
    # '无锡市': 10,
    # '苏州市': 10,
    # '镇江市': 10,
}

JIANGSU_CITY_SOLAR_CAPACITY = {
    '徐州市': 900,
    '连云港市': 700,
    '淮安市': 600,
    '盐城市': 1064,
    '宿迁市': 650,
    '扬州市': 500,
    '镇江市': 350,
    '常州市': 350,
    '无锡市': 350,
    '苏州市': 350,
    '南通市': 800,
    '南京市': 350,
    '泰州市': 500,
}

ANHUI_SOLAR_CAPACITY = {
    2025: {
        7: 13531.65,
    }
}
ANHUI_WIND_CAPACITY = {
    2025: {
        7: 9622.25,
    }
}


METEO_SOURCE = {
    'HRES': 'ads_terraqt_ecmwf_hres_by_station_v',##ods_terraqt_ecmwf_hres_by_station_v,ads_terraqt_ecmwf_hres_by_station_v
    'ENS': 'ads_terraqt_ecmwf_ens_by_station_v',
    'ENS_EXTEND': 'ads_terraqt_ecmwf_ens_extend_by_station_v',
    'HRES_W': {
        # 'wind': 'ads_hres_jiangsu_weighted_wind_24p',
        # 'solar': 'ads_hres_jiangsu_weighted_solar_24p'
        
        'wind': 'ads_hres_anhui_weighted_wind_24p',
        'solar': 'ads_hres_anhui_weighted_solar_24p'

    }
}


METEO_SEA_SOURCE = {
    'HRES': 'hs_ies_sca_tra_weather_b',
    'ENS': 'hs_ies_sca_tra_weather_b',

}

TEN_DAYS_PLUS_HOUR = 160 # 24*7-8
MONTH_PLUS_HOUR = 232 # 24*10-8


@dataclass
class HuaweicloudMongoConstants:
    """
    华为云Mongo数据库-交易中心数据-只读
    """
    mongo_url = '1.94.208.154:8635'
    mongo_user = 'algorithm'
    mongo_password = 'algorithm123'
    mongo_auth_db = 'yn_ams_dw'
    mongo_db = 'yn_ams_dw'

    mongo_collection_provincial_load_ahead = 'dwd_jiangsu_inter_provincial_dispatch_load_day_ahead'
    mongo_collection_tie_line_ahead = 'dwd_jiangsu_call_wire_plan_day_ahead'
    mongo_collection_non_market_ahead = 'dwd_jiangsu_non_market_output_day_ahead'
    mongo_collection_new_energy_ahead = 'dwd_jiangsu_new_energy_load_day_ahead'
    mongo_collection_new_energy_solar_ahead = 'dwd_jiangsu_new_energy_load_luce_day_ahead'
    mongo_collection_new_energy_wind_ahead = 'dwd_jiangsu_new_energy_load_wind_day_ahead'
    

    # mongo_collection_provincial_load_realtime = 'dwd_jiangsu_inter_provincial_dispatch_load_real_time'
    # mongo_collection_tie_line_realtime = 'dwd_jiangsu_call_wire_plan_real_time'
    # mongo_collection_non_market_realtime = 'dwd_jiangsu_non_market_output_real_time'
    # mongo_collection_new_energy_realtime = 'dwd_jiangsu_new_energy_load_real_time'
    # mongo_collection_new_energy_solar_realtime = 'dwd_jiangsu_new_energy_load_luce_real_time'
    # mongo_collection_new_energy_wind_realtime = 'dwd_jiangsu_new_energy_load_wind_real_time'


@dataclass
class MengxiServerMongoTestConstants:
    """
    Mongo数据库-测试库-读写权限
    """
    mongo_url = '192.168.5.13:27017'
    mongo_user = 'algorithm'
    mongo_password = 'algorithm123'
    mongo_auth_db = 'yn_model_predicts'
    mongo_db = 'yn_model_predicts'

    mongo_collection_power_realtime_by_day = 'ods_jiangsu_power_predictions_realtime'
    mongo_collection_power_realtime_longterm = 'ods_jiangsu_power_longterm_predictions_realtime'


@dataclass
class MengxiDorisConstants:
    """
    本地Doris数据库-气象-只读
    """
    mysql_url = '192.168.21.2'
    mysql_user = 'mengxi'
    mysql_password = 'wgkAtjzYPpkqY8C5'
    mysql_port = '31363'
    mysql_db = 'mengxi_test'

    mysql_table_power_realtime = 'ods_power_predictions_realtime_opt'
    mysql_table_power_longterm_realtime = 'ods_power_longterm_predictions_realtime_opt'
    power_realtime_online = 'ods_realtime_power_v_96p'


@dataclass
class MeteoDorisConstants:
    """
    本地Doris数据库-气象-只读
    """
    mysql_url = '192.168.21.2'
    mysql_user = 'meteo_readonly'
    mysql_password = '5hfLpOz4'
    mysql_port = '31363'
    mysql_db = 'meteo'

@dataclass
class MeteoTestDorisConstants:
    """
    本地Doris数据库-气象-只读
    """
    mysql_url = '192.168.21.2'
    mysql_user = 'meteo_readonly'
    mysql_password = '5hfLpOz4'
    mysql_port = '31363'
    mysql_port_be = '31363'
    mysql_port_fe = '31116'
    mysql_db = 'meteo_test'


@dataclass
class MengxiServerMongoTestConstants:
    """
    Mongo数据库-测试库-读写权限
    """
    mongo_url = '192.168.5.13:27017'
    mongo_user = 'algorithm'
    mongo_password = 'algorithm123'
    mongo_auth_db = 'yn_model_predicts'
    mongo_db = 'yn_model_predicts'

    mongo_collection_power_realtime_by_day = 'ods_jiangsu_power_predictions_realtime'
    mongo_collection_power_realtime_longterm = 'ods_jiangsu_power_longterm_predictions_realtime'




@dataclass
class MeteoSqlConstants:
    """
    本地Doris数据库-气象-只读
    """
    mysql_url = 'rm-2zeo6b82mxpp29909so.mysql.rds.aliyuncs.com'
    mysql_user = 'zwlh_algo'
    mysql_password = 'jeC6b7ZUN6MVrXYa'
    mysql_port = '3306'
    mysql_db = 'niescloud_emad_gn'
    ##329001-329037,113-149

MINIO_SERVER = MinioInfo(
    minio_url='minio-api.aienertech.cn',
    minio_access_key='admin',
    minio_secret_key='k2whP9awwZpuNHLZ',
    bucket_name='meteor',
    minio_secure=True
)


MONGO_CONST = HuaweicloudMongoConstants()
MONGO_SERVE_CONST = MengxiServerMongoTestConstants()
# MONGO_SERVE_CONST = HuaweicloudMongoServeConstants()
METEO_DORIS_CONST = MeteoDorisConstants()
METEO_TEST_DORIS_CONST = MeteoTestDorisConstants()
METEO_TEST_WRT_DORIS_CONST = MeteoTestDorisConstants()
DORIS_SERVER_CONST = MengxiDorisConstants()

##海上风机
SEA_WIND_CONST = MeteoSqlConstants()
##江苏功率预测相关配置
data_infos = '/data00/chenjiajun/20250711/daily_test/jiangsu_power_cronjob/configs/Data_Info.yaml'
tag_name = "jiangsu_realtime_new_energy"
sea_power = '/data00/chenjiajun/20250711/daily_test/jiangsu_power_sea/output/pdf_rst.csv'
