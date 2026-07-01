"""
ECMWF 15日集合预报模型
"""
from ..api import metadata_obj

from sqlalchemy import String, Column, Table, MetaData, DateTime
import sqlalchemy as sa
from sqlalchemy_doris import datatype
from sqlalchemy_doris import HASH, RANGE


# 15日集合预报模型ads应用数据表格
ads_terraqt_ecmwf_ens_table = Table(
    'ods_terraqt_ecmwf_ens_by_station_v',
    metadata_obj,
    Column('station_code', datatype.String(5)),
    Column('province', datatype.String(20)),
    Column('city', datatype.String(40)),
    Column('district', datatype.String(40)),
    Column('coordinate', datatype.String(40)),
    Column('date_time', sa.DateTime),
    Column('version', sa.DATETIME),
    Column('temperature_2m', datatype.DOUBLE, nullable=True),
    Column('dewpoint_2m', datatype.DOUBLE, nullable=True),
    Column('pressure_surface', datatype.DOUBLE, nullable=True),
    Column('apparent_temperature', datatype.DOUBLE, nullable=True),
    Column('humidity', datatype.DOUBLE, nullable=True),
    Column('cloudcover', datatype.DOUBLE, nullable=True),
    Column('cloudcover_low', datatype.DOUBLE, nullable=True),
    Column('cloudcover_medium', datatype.DOUBLE, nullable=True),
    Column('cloudcover_high', datatype.DOUBLE, nullable=True),
    Column('windspeed_10m', datatype.DOUBLE, nullable=True),
    Column('windspeed_100m', datatype.DOUBLE, nullable=True),
    Column('windspeed_200m', datatype.DOUBLE, nullable=True),
    Column('precipitation_instant', datatype.DOUBLE, nullable=True),
    Column('shortwave_radiation_instant', datatype.DOUBLE, nullable=True),
    Column('snow_fall_instant', datatype.DOUBLE, nullable=True),
    Column('total_precipitation', datatype.DOUBLE, nullable=True),
    Column('shortwave_radiation', datatype.DOUBLE, nullable=True),
    Column('snow_fall', datatype.DOUBLE, nullable=True, comment='累计降雪量， 单位米'),
    doris_unique_key=('station_code', 'date_time', 'version',),
    doris_partition_by=RANGE('version'),
    doris_distributed_by=HASH('version'),
    doris_properties={"replication_allocation": "tag.location.default: 1"}
)
