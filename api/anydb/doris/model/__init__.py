"""
用于气象结构化数据数据表

其中包括但不限于

- 2169个中国气象台站逐小时观测数据
- ERA5再分析逐小时网格数据
- CMS大气污染物逐小时5日预测网格数据
- GFS15日确定性预报网格数据
- ECMWF HRES10日确定性预报网格数据
- ECMWF ENS15日集合预报（一条）网格数据
- ECMWF ENS Extend 16-46日集合预报（一条）网格数据
- 所有网格原始文件与Minio object key的元信息表
"""

from .ens import ads_terraqt_ecmwf_ens_table


__all__ = [
    'ads_terraqt_ecmwf_ens_table',
]