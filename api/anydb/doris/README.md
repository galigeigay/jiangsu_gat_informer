## doris

[Apache Doris](https://doris.apache.org/zh-CN/docs/get-starting/quick-start/) 是一个基于 MPP 架构的高性能、实时的分析型数据库，以极速易用的特点被人们所熟知，仅需亚秒级响应时间即可返回海量数据下的查询结果，不仅可以支持高并发的点查询场景，也能支持高吞吐的复杂分析场景，这个简短的指南将告诉你如何下载 Doris 最新稳定版本，在单节点上安装并运行它，包括创建数据库、数据表、导入数据及查询等。

此sdk提供基本易能科技所有业务、算法相关的doris基本气象表格的灵活配置连接、写入、查询等功能。其底层由sqlalchemy与pymysql的驱动，可以方便的进行sql查询。

### 使用方法

#### 简单查询
```python
from datetime import datetime

from doris.model import ads_terraqt_ecmwf_ens_table
from doris import DorisInfo, new_engine

import pandas as pd

# 根据配置文件创建连接信息类
# 连接信息字典，可自己设定或者读取yaml文件等
connect_info = {}

doris_info = DorisInfo.from_dict(**connect_info)

# 创建连接引擎
engine = new_engine(doris_info)

# 拉去ads_terraqt_ecmwf_ens_table表格中version为 datetime(2024, 1, 29, 0) 的所有山西省预测的数据
stmt = ads_terraqt_ecmwf_ens_table.select().\
    where(ads_terraqt_ecmwf_ens_table.c.version == datetime(2024, 1, 29, 0)).\
    where(ads_terraqt_ecmwf_ens_table.c.province == '山西省').\
    order_by(ads_terraqt_ecmwf_ens_table.c.station_code).\
    order_by(ads_terraqt_ecmwf_ens_table.c.date_time.desc())

# 读取数据
df = pd.read_sql(stmt, engine)
```
