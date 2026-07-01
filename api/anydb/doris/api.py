from typing import List, Tuple, Any
import base64

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

from .info import DorisInfo
from . import errors

import MySQLdb


def create_new_connection(d: DorisInfo, connect_timeout: int = 3) -> MySQLdb.Connection:
    """
    创建一个新的 Doris 连接

    :param d: DorisInfo 对象
    :param connect_timeout: 连接超时时间, 单位秒

    :return: MySQLdb.Connection 对象
    """

    # 获得一个新的kwargs字典， 并删除为None的属性
    _input = vars(d)

    init_args = {
        'host': 'doris_host',
        'user': 'doris_user',
        'passwd': 'doris_password',
        'db': 'doris_database',
        'port': 'doris_be_port'
    }

    kwargs = {
        'charset': 'utf8',
        'connect_timeout': connect_timeout
    }
    for k, v in init_args.items():
        if hasattr(d, v) and getattr(d, v) is not None:
            kwargs[k] = _input[v]

    return MySQLdb.connect(
        **kwargs
    )


def check_table_exist(d: DorisInfo, table_name: str) -> bool:
    """
    检查表是否存在

    :param d: DorisInfo 对象
    :param table_name: 表名

    :return: 表是否存在
    """
    conn = create_new_connection(d)
    cursor = conn.cursor()
    cursor.execute(f"show tables like '{table_name}'")
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return len(result) > 0


def create_table(d: DorisInfo,
                 create_table_sql: str,
                 exist_ok: bool = False):
    """
    创建表

    :param d: DorisInfo 对象
    :param create_table_sql: 创建表的 SQL 语句
    :param exist_ok: 如果表已经存在是否忽略

    :return: None
    """
    conn = create_new_connection(d)
    cursor = conn.cursor()
    try:
        cursor.execute(create_table_sql)
    except MySQLdb.OperationalError as e:

        # MySQLbb error code 1050 means table already exists
        if e.args[0] in [1050, 1105] and exist_ok:
            ...
        else:
            raise e

    cursor.close()
    conn.close()


def drop_table(d: DorisInfo, table_name: str):
    """
    删除表

    :param d: DorisInfo 对象
    :param table_name: 表名

    :return: None
    """
    conn = create_new_connection(d)
    cursor = conn.cursor()
    cursor.execute(f"drop table if exists {table_name}")
    conn.commit()
    cursor.close()
    conn.close()


def query(d: DorisInfo, sql: str) -> list:
    """
    查询数据

    :param d: DorisInfo 对象
    :param sql: 查询 SQL 语句

    :return: 查询结果
    """
    conn = create_new_connection(d)
    cursor = conn.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result


def insert_one(d: DorisInfo, table_name: str, data: dict):
    """
    插入一条数据

    :param d: DorisInfo 对象
    :param table_name: 表名
    :param data: 数据

    :return: None
    """
    conn = create_new_connection(d)
    cursor = conn.cursor()
    columns = ', '.join(data.keys())

    _sql = f"INSERT INTO {table_name} ({columns}) VALUES ({','.join(['%s' for _ in range(len(data.keys()))])})"

    try:
        cursor.execute(_sql, tuple(data.values()))
    except MySQLdb.ProgrammingError as e:
        # MySQLbb error code 1146 means table does not exist
        if e.args[0] == 1146:
            raise errors.TableNotExist(f"Table {table_name} does not exist")
        else:
            raise e
    except MySQLdb.OperationalError as e:
        if e.args[0] == 1105:
            raise errors.TableNotExist(f"Table {table_name} does not exist")
        else:
            raise e

    conn.commit()

    cursor.close()
    conn.close()


def insert_many(d: DorisInfo, table_name: str, data: list):
    """
    插入多条数据

    :param d: DorisInfo 对象
    :param table_name: 表名
    :param data: 数据

    :return: None
    """
    conn = create_new_connection(d)
    cursor = conn.cursor()
    columns = ', '.join(data[0].keys())

    # convert into list of tuples
    values: List[Tuple[Any]] = [tuple(record.values()) for record in data]

    _sql = f"INSERT INTO {table_name} ({columns}) VALUES ({','.join(['%s' for _ in range(len(data[0].keys()))])})"

    try:
        cursor.executemany(_sql, values)
    except MySQLdb.ProgrammingError as e:
        # MySQLbb error code 1146 means table does not exist
        if e.args[0] == 1146:
            raise errors.TableNotExist(f"Table {table_name} does not exist")
        else:
            raise e

    conn.commit()

    cursor.close()
    conn.close()



def stream_load_with_pdf(d: DorisInfo,
                         table_name: str,
                         data: pd.DataFrame,
                         load_params: dict) -> requests.Response:
    """
    使用 Pandas DataFrame 进行流式加载, 一般当Dataframe特别大的时候使用此方法

    stream load是doris server暴露的一个接口，可以通过http的PUT请求的方式将数据发送到服务端，然后由doris进行数据刷入
    关于stream load更为详细请参考: https://doris.apache.org/blog/principle-of-Doris-Stream-Load/
    """
    json_data = data.to_json(orient='records')

    # Dynamically generate columns and jsonpaths from DataFrame columns
    columns = ','.join(data.columns)
    jsonpaths = '[' + ', '.join(f'"$.{col}"' for col in data.columns) + ']'

    header = {
        "Authorization": __encode_basic_auth(d.doris_user, d.doris_password),
        "Expect": "100-continue",  # Use for handling larger datasets
        "format": "json",  # Specify format of the stream load
        "strip_outer_array": "true",  # Strip the outer array if your JSON root is an array
        "read_json_by_line": "true",  # Read JSON by line
        "jsonpaths": jsonpaths,  # JSON paths mapping to Doris table columns
        "columns": columns  # Map DataFrame columns to Doris table columns
    }

    # 添加加载参数到请求头（如合并策略）
    if load_params:
        header.update(load_params)

    resp = requests.put(d.stream_load_url(table_name),
                        headers=header,
                        data=json_data)
    print(resp)
    
    if resp.status_code != 200:
        raise errors.DorisStreamLoadError(f"Stream load failed with status code {resp.status_code}: {resp.text}")

    if resp.json()['Status'] != 'Success' and resp.json()['Message'] != 'OK':
        raise errors.DorisStreamLoadError(f"Stream load failed with status {resp.json()['Status']}: {resp.text}")

    return resp


# def stream_load_with_pdf(d: DorisInfo,
#                          table_name: str,
#                          data: pd.DataFrame) -> requests.Response:
#     """
#     使用 Pandas DataFrame 进行流式加载, 一般当Dataframe特别大的时候使用此方法

#     stream load是doris server暴露的一个接口，可以通过http的PUT请求的方式将数据发送到服务端，然后由doris进行数据刷入
#     关于stream load更为详细请参考: https://doris.apache.org/blog/principle-of-Doris-Stream-Load/
#     """
#     json_data = data.to_json(orient='records')

#     # Dynamically generate columns and jsonpaths from DataFrame columns
#     columns = ','.join(data.columns)
#     jsonpaths = '[' + ', '.join(f'"$.{col}"' for col in data.columns) + ']'

#     header = {
#         "Authorization": __encode_basic_auth(d.doris_user, d.doris_password),
#         "Expect": "100-continue",  # Use for handling larger datasets
#         "format": "json",  # Specify format of the stream load
#         "strip_outer_array": "true",  # Strip the outer array if your JSON root is an array
#         "read_json_by_line": "true",  # Read JSON by line
#         "jsonpaths": jsonpaths,  # JSON paths mapping to Doris table columns
#         "columns": columns  # Map DataFrame columns to Doris table columns
#     }

#     resp = requests.put(d.stream_load_url(table_name),
#                         headers=header,
#                         data=json_data)

#     if resp.status_code != 200:
#         raise errors.DorisStreamLoadError(f"Stream load failed with status code {resp.status_code}: {resp.text}")

#     if resp.json()['Status'] != 'Success' and resp.json()['Message'] != 'OK':
#         raise errors.DorisStreamLoadError(f"Stream load failed with status {resp.json()['Status']}: {resp.text}")

#     return resp


def __encode_basic_auth(username: str, password: str = None):
    # Encode 'username:' since the password is not required or is empty
    auth_str = f"{username}:{password if password is not None else ''}"
    auth_bytes = auth_str.encode('utf-8')  # Convert string to bytes
    encoded_auth = base64.b64encode(auth_bytes).decode('utf-8')  # Encode to Base64 and convert back to string
    return f"Basic {encoded_auth}"
