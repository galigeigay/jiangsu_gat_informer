import urllib

# from .. anydb import doris as cusdoris
import anydb.doris as cusdoris
import pandas as pd
import sqlalchemy as sql

from dataclasses import dataclass


@dataclass
class MysqlInfo:
    mysql_url: str
    mysql_user: str
    mysql_password: str
    mysql_port: int
    mysql_db: str


def read_mysql(mysql_info: MysqlInfo, query=""):
    """
    在MySQL执行查询命令query
    """
    mysql_url = "mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(mysql_info.mysql_user, urllib.parse.quote(mysql_info.mysql_password), mysql_info.mysql_url, mysql_info.mysql_port, mysql_info.mysql_db)
    engine = sql.create_engine(mysql_url)
    with engine.connect() as conn:
        df = pd.read_sql(
            sql=query,
            con=conn.connection
        )
    return df


def read_mysql2(mysql_info: MysqlInfo, query=""):
    doris_info = cusdoris.DorisInfo(doris_user=mysql_info.mysql_user,
                                    doris_password=mysql_info.mysql_password,
                                    doris_host=mysql_info.mysql_url,
                                    doris_fe_port=int(mysql_info.mysql_port),
                                    doris_database=mysql_info.mysql_db,
                                    doris_be_port=int(mysql_info.mysql_port))
    engine = cusdoris.api.create_new_connection(doris_info, connect_timeout=300)
    df = pd.read_sql(query, engine)
    return df


def upload_mysql(mysql_info: MysqlInfo, pdf: pd.DataFrame, table_name: str):
    mysql_url = "mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(mysql_info.mysql_user, urllib.parse.quote(mysql_info.mysql_password), mysql_info.mysql_url, mysql_info.mysql_port, mysql_info.mysql_db)
    engine = sql.create_engine(mysql_url)

    with engine.connect() as conn:
        pdf.to_sql(table_name, con=conn.connection, if_exists='append', index=False)