from datetime import datetime

import MySQLdb

from doris import errors
from doris import api
from doris.info import DorisInfo
import mongo

import pandas as pd
import numpy as np
import pytest


doris_cluster_info = DorisInfo.from_dict({
        "doris_user": "admin",
        "doris_host": "192.168.21.2",
        "doris_be_port": 31363,
        "doris_fe_port": 31116,
        "doris_database": "meteor",
    })


mysql_info = DorisInfo.from_dict({
        "doris_user": "root",
        "doris_password": "example",
        "doris_host": "localhost",
        "doris_be_port": 3306,
        "doris_database": "meteor",
    })


@pytest.mark.parametrize('doris_config', [doris_cluster_info])
class TestConnection:
    def test_create_new_connection(self, doris_config):
        conn = api.create_new_connection(doris_config)
        assert conn is not None
        conn.close()

    def test_create_new_connection_with_timeout(self, doris_config):
        _doris_cluster_info = doris_config.copy(doris_be_port=11111)

        with pytest.raises(MySQLdb.OperationalError):
            conn = api.create_new_connection(_doris_cluster_info, connect_timeout=1)
            conn.close()


@pytest.mark.parametrize('doris_config', [doris_cluster_info])
class TestApi:
    def test_create_table(self, doris_config):
        if doris_config.doris_host == 'localhost':
            with open('doris/sql/ens.sql', encoding='utf-8') as f:
                create_table_sql = f.read()
        elif doris_config.doris_host == doris_cluster_info.doris_host:
            with open('doris/tests/sql/ens_doris.sql', encoding='utf-8') as f:
                create_table_sql = f.read()

        api.create_table(doris_config, create_table_sql, exist_ok=True)

    def test_check_table_exist(self, doris_config):
        assert api.check_table_exist(doris_config, "test") is False
        assert api.check_table_exist(doris_config, "ads_terraqt_ecmwf_ens_by_station_v") is True

    def test_insert_one(self, doris_config):
        data = {
            "version": datetime(2021, 1, 1, 0),
            "date_time": datetime(2021, 1, 1, 1),
            "station_code": 53778,
            "province": "广东省",
            "city": "广州市",
            "district": "白云区",
            "coordinates": "113.33,23.19",
            "temperature_2m": 20.0,
            "humidity": 78.9,
        }
        with pytest.raises(errors.TableNotExist):
            api.insert_one(doris_config, "not_exist_table", data)

        api.insert_one(doris_config, "ads_terraqt_ecmwf_ens_by_station_v", data)
        result = api.query(doris_config,
                           "select * from ads_terraqt_ecmwf_ens_by_station_v where station_code = 53778")

        assert len(result) == 1

    def test_insert_many(self, doris_config):
        df = pd.read_csv("doris/tests/sql/ens.csv")

        # replace all [] to None
        df = df.replace({'[]': None})
        df = df.replace({np.nan: None})
        df = df.drop(columns=['_id'])

        api.insert_many(doris_config, "ads_terraqt_ecmwf_ens_by_station_v", df.to_dict(orient='records'))

        result = api.query(doris_config, "select * from ads_terraqt_ecmwf_ens_by_station_v where province = '北京市'")
        assert len(result) > 500

    @pytest.mark.skip
    def test_drop_table(self, doris_config):
        api.drop_table(doris_config, "ads_terraqt_ecmwf_ens_by_station_v")
        assert api.check_table_exist(doris_config, "ads_terraqt_ecmwf_ens_by_station_v") is False


@pytest.mark.load
class TestLoad:
    """
    只对Doris的各类Load操作进行测试
    """
    def test_stream_load_with_pdf_from_mongo(self):
        doris_config = doris_cluster_info

        mongo_info = mongo.MongoInfo.from_dict({
            "mongo_url": "192.168.5.13:27017",
            "mongo_user": "algorithm",
            "mongo_pass": "algorithm123",
            "mongo_auth_source": "yn_model_predicts",
            "mongo_database": "yn_model_test",
        }).set_collection("ads_terraqt_ecmwf_ens_extended_hourly_by_station_v")

        docs = mongo.query(mongo_info,
                           filter={"version": "2024-05-16 08:00:00", "province": "山西省"},
                           projection={"_id": 0}
                           )

        df = pd.DataFrame(list(docs))

        df = df.replace({'[]': None})
        df = df.replace({np.nan: None})

        # station_code convert to int
        df['station_code'] = df['station_code'].astype(int)

        response = api.stream_load_with_pdf(doris_config, "ads_terraqt_ecmwf_ens_by_station_v", df)
        assert response.status_code == 200

        resp = response.json()

        assert "[INTERNAL_ERROR]" not in resp.get('Message')
        assert resp.get('Status') == 'Success'

    def test_stream_load_with_pdf_from_csv(self):
        doris_config = doris_cluster_info

        df = pd.read_csv("doris/tests/sql/ens.csv")

        # replace all [] to None
        df = df.replace({'[]': None})
        df = df.replace({np.nan: None})
        df = df.drop(columns=['_id'])

        response = api.stream_load_with_pdf(doris_config, "ads_terraqt_ecmwf_ens_by_station_v", df)
        assert response.status_code == 200

        resp = response.json()

        assert "[INTERNAL_ERROR]" not in resp.get('Message')
        assert resp.get('NumberTotalRows') > 0
        assert resp.get('Status') == 'Success'
