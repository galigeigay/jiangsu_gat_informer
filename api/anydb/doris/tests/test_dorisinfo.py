from doris.info import DorisInfo

import pytest


class TestDorisInfo:
    def test_init(self):
        d = {
            "doris_user": "test",
            "doris_password": "test",
            "doris_host": "test",
            "doris_be_port": 3306,
            "doris_database": "test"
        }

        doris = DorisInfo.from_dict(d)

        assert doris.doris_user == "test"
        assert doris.doris_password == "test"
        assert doris.doris_host == "test"
        assert doris.doris_be_port == 3306
        assert doris.doris_database == "test"

        assert doris.doris_url == "doris+pymysql://test:test@test:3306/test"

    def test_stream_load_url(self):
        d = {
            "doris_user": "test",
            "doris_password": "test",
            "doris_host": "test",
            "doris_fe_port": 3306,
            "doris_database": "test"
        }

        doris = DorisInfo.from_dict(d)

        table_name = 'test_stream_load'

        assert doris.stream_load_url(table_name) == 'http://test:3306/api/test/test_stream_load/_stream_load', \
            "stream_load_url is not correct"

    def test_init_without_passwd(self):
        d = {
            "doris_user": "test",
            "doris_host": "test",
            "doris_be_port": 3306,
            "doris_database": "test"
        }

        doris = DorisInfo.from_dict(d)

        assert doris.doris_user == "test"
        assert doris.doris_password is None
        assert doris.doris_host == "test"
        assert doris.doris_be_port == 3306
        assert doris.doris_database == "test"

        assert doris.doris_url == "doris+pymysql://test@test:3306/test"
