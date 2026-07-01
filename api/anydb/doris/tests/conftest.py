from doris.info import DorisInfo

import pytest


@pytest.fixture
def mysql_info():
    # your existing setup
    return DorisInfo.from_dict({
        "doris_user": "root",
        "doris_password": "example",
        "doris_host": "localhost",
        "doris_be_port": 3306,
        "doris_database": "meteor",
    })


@pytest.fixture
def doris_cluster_info():
    # your existing setup
    return DorisInfo.from_dict({
        "doris_user": "admin",
        "doris_host": "192.168.21.6",
        "doris_be_port": 32005,
        "doris_database": "meteor",
    })
