from pathlib import Path
from datetime import datetime, timedelta

from obsdb._types import ObsInfo
from obsdb.oper import list_objects, download_file, download_files_concurrency
from obsdb import errors

import pytest
import yaml


@pytest.fixture(scope="module")
def obs_info():
    with open("tests/info.yml", "r", encoding='utf-8') as f:
        conf = yaml.safe_load(f)

    return ObsInfo.from_dict(conf["obs"])


class TestObsInfo:
    def test_init(self):
        i = ObsInfo(
            server="https://obs.cn-north-4.myhuaweicloud.com",
            access_key="123",
            secret_key="abc",
            bucket_name="bucket"
        )

        assert i.server == "https://obs.cn-north-4.myhuaweicloud.com"
        assert i.access_key == "123"
        assert i.secret_key == "abc"
        assert i.bucket_name == "bucket"

    def test_from_dict(self):
        conf = {
            "server": "https://obs.cn-north-4.myhuaweicloud.com",
            "access_key": "123",
            "secret_key": "abc",
            "bucket_name": "bucket"
        }

        i = ObsInfo.from_dict(conf)

        assert i.server == "https://obs.cn-north-4.myhuaweicloud.com"
        assert i.access_key == "123"
        assert i.secret_key == "abc"
        assert i.bucket_name == "bucket"


class TestOperation:
    def test_list_object(self, obs_info):
        version = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        version -= timedelta(days=1)

        prefix = f'A1/ENS_EXTENDED/{version.strftime("%Y%m%d%H")}'
        keys = list_objects(obs_info, prefix)
        assert len(keys) > 0, f"list objects failed @ prefix: {prefix}"

        prefix = f'error_path'
        with pytest.raises(errors.ObsObjectListEmtpy):
            list_objects(obs_info, prefix)

    def test_download_file(self, obs_info):
        version = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        version -= timedelta(days=1)

        prefix = f'A2/ENS_EXTENDED/{version.strftime("%Y%m%d%H")}'
        keys = list_objects(obs_info, prefix)

        target_path = download_file(obs_info, keys[0], "./tests/ens_extend")

        assert target_path.exists()
        target_path.unlink()

    def test_download_concurrency(self, obs_info):
        version = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        version -= timedelta(days=1)

        prefix = f'A2/ENS_EXTENDED/{version.strftime("%Y%m%d%H")}'
        keys = list_objects(obs_info, prefix)

        target_path = download_files_concurrency(obs_info, keys, f"./tests/{version.strftime('%Y%m%d%H')}")

        for f in target_path:
            assert f.exists()
            f.unlink()

