from pathlib import Path

from oss.oper import download_file, upload_file, delete_file, download_file_concurrency, upload_file_concurrency, delete_file_concurrency
from oss._types import OSSInfo

import pytest
import yaml


@pytest.fixture(scope="module")
def dianteng_conf():
    with open("tests/info.yml", "r", encoding='utf-8') as f:
        conf = yaml.safe_load(f)

    return conf["dianteng"]["oss"]


class TestOSSInfo:
    def test_init(self):
        i = OSSInfo(
            oss_url="https://oss-cn-hangzhou.aliyuncs.com",
            oss_access_key_id="123",
            oss_access_key_secret="abc"
        )

        assert i.oss_url == "https://oss-cn-hangzhou.aliyuncs.com"
        assert i.oss_access_key_id == "123"
        assert i.oss_access_key_secret == "abc"
        assert i.oss_bucket is None

    def test_from_dict(self):
        conf = {
            "oss_url": "https://oss-cn-hangzhou.aliyuncs.com",
            "oss_access_key_id": "123",
            "oss_access_key_secret": "abc",
            "oss_bucket": "bucket"
        }

        i = OSSInfo.from_dict(conf)

        assert i.oss_url == "https://oss-cn-hangzhou.aliyuncs.com"
        assert i.oss_access_key_id == "123"
        assert i.oss_access_key_secret == "abc"
        assert i.oss_bucket == "bucket"

    def test_from_dict_without_bucket(self):
        conf = {
            "oss_url": "https://oss-cn-hangzhou.aliyuncs.com",
            "oss_access_key_id": "123",
            "oss_access_key_secret": "abc",
        }

        i = OSSInfo.from_dict(conf)

        assert i.oss_bucket is None


class TestOperation:
    def test_upload_file(self, dianteng_conf):
        oss_info = OSSInfo.from_dict(dianteng_conf)
        upload_file(oss_info, "GFS/2024010319/gfs_A12024010319", "tests/data/gfs_A12024010319")

    def test_upload_file_concurrency(self, dianteng_conf):
        oss_info = OSSInfo.from_dict(dianteng_conf)

        local_files = [f.as_posix() for f in Path("tests/data").iterdir() if f.is_file()]

        upload_file_concurrency(oss_info, "GFS/2024010319", local_files)
