from urllib3 import HTTPHeaderDict
from pathlib import Path

from miniolib._types import MinioInfo
from miniolib.oper import (upload_file,
                           bucket_exists,
                           list_objects,
                           upload_folder,
                           delete_prefix,
                           get_object,
                           get_folder,
                           )

import pytest
import yaml


@pytest.fixture(scope="module")
def conf():
    with open("miniolib/test/info.yml", "r", encoding='utf-8') as f:
        conf = yaml.safe_load(f)

    return conf["minio"]["shengbin"]


@pytest.fixture(scope="module")
def client(conf):
    return MinioInfo.from_dict(conf)


class TestMinioInfo:
    def test_init(self):
        i = MinioInfo(
            minio_url="https://oss-cn-hangzhou.aliyuncs.com",
            minio_access_key="123",
            minio_secret_key="abc",
            bucket_name="bucket"
        )

        assert i.minio_url == "https://oss-cn-hangzhou.aliyuncs.com"
        assert i.minio_access_key == "123"
        assert i.minio_secret_key == "abc"
        assert i.bucket_name == "bucket"

    def test_from_dict(self, conf):
        i = MinioInfo.from_dict(conf)

        assert i.minio_url == conf['minio_url']
        assert i.minio_access_key == conf['minio_access_key']


class TestOperation:
    def test_bucket_exists(self, client):
        res = bucket_exists(client)

        assert res is True

    def test_upload_file(self, client):
        f = 'miniolib/test/data/2024022000/A12024022023'

        res = upload_file(client,
                          f,
                          "test/terraqt/ecmwf_ens/A12024022023")

        assert isinstance(res, HTTPHeaderDict)

    def test_upload_folder(self, client):
        folder = "miniolib/test/data/2024022000"

        res = upload_folder(client, folder, "test/terraqt/ecmwf_ens/2024022000")

        assert isinstance(res, list)

    def test_list_objects(self, client):
        res = list_objects(client, "test/terraqt/ecmwf_ens")

        assert len(res) > 0

    def test_get_object(self, client):
        res = get_object(client, "test/terraqt/ecmwf_ens/A12024022023", "miniolib/test/data/download/A12024022023")

        assert Path(res).exists() and Path(res).is_file()

    def test_get_folder(self, client):
        res = get_folder(client, "test/terraqt/ecmwf_ens/2024022000", "miniolib/test/data/download/2024022000")

        for r in res:
            assert Path(r).exists() and Path(r).is_file()

    def test_delete_prefix(self, client):
        delete_prefix(client, "test/")

        res = list_objects(client, "test/")

        assert len(res) == 0
