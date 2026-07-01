from typing import Any

from mongo._types import MongoInfo

from mongo.oper import ping, query, drop, insert_one, insert_many, update_many

import pytest
from yaml import safe_load


@pytest.fixture(scope='module')
def conf() -> Any:
    with open("tests/info.yml", 'r', encoding='utf-8') as f:
        conf = safe_load(f)

    return conf


@pytest.fixture(scope='module')
def mongo_info(conf) -> MongoInfo:
    return MongoInfo.from_dict(conf['local_test_mongo'])


class TestMongoInfo:
    def test_init(self):
        i = MongoInfo(
            mongo_url='localhost:27017',
            mongo_database='test',
        )

        assert i.mongo_url == 'localhost:27017'
        assert i.mongo_database == 'test'
        assert i.mongo_user is None
        assert i.mongo_pass is None
        assert i.mongo_auth_source is None
        assert i.mongo_collection is None

    def test_from_conf(self):
        conf = {
            'mongo_url': 'localhost:27017',
            'mongo_database': 'test',
        }

        i = MongoInfo.from_dict(conf)

        assert i.mongo_url == 'localhost:27017'
        assert i.mongo_database == 'test'
        assert i.mongo_user is None
        assert i.mongo_pass is None
        assert i.mongo_auth_source is None
        assert i.mongo_collection is None

    def test_connect_url(self):
        i = MongoInfo(
            mongo_url='localhost:27017',
            mongo_database='test',
        )

        assert i.connect_url == 'mongodb://localhost:27017'

        i = MongoInfo(
            mongo_url='localhost:27017',
            mongo_database='test',
            mongo_user='test',
            mongo_password='test',
        )

        assert i.connect_url == 'mongodb://test:test@localhost:27017'

        i = MongoInfo(
            mongo_url='localhost:27017',
            mongo_database='test',
            mongo_user='test',
            mongo_password='test',
            mongo_auth_source='test',
        )

        assert i.connect_url == 'mongodb://test:test@localhost:27017/test'

    def test_from_yml(self, conf):
        i = MongoInfo.from_dict(conf['gpu_test_mongo'])

        assert i.mongo_url == conf['gpu_test_mongo']['mongo_url']
        assert i.mongo_database == conf['gpu_test_mongo']['mongo_database']
        assert i.mongo_user == conf['gpu_test_mongo'].get('mongo_user', None)
        assert i.mongo_pass == conf['gpu_test_mongo'].get('mongo_pass', None)
        assert i.mongo_auth_source == conf['gpu_test_mongo'].get('mongo_auth_source', None)
        assert i.mongo_collection == conf['gpu_test_mongo'].get('mongo_collection', None)

    def test_set_collection(self, mongo_info):
        assert mongo_info.mongo_collection is None

        new_mongo_info = mongo_info.set_collection('mongo_test')

        assert new_mongo_info != mongo_info


class TestOperation:
    def test_ping(self, mongo_info):
        assert ping(mongo_info) is True

    def test_insert_one(self, mongo_info):
        insert_one(mongo_info.set_collection('mongo_test'),
                   {'name': 'Apple', 'age': 18})

    def test_insert_many(self, mongo_info):
        insert_many(mongo_info.set_collection('mongo_test'),
                    [
                        {'name': 'Bob', 'age': 19},
                        {'name': 'Chris', 'age': 19},
                        {'name': 'David', 'age': 20}
                    ])

    def test_query(self, mongo_info):
        docs = query(mongo_info.set_collection('mongo_test'))

        assert len(list(docs)) == 4

    def test_update_one(self, mongo_info):
        update_values = {
            '$set': {'age': 24}
        }

        condition = {'name': 'David'}

        update_many(mongo_info.set_collection('mongo_test'),
                    condition,
                    update_values)

        docs = query(mongo_info.set_collection('mongo_test'),
                     filter=condition)

        for doc in docs:
            assert doc['age'] == 24
            assert doc['name'] == 'David'

    def test_update_many(self, mongo_info):
        update_values = [
            {'$set': {'name': 'David', 'age': 25}},
            {'$set': {'name': 'Eve', 'age': 19}},
            {'$set': {'name': 'Frank', 'age': 19}}
        ]

        update_many(mongo_info.set_collection('mongo_test'),
                    {},
                    update_values)

        docs = query(mongo_info.set_collection('mongo_test'),
                     filter={})

        assert len(list(docs)) == 4

    def test_drop(self, mongo_info):
        drop(mongo_info.set_collection('mongo_test'))

        docs = query(mongo_info.set_collection('mongo_test'))

        assert len(list(docs)) == 0
