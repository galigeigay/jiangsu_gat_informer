import logging

import mongo
from mongo.transactions import PandasDataFrameTransaction
from mongo.transactions import InsertMany, DeleteMany, UpdateMany, UpdateOne
from mongo import MongoInfo


import pytest
from yaml import safe_load
import pandas as pd


@pytest.fixture(scope='module')
def conf() -> dict:
    with open('tests/info.yml') as f:
        conf = safe_load(f)

    return conf


@pytest.fixture(scope='module')
def collection1(conf) -> MongoInfo:
    db = MongoInfo.from_dict(conf['local_test_mongo'])
    col = db.set_collection('col1')
    mongo.drop(col)

    yield col

    mongo.drop(col)


@pytest.fixture(scope='module')
def collection2(conf) -> MongoInfo:
    db = MongoInfo.from_dict(conf['local_test_mongo'])
    col = db.set_collection('col2')
    mongo.drop(col)

    yield col

    mongo.drop(col)


class TestPandasDataFrameTransaction:

    df1 = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [20, 30, 40]
    })

    df2 = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [20, 30, 40]
    })

    @pytest.fixture(scope='class')
    def pd_transaction(self, collection1):
        return PandasDataFrameTransaction(collection1)

    def test_insert_many(self, pd_transaction, collection1, collection2):
        op = InsertMany(self.df1)
        pd_transaction.register_operation(collection1, op)

        op = InsertMany(self.df2)
        pd_transaction.register_operation(collection2, op)

    def test_update_one(self, pd_transaction, collection1, collection2):
        update = pd.DataFrame({
            'name': 'Dva',
            'age': 15
        }, index=[0])

        op = UpdateOne({'name': 'Dva', 'age': 15}, update)
        pd_transaction.register_operation(collection1, op)

    def test_delete_many(self, pd_transaction, collection1, collection2):
        op = DeleteMany({'name': 'Charlie'})
        pd_transaction.register_operation(collection2, op)

    def test_ingest(self, pd_transaction, collection1, collection2):
        logger = logging.getLogger(__name__)

        pd_transaction.run(logger=logger)

        # check collection1
        docs = mongo.query(collection1)
        assert docs is not None

        df = pd.DataFrame(list(docs))
        assert df.shape[0] == 3
        assert df['name'].tolist() == ['Alice', 'Bob']
        assert df['age'].tolist() == [25, 30]

        # check collection2
        docs = mongo.query(collection2, filter={})
        assert docs is not None

        df = pd.DataFrame(list(docs))
        assert df.shape[0] == 2
        assert df['name'].tolist() == ['Alice', 'Bob']
        assert df['age'].tolist() == [20, 30]
