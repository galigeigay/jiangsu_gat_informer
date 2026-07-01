"""
事务性批量插入
"""
from typing import Dict, TypeVar, Generic
from queue import Queue
from logging import Logger

from ._types import MongoInfo

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.results import InsertManyResult, UpdateResult, DeleteResult
from pymongo.errors import OperationFailure
import pandas as pd

Result = TypeVar("Result", InsertManyResult, UpdateResult, DeleteResult)


class Operation:
    def execute(self, col: Collection, session, **kwargs) -> Generic[Result]:
        raise NotImplementedError


class InsertMany(Operation):
    def __init__(self, pdf: pd.DataFrame):
        self.data_frame = pdf

    def execute(self, col: Collection, session, **kwargs):
        resp = col.insert_many(self.data_frame.to_dict(orient='records'), session=session)
        logger: Logger | bool = kwargs.get("logger", False)

        if logger:
            logger.info(f"Inserted {len(resp.inserted_ids)} records into {col.full_name}")


class UpdateOne(Operation):
    def __init__(self, conditions: dict, update: pd.DataFrame):
        self.conditions = conditions
        self.update = update

    def execute(self, col: Collection, session, **kwargs):
        update = {'$set': self.update.to_dict(orient='records')[0]}
        resp = col.update_one(self.conditions, update, session=session, upsert=True)

        logger: Logger | bool = kwargs.get("logger", False)

        if logger:
            logger.info(f"Updated {resp.modified_count} records in {col.full_name}")


class DeleteMany(Operation):
    def __init__(self, conditions: dict):
        self.conditions = conditions

    def execute(self, col: Collection, session, **kwargs):
        resp = col.delete_many(self.conditions, session=session)
        logger: Logger | bool = kwargs.get("logger", False)

        if logger:
            logger.info(f"Deleted {resp.deleted_count} records in {col.full_name}")


class UpdateMany(Operation):
    def __init__(self, conditions: dict, update: pd.DataFrame):
        self.conditions = conditions
        self.update = update

    def execute(self, col: Collection, session, **kwargs):
        update = {'$set': self.update.to_dict(orient='records')}
        resp = col.update_many(self.conditions, update, session=session, upsert=True)

        logger: Logger | bool = kwargs.get("logger", False)

        if logger:
            logger.info(f"Updated {resp.modified_count} records in {col.full_name}")


class PandasDataFrameTransaction:
    def __init__(self, database_mongo_info: MongoInfo):
        self.client = MongoClient(database_mongo_info.connect_url)
        self.operations: Dict[MongoInfo, Queue[Operation]] = {}

    def register_operation(self, mongo_info: MongoInfo, oper: Operation):
        """
        Register a DataFrame to be inserted into a specified collection.
        """
        if not isinstance(mongo_info, MongoInfo):
            raise TypeError("mongo_info must be a MongoInfo object")

        if not isinstance(oper, Operation):
            raise TypeError("oper must be an Operation object")

        if mongo_info not in self.operations:
            self.operations[mongo_info] = Queue()

            # 重大BUG， 如果不是Replicate Set Mongo客户端，因此事务第一条永远会被忽略
            # 因此需要在这里初始化一个虚拟操作

        self.operations[mongo_info].put(oper)

    def run(self, **kwargs):
        """
        Execute all registered inserts as a single transaction. Roll back if any insert fails.
        """
        logger: Logger | bool = kwargs.get("logger", False)

        with self.client.start_session() as _session:

            with _session.start_transaction():

                for mongo_info, oper in self.operations.items():
                    db = self.client[mongo_info.mongo_database]
                    mycol = db[mongo_info.mongo_collection]

                    while not oper.empty():
                        # 队列保证了操作的顺序
                        op = oper.get()
                        try:
                            op.execute(mycol, session=_session, logger=logger)
                        except Exception as e:
                            _session.abort_transaction()
                            if logger:
                                logger.error(f"An error occurred: {e}")

    def clear(self):
        """
        Clear all registered data frames.
        """
        self.operations.clear()
