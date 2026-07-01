"""
MongoDB操作方法
"""
from typing import Mapping, Any, Sequence

from ._types import MongoInfo

import pymongo
from pymongo.errors import ServerSelectionTimeoutError
from pymongo.results import DeleteResult


def ping(mongo_info: MongoInfo) -> bool:
    """
    测试MongoDB连接是否正常
    """
    client = mongo_info.create_new_client()

    try:
        client.server_info()
    except ServerSelectionTimeoutError:
        raise ConnectionError(f"MongoDB连接失败, url: {mongo_info.connect_url}")
    else:
        return True


def query(mongo_info: MongoInfo, **kwargs) -> pymongo.cursor.Cursor:
    """
        根据条件conditions捞取数据并返回符合需求的数据结构projection
        """
    client = mongo_info.create_new_client()
    db = client[mongo_info.mongo_database]
    mycol = db[mongo_info.mongo_collection]

    _query = kwargs.pop('filter', {})
    _projection = kwargs.pop('projection', None)

    if _projection is None:
        return mycol.find(filter=_query, **kwargs)
    else:
        return mycol.find(filter=_query, projection=_projection, **kwargs)


def drop(mongo_info: MongoInfo) -> None:
    """
    根据条件conditions删除数据
    """
    client = mongo_info.create_new_client()
    db = client[mongo_info.mongo_database]
    mycol = db[mongo_info.mongo_collection]

    return mycol.drop()


def insert_one(mongo_info: MongoInfo, values: dict):
    """
    把单条记录values插入到MongoDB
    """
    client = mongo_info.create_new_client()
    db = client[mongo_info.mongo_database]
    mycol = db[mongo_info.mongo_collection]

    return mycol.insert_one(values)


def insert_many(mongo_info: MongoInfo, values: list):
    """
    把多条记录values插入到MongoDB
    """
    client = mongo_info.create_new_client()
    db = client[mongo_info.mongo_database]
    mycol = db[mongo_info.mongo_collection]

    return mycol.insert_many(values)


def delete_one(mongo_info: MongoInfo,
               conditions: dict) -> DeleteResult:
    """
    根据条件conditions删除一条数据
    """
    client = mongo_info.create_new_client()
    db = client[mongo_info.mongo_database]
    mycol = db[mongo_info.mongo_collection]

    return mycol.delete_one(conditions)


def delete_many(mongo_info: MongoInfo,
                conditions: dict) -> DeleteResult:
    """
    根据条件conditions删除多条数据
    """
    client = mongo_info.create_new_client()
    db = client[mongo_info.mongo_database]
    mycol = db[mongo_info.mongo_collection]

    return mycol.delete_many(conditions)


def update_many(mongo_info: MongoInfo,
                conditions: dict,
                update: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> pymongo.results.UpdateResult:
    """
    根据条件conditions更新多条数据, 如果存在则更新，不存在则插入
    """
    client = mongo_info.create_new_client()
    db = client[mongo_info.mongo_database]
    mycol = db[mongo_info.mongo_collection]

    return mycol.update_many(conditions, update, upsert=True)


def update_one(mongo_info: MongoInfo,
               conditions: dict,
               update: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> pymongo.results.UpdateResult:
    """
    根据条件conditions更新一条数据, 如果存在则更新，不存在则插入
    """
    client = mongo_info.create_new_client()
    db = client[mongo_info.mongo_database]
    mycol = db[mongo_info.mongo_collection]

    return mycol.update_one(conditions, update, upsert=True)


def pipeline(mongo_info: MongoInfo,
             _pipeline: list) -> pymongo.cursor.Cursor:
    """
    执行聚合查询
    """
    client = mongo_info.create_new_client()
    db = client[mongo_info.mongo_database]
    mycol = db[mongo_info.mongo_collection]

    result = mycol.aggregate(_pipeline)

    for r in result:
        print(r)

    return result


def bulk_write(mongo_info: MongoInfo,
               operations: list) -> pymongo.results.BulkWriteResult:
    """
    执行批量写入
    """
    client = mongo_info.create_new_client()
    db = client[mongo_info.mongo_database]
    mycol = db[mongo_info.mongo_collection]

    return mycol.bulk_write(operations)
