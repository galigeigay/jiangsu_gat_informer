import urllib

import pandas as pd

from dataclasses import dataclass

from pymongo import MongoClient


@dataclass
class MongoInfo:
    mongo_url: str
    mongo_user: str
    mongo_password: str
    mongo_auth_db: str
    mongo_db: str
    mongo_collection: str


def read_mongo(mongo_info: MongoInfo, conditions={}, fields={}):
    """
    根据条件conditions从MongoDB读数据
    """
    client = MongoClient('mongodb://{}:{}@{}/{}'.format(mongo_info.mongo_user, urllib.parse.quote(mongo_info.mongo_password), mongo_info.mongo_url, mongo_info.mongo_auth_db))
    db = client[mongo_info.mongo_db]
    mycol = db[mongo_info.mongo_collection]
    if conditions:
        if fields:
            data = pd.DataFrame(list(mycol.find(conditions, fields)))
        else:
            data = pd.DataFrame(list(mycol.find(conditions)))
    else:
        if fields:
            data = pd.DataFrame(list(mycol.find({}, fields)))
        else:
            data = pd.DataFrame(list(mycol.find()))
    return data


def insert_mongo(mongo_info: MongoInfo, values: dict):
    """
    把数据values上传到MongoDB
    """
    if mongo_info.mongo_auth_db:
        client = MongoClient('mongodb://{}:{}@{}/{}'.format(mongo_info.mongo_user, urllib.parse.quote(mongo_info.mongo_password), mongo_info.mongo_url, mongo_info.mongo_auth_db))
    else:
        client = MongoClient('mongodb://{}:{}@{}'.format(mongo_info.mongo_user, urllib.parse.quote(mongo_info.mongo_password), mongo_info.mongo_url))
    db = client[mongo_info.mongo_db]
    mycol = db[mongo_info.mongo_collection]
    result = mycol.insert_many(values)
    return len(result.inserted_ids)


def delete_mongo(mongo_info: MongoInfo, conditions: dict):
    """
    根据条件conditions从MongoDB删除数据
    """
    client = MongoClient('mongodb://{}:{}@{}/{}'.format(mongo_info.mongo_user, urllib.parse.quote(mongo_info.mongo_password), mongo_info.mongo_url, mongo_info.mongo_auth_db))
    db = client[mongo_info.mongo_db]
    mycol = db[mongo_info.mongo_collection]
    result = mycol.delete_many(conditions)
    return result.deleted_count


def mongo_decimal_to_float(input_decimal):
    """
    将Mongo的Decimal格式转为float
    """
    if pd.isna(input_decimal):
        rst = None
    else:
        rst = float(input_decimal.to_decimal())
    return rst
