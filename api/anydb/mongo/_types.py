from typing import Optional
from copy import deepcopy

from pymongo import MongoClient


class MongoInfo:
    mongo_url: str
    mongo_user: Optional[str]
    mongo_pass: Optional[str]
    mongo_auth_source: Optional[str]
    mongo_database: str
    mongo_collection: Optional[str]
    """
    MongoDB连接信息类
    """

    def __init__(self,
                 mongo_url: str,
                 mongo_database: str,
                 **kwargs) -> None:

        self.mongo_url = mongo_url
        self.mongo_database = mongo_database

        self.mongo_user = kwargs.get('mongo_user', None)
        self.mongo_pass = kwargs.get('mongo_password', None)
        self.mongo_auth_source = kwargs.get('mongo_auth_source', None)
        self.mongo_collection = kwargs.get('mongo_collection', None)

    @classmethod
    def from_dict(cls, conf: dict):
        """
        从字典中读取MongoDB连接信息
        """
        mongo_url = conf.get('mongo_url')
        mongo_database = conf.get('mongo_database')

        mongo_user = conf.get('mongo_user', None)
        mongo_password = conf.get('mongo_pass', None)
        mongo_auth_source = conf.get('mongo_auth_source', None)
        mongo_collection = conf.get('mongo_collection', None)

        return cls(
            mongo_url=mongo_url,
            mongo_database=mongo_database,
            mongo_user=mongo_user,
            mongo_password=mongo_password,
            mongo_auth_source=mongo_auth_source,
            mongo_collection=mongo_collection,
        )

    @property
    def connect_url(self) -> str:
        if all([i is None for i in [self.mongo_user, self.mongo_pass, self.mongo_auth_source]]):
            url = f"mongodb://{self.mongo_url}"
        elif all([i is None for i in [self.mongo_user, self.mongo_pass]]):
            url = f"mongodb://{self.mongo_url}/{self.mongo_auth_source}"
        else:
            if self.mongo_auth_source is not None:
                url = f"mongodb://{self.mongo_user}:{self.mongo_pass}@{self.mongo_url}/{self.mongo_auth_source}"
            else:
                url = f"mongodb://{self.mongo_user}:{self.mongo_pass}@{self.mongo_url}"

        return url

    def create_new_client(self) -> MongoClient:
        """
        创建新的MongoDB客户端
        """
        return MongoClient(self.connect_url)

    def set_collection(self, collection_name: str) -> 'MongoInfo':
        """
        修改MongoDB连接信息的collection表名，并且返回一个新内存地址的MongoInfo对象
        """
        new_mongo_info = deepcopy(self)
        new_mongo_info.mongo_collection = collection_name

        return new_mongo_info
