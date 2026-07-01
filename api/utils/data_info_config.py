from collections import defaultdict
from copy import deepcopy
from typing import Dict

import yaml

from api.anydb.miniolib import MinioInfo
from api.anydb.mongo import MongoInfo
from api.anydb.doris import DorisInfo


class DataInfoConfig(object):
    def __init__(self, config_data: Dict) -> None:
        self.config_data = config_data
        self.mongo_info_dict = defaultdict()
        self.minio_info_dict = defaultdict()
        self.requests_dict = defaultdict()
        self.doris_table_dict = defaultdict()
        self._parse_collection_data()

    def _parse_collection_data(self) -> None:
        for dt_source, dt_infos in self.config_data.items():
            if dt_source == 'mongo':
                for dt_database, dtb_infos in dt_infos.items():
                    mongo_info = MongoInfo(
                        mongo_url=dtb_infos['mongo_url'],
                        mongo_user=dtb_infos['mongo_user'],
                        mongo_password=dtb_infos['mongo_pass'],
                        mongo_auth_source=dtb_infos['mongo_auth_source'],
                        mongo_database=dt_database,
                        mongo_collection="")
                    for dt_collection_p, dtc_info in dtb_infos['mongo_collection'].items():
                        for dt_collection, mc_name in dtc_info.items():
                            mongo_info_item = deepcopy(mongo_info)
                            mongo_info_item.mongo_collection = mc_name
                            self.mongo_info_dict[dt_collection_p + '_' + dt_collection] = mongo_info_item
            elif dt_source == 'minio':
                for dt_database, dtb_infos in dt_infos.items():
                    minio_info_item = MinioInfo(
                        minio_url=dtb_infos['minio_url'],
                        minio_access_key=dtb_infos['minio_access_key'],
                        minio_secret_key=dtb_infos['minio_secret_key'],
                        bucket_name=dtb_infos['bucket_name'],
                        minio_secure=dtb_infos.get('minio_secure', False),
                    )
                    self.minio_info_dict[dt_database] = minio_info_item
            elif dt_source == 'requests':
                for req_name, req_url in dt_infos.items():
                    self.requests_dict[req_name] = req_url
            elif dt_source == 'doris':
                for dt_database, dtb_infos in dt_infos.items():
                    for span_date, table_names in dtb_infos["table_name"].items():
                        for table_tag, table_name in table_names.items():
                            self.doris_table_dict[f"{dt_database}_{span_date}_{table_tag}"] = table_name

    def get_mongo_dtb_info(self, dtb_name: str, rm_collection: bool = True) -> MongoInfo:
        mongo_dtb_info_dict = self.config_data["mongo"][dtb_name]
        if rm_collection:
            mongo_dtb_info_dict.pop('mongo_collection')
        mongo_dtb_info_dict['mongo_database'] = dtb_name
        mongo_dtb_info = MongoInfo.from_dict(mongo_dtb_info_dict)
        return mongo_dtb_info

    def get_minio_dtb_info(self, bucket_name: str) -> MinioInfo:
        minio_info_dict = self.config_data["minio"][bucket_name]
        minio_info = MinioInfo.from_dict(minio_info_dict)
        return minio_info
    
    def get_doris_dtb_info(self, dtb_name: str) -> DorisInfo:
        doris_info_dict = deepcopy(self.config_data["doris"][dtb_name])
        doris_info_dict.pop("table_name")
        doris_info = DorisInfo.from_dict(doris_info_dict)
        return doris_info


_config_instance = None


def load_data_info_config(file_path: str = "config.yaml") -> DataInfoConfig:
    global _config_instance
    if _config_instance is None:
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)

        _config_instance = DataInfoConfig(config_data)
    return _config_instance

