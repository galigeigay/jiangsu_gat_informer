from typing import Optional

import oss2


class OSSInfo:
    oss_url: str
    oss_access_key_id: str
    oss_access_key_secret: str
    oss_bucket: Optional[str]
    """
    OSS对象存储配置信息类
    """

    def __init__(self,
                 oss_url: str,
                 oss_access_key_id: str,
                 oss_access_key_secret: str,
                 oss_bucket: Optional[str] = None):
        self.oss_url = oss_url
        self.oss_access_key_id = oss_access_key_id
        self.oss_access_key_secret = oss_access_key_secret
        self.oss_bucket = oss_bucket

    @classmethod
    def from_dict(cls, oss_info: dict):
        if 'oss_url' not in oss_info:
            raise ValueError("oss_url not found in oss_info")

        if 'oss_access_key_id' not in oss_info:
            raise ValueError("oss_access_key_id not found in oss_info")

        if 'oss_access_key_secret' not in oss_info:
            raise ValueError("oss_access_key_secret not found in oss_info")

        return cls(
            oss_url=oss_info['oss_url'],
            oss_access_key_id=oss_info['oss_access_key_id'],
            oss_access_key_secret=oss_info['oss_access_key_secret'],
            oss_bucket=oss_info.get('oss_bucket', None),
        )

    def create_conn(self) -> oss2.Bucket:
        auth = oss2.Auth(self.oss_access_key_id, self.oss_access_key_secret)
        bucket = oss2.Bucket(auth, self.oss_url, self.oss_bucket)
        return bucket
