from typing import Optional

from minio import Minio


class MinioInfo:
    """
    Minio连接信息类
    """
    minio_url: str
    minio_access_key: Optional[str]
    minio_secret_key: Optional[str]
    minio_secure: bool
    bucket_name: str

    def __init__(self,
                 minio_url: str,
                 minio_access_key: Optional[str],
                 minio_secret_key: Optional[str],
                 bucket_name: str,
                 minio_secure: bool = False) -> None:
        self.minio_url = minio_url
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.bucket_name = bucket_name
        self.minio_secure = minio_secure

    @classmethod
    def from_dict(cls, conf: dict):
        """
        从字典中读取Minio连接信息
        """
        minio_url = conf.get('minio_url')
        minio_access_key = conf.get('minio_access_key', None)
        minio_secret_key = conf.get('minio_secret_key', None)
        bucket_name = conf.get('bucket_name', '')
        minio_secure = conf.get('minio_secure', False)

        return cls(
            minio_url=minio_url,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            bucket_name=bucket_name,
            minio_secure=minio_secure,
        )


def create_new_client(minio_info: MinioInfo):
    return Minio(
        endpoint=minio_info.minio_url,
        access_key=minio_info.minio_access_key,
        secret_key=minio_info.minio_secret_key,
        secure=minio_info.minio_secure,
    )
