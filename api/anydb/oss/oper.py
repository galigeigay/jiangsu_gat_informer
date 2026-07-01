from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
import os

from ._types import OSSInfo

import oss2
from oss2 import exceptions


def list_files_by_prefix(oss_info: OSSInfo,
                         prefix: str) -> List[str]:
    """
    列出OSS bucket路径前缀prefix下所有文件名

    :param oss_info: OSS配置信息
    :param prefix: OSS路径前缀

    :return: 文件名列表
    """
    bucket = oss_info.create_conn()

    rst_list = []
    for obj in oss2.ObjectIterator(bucket, prefix=prefix):
        rst_list.append(obj.key)

    return rst_list


def download_file(oss_info: OSSInfo,
                  key: str,
                  save_dir: str) -> str:
    """
    从OSS下载文件, 且为单个文件，并发下最小执行单元

    :param oss_info: OSS配置信息
    :param key: OSS文件路径
    :param save_dir: 本地保存文件夹路径

    :return: 本地文件路径
    """
    bucket = oss_info.create_conn()

    if not Path(save_dir).exists():
        raise FileNotFoundError("save_dir {} not exists".format(save_dir))

    filename = Path(key).name
    local_filepath = Path(save_dir).joinpath(filename)

    bucket.get_object_to_file(key, local_filepath)

    print("downloaded {}".format(key))
    return local_filepath.__str__()


def upload_file(oss_info: OSSInfo,
                key: str,
                local_filepath: str,
                retry: int = 3) -> str:
    """
    上传文件到OSS

    :param oss_info: OSS配置信息
    :param key: OSS文件路径
    :param local_filepath: 本地文件路径
    :param retry: 重试次数

    :return: OSS文件路径
    """
    bucket = oss_info.create_conn()

    for _ in range(retry):
        try:
            bucket.put_object_from_file(key, local_filepath)
        except exceptions.InvalidRequest:
            continue
        except Exception as e:
            raise UserWarning(f"Upload {local_filepath} to {key} failed. Error: {e}")
        else:
            print("uploaded {}".format(key))
            return key


def delete_file(oss_info: OSSInfo,
                key: str) -> str:

    bucket = oss_info.create_conn()
    bucket.delete_object(key)

    print("deleted {}".format(key))

    return key


def download_file_concurrency(oss_info: OSSInfo,
                              keys: List[str],
                              save_dir: str,
                              **kwargs) -> List[str]:
    """
    并发下载OSS文件

    :param oss_info: OSS配置信息
    :param keys: OSS文件路径列表
    :param save_dir: 本地保存文件夹路径
    :param kwargs: max_workers: 最大并发数

    :return: 本地文件路径列表
    """

    max_workers = kwargs.get('max_workers', 10)

    downloaded_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(download_file, oss_info, key, save_dir): key for key in keys}

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                data = future.result()
            except Exception as e:
                print(f"download {key} failed. Error: {e}")
            else:
                downloaded_files.append(data)

    return downloaded_files


def upload_file_concurrency(oss_info: OSSInfo,
                            prefix: str,
                            local_files: List[str],
                            **kwargs) -> List[str]:
    """
    并发上传文件到OSS

    :param oss_info: OSS配置信息
    :param prefix: OSS文件路径前缀
    :param local_files: 本地文件路径列表
    :param kwargs: max_workers: 最大并发数

    :return: OSS文件路径列表
    """

    max_workers = kwargs.get('max_workers', 5)

    uploaded_keys = []
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for local_file in local_files:
            key = Path(prefix).joinpath(Path(local_file).name).as_posix()
            futures.append(executor.submit(upload_file,
                                             oss_info, key,
                                             local_file,
                                             retry=3))

        for future in as_completed(futures):
            try:
                data = future.result()
            except Exception as e:
                print(f"upload {local_file} failed. Error: {e}")
            else:
                uploaded_keys.append(data)

    return uploaded_keys


def delete_file_concurrency(oss_info: OSSInfo,
                            keys: List[str],
                            **kwargs) -> List[str]:
    """
    并发删除OSS文件

    :param oss_info: OSS配置信息
    :param keys: OSS文件路径列表
    :param kwargs: max_workers: 最大并发数

    :return: OSS文件路径列表
    """

    max_workers = kwargs.get('max_workers', 5)

    deleted_keys = []
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for key in keys:
            futures.append(executor.submit(delete_file, oss_info, key))

        for future in as_completed(futures):
            try:
                data = future.result()
            except Exception as e:
                print(f"delete {key} failed. Error: {e}")
            else:
                deleted_keys.append(data)

    return deleted_keys
