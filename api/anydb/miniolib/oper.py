from urllib3 import HTTPHeaderDict, HTTPResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from pathlib import Path

from ._types import MinioInfo, create_new_client

# import DeleteObject from minio
from minio.api import DeleteObject


Max_IO_Concurrency = 10


def bucket_exists(minio_info: MinioInfo) -> bool:
    """
    检查bucket是否存在

    :param minio_info: minio连接信息
    :return: 返回bucket是否存在
    """
    client = create_new_client(minio_info)
    return client.bucket_exists(minio_info.bucket_name)


def upload_file(minio_info: MinioInfo,
                file_path: str,
                object_name: str) -> HTTPHeaderDict:
    """
    上传文件到Minio

    :param minio_info: minio连接信息
    :param file_path: 本地文件路径
    :param object_name: minio对象名
    :return: 返回上传文件的http头信息
    """
    client = create_new_client(minio_info)
    res = client.fput_object(minio_info.bucket_name, object_name, file_path)

    return res.http_headers


def list_objects(minio_info: MinioInfo,
                 prefix: str,
                 recursive: bool = True) -> List[str]:
    """
    列出bucket下所有对象

    :param minio_info: minio连接信息
    :param prefix: 对象名前缀
    :param recursive: 是否递归
    :return: 返回所有对象名
    """
    client = create_new_client(minio_info)
    return [o.object_name for o in client.list_objects(minio_info.bucket_name,
                                                       prefix=prefix,
                                                       recursive=recursive)]


def delete_prefix(minio_info: MinioInfo,
                  prefix: str,
                  recursive: bool = True) -> None:

    """
    删除指定前缀的对象, 如果recursive为True, 则递归删除
    :param minio_info: minio连接信息
    :param prefix: 对象名前缀
    :param recursive: 是否递归
    :return:
    """
    client = create_new_client(minio_info)
    delete_object_list = map(
        lambda o: DeleteObject(o.object_name),
        client.list_objects(minio_info.bucket_name, prefix=prefix, recursive=recursive)
    )

    errors = client.remove_objects(minio_info.bucket_name, delete_object_list)
    for e in errors:
        print("error occurred when deleting object:", e)


def delete_object(minio_info: MinioInfo,
                  object_name: str,
                  version: str = None) -> None:
    """
    删除对象

    :param minio_info: minio连接信息
    :param object_name: 对象名
    :param version: 对象版本
    :return:
    """
    client = create_new_client(minio_info)
    client.remove_object(minio_info.bucket_name, object_name, version)


def upload_folder(minio_info: MinioInfo,
                  folder_path: str,
                  prefix_path: str) -> List[HTTPHeaderDict]:
    """
    上传文件夹到Minio
    :param minio_info: minio连接信息
    :param folder_path: 本地文件夹路径
    :param prefix_path: minio对象名前缀
    :return:
    """
    # 采用pathlib库，遍历文件夹下所有文件
    # 最后再统一并发上传
    path = Path(folder_path)

    if not path.is_dir():
        raise ValueError("folder_path must be a directory")

    files = [p for p in path.rglob("*") if p.is_file()]

    with ThreadPoolExecutor(max_workers=Max_IO_Concurrency) as executor:
        futures = [executor.submit(upload_file,
                                   minio_info,
                                   str(f),
                                   f"{prefix_path}/{f.relative_to(path)}") for f in files]

        res = [f.result() for f in as_completed(futures)]

        return res


def get_object(minio_info: MinioInfo, object_name: str, file_path: str) -> str:
    """
    下载对象到本地
    :param minio_info: minio连接信息
    :param object_name: 对象名
    :param file_path: 本地文件路径
    :return:
    """
    client = create_new_client(minio_info)
    client.fget_object(minio_info.bucket_name, object_name, file_path)

    return file_path


def get_folder(minio_info: MinioInfo,
               prefix: str,
               folder_path: str,
               recursive: bool = True) -> List[str]:
    """
    下载对象到本地
    :param minio_info: minio连接信息
    :param prefix: 对象名前缀
    :param folder_path: 本地文件夹路径
    :param recursive: 是否递归
    :return:
    """
    client = create_new_client(minio_info)
    objects = client.list_objects(minio_info.bucket_name, prefix=prefix, recursive=recursive)

    if not Path(folder_path).exists():
        Path(folder_path).mkdir(parents=True)

    with ThreadPoolExecutor(max_workers=Max_IO_Concurrency) as executor:
        futures = [executor.submit(get_object,
                                   minio_info,
                                   o.object_name,
                                   f"{folder_path}/{Path(o.object_name).name}") for o in objects]

        res = [f.result() for f in as_completed(futures)]

        return res


def get_bytes(minio_info: MinioInfo,
              object_name: str) -> HTTPResponse:
    """
    获取对象的字节流
    :param minio_info: minio连接信息
    :param object_name: 对象名
    :return:
    """
    client = create_new_client(minio_info)
    return client.get_object(
        minio_info.bucket_name,
        object_name
    )
