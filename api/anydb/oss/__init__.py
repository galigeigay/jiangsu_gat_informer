"""
阿里云OSS存储交互模块
"""
from ._types import OSSInfo
from .oper import download_file, upload_file, delete_file, download_file_concurrency, upload_file_concurrency, delete_file_concurrency


__all__ = [
    "OSSInfo",
    "download_file",
    "upload_file",
    "download_file_concurrency",
    "upload_file_concurrency",
    "delete_file",
    "delete_file_concurrency",
]
