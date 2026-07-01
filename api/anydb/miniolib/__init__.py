from ._types import MinioInfo
from .oper import (
    upload_file,
    upload_folder,
    get_folder,
    get_object,
    list_objects,
    delete_object,
    get_bytes,
    create_new_client
)


__all__ = [
    'MinioInfo',
    'upload_file',
    'upload_folder',
    'get_folder',
    'get_object',
    'list_objects',
    'delete_object',
    'get_bytes',
    'create_new_client'
]
