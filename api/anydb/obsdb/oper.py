from typing import Sequence, List
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

from ._types import ObsInfo, new_obs_client
from . import errors


def _default_filter_hook(keys: Sequence[str],
                         *args,
                         **kwargs) -> Sequence[str]:
    """
    默认过滤函数
    """
    return keys


def list_objects(obs_info: ObsInfo, prefix: str, *args, **kwargs) -> Sequence[str]:
    """
    列出obs对象, 返回SDK公共结果对象
    """
    client = new_obs_client(obs_info=obs_info)

    filter_hook: callable = kwargs.pop('filter_hook', _default_filter_hook)

    resp = client.listObjects(
        bucketName=obs_info.bucket_name,
        prefix=prefix,
    )

    # 检查结果状态
    if resp.status != 200:
        raise errors.ObsObjectFetchFailed(f'list objects failed: {resp.errorMessage}')

    # 从结果对象中提取keys
    keys = [content.key for content in resp.body.contents]

    if len(keys) == 0:
        raise errors.ObsObjectListEmtpy(f'empty list @ {prefix}')

    return filter_hook(keys, *args, **kwargs)


def download_file(
        obs_info: ObsInfo,
        object_key: str,
        target_path: str,
        progress_callback: callable = None,
        *args,
        **kwargs) -> Path:
    """
    obs对象下载到本地, 并发下的最小执行单元
    """
    client = new_obs_client(obs_info=obs_info)

    # 如果本地已经存在, 则不再下载
    if Path(target_path).is_file() and Path(target_path).stat().st_size > 0:
        return Path(target_path)

    try:
        resp = client.getObject(
            bucketName=obs_info.bucket_name,
            objectKey=object_key,
            downloadPath=target_path
        )

    except Exception as e:
        raise errors.ObsObjectFetchFailed(f'download object failed: {e}')

    if resp.status >= 300:
        raise errors.ObsObjectFetchFailed(f'download object failed: {resp.errorMessage}')

    if progress_callback:
        progress_callback = partial(progress_callback, *args, **kwargs)
        progress_callback(*args, **kwargs)

    return Path(target_path)


def download_files_concurrency(obs_info: ObsInfo,
                               keys: Sequence[str],
                               save_dir: str,
                               **kwargs) -> Sequence[Path]:
    """
    并发下载obs对象, 返回下载后的本地路径

    Args:
        obs_info: ObsBucketInfo
        keys: obs对象key列表
        save_dir: 本地保存路径
        **kwargs:
            rename_hook: 改名钩子函数
            max_workers: 最大并发数
    """
    if len(keys) == 0:
        raise errors.ObsObjectListEmtpy('empty keys list')

    # 改名钩子函数
    rename_hook: callable = kwargs.pop('rename_hook', lambda x: x)

    # 最大并发数
    max_workers = kwargs.pop('max_workers', 10)

    create_new_dir(save_dir)

    # 保存下载后的本地路径
    local_paths: List[Path] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for key in keys:
            # 本地保存路径
            local_path = Path(save_dir).joinpath(Path(key).name)

            # 改名
            local_path = rename_hook(local_path)

            # 下载obs对象
            future = executor.submit(download_file, obs_info, key, str(local_path), **kwargs)
            futures.append(future)

        for future in as_completed(futures):
            try:
                downloaded_path = future.result()
            except errors.ObsObjectFetchFailed as e:
                print(f'ObsObjectFetchFailed: {e}')
            else:
                local_paths.append(downloaded_path)

    return local_paths


def create_new_dir(dir_path: str) -> Path:
    """
    创建新的文件夹

    :param dir_path: 文件夹路径
    :return: 文件夹路径对象
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    return dir_path
