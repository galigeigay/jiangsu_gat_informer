class ObsException(Exception):
    """
    大地量子Obs异常类
    """
    ...


class ObsObjectListEmtpy(ObsException):
    """
    Obs对象列表为空
    """
    ...


class ObsObjectFetchFailed(ObsException):
    """
    Obs对象获取失败
    """
    ...
