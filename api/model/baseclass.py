from abc import ABCMeta, abstractmethod
from typing import Any


class BaseData(metaclass=ABCMeta):
    """
    数据基类

    数据默认放在self._data, 可通过self.data获取
    """
    @abstractmethod
    def __init__(self, class_name='BaseData'):
        self._class_name = class_name
        self._data = None

    def __repr__(self):
        return "{}()".format(self._class_name)

    @property
    def data(self):
        return self._data


class BaseModel(metaclass=ABCMeta):
    """
    模型基类

    模型默认放在self._model, 可通过self.model获取

    需要实现self.train(), self.predict(), self.save(), self.load()
    """
    @abstractmethod
    def __init__(self, class_name='BaseModel'):
        self._class_name = class_name
        self._model = None

    def __repr__(self):
        return "{}()".format(self._class_name)

    @property
    def model(self):
        return self._model

    @abstractmethod
    def train(self, **kwargs):
        ...

    @abstractmethod
    def predict(self, **kwargs) -> Any:
        ...

    @abstractmethod
    def save(self, **kwargs):
        ...

    @abstractmethod
    def load(self, **kwargs):
        ...


class BasePipeline(metaclass=ABCMeta):
    """
    模型流程基类

    需要实现self.run()
    """
    @abstractmethod
    def __init__(self, class_name='BasePipeline'):
        self._class_name = class_name

    def __repr__(self):
        return "{}()".format(self._class_name)

    @abstractmethod
    def run(self, **kwargs):
        ...
