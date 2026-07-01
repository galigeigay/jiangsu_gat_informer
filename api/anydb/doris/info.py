from typing import Optional
from dataclasses import dataclass
from copy import deepcopy


@dataclass(frozen=True)
class DorisInfo:
    """
    doris 连接信息类, 最小单元应该是数据库名
    """
    doris_user: str
    doris_password: Optional[str]
    doris_host: str

    # 因为stream load 需要前端接受端口
    # 普通查询需要后端接受端口
    doris_be_port: int
    doris_fe_port: int
    doris_database: str

    @property
    def doris_url(self) -> str:
        """
        doris 连接地址
        """
        if self.doris_password:
            return (f"doris+pymysql://{self.doris_user}:{self.doris_password}@"
                    f"{self.doris_host}:{self.doris_be_port}/{self.doris_database}")
        else:
            return f"doris+pymysql://{self.doris_user}@{self.doris_host}:{self.doris_be_port}/{self.doris_database}"

    def stream_load_url(self, table_name: str) -> str:
        """
        doris stream load 后端接受地址
        """
        return f"http://{self.doris_host}:{self.doris_fe_port}/api/{self.doris_database}/{table_name}/_stream_load"

    @classmethod
    def from_dict(cls, d: dict):
        """
        从字典中构造 DorisInfo 对象
        """
        return cls(
            doris_user=d.get("doris_user"),
            doris_password=d.get("doris_password", None),
            doris_host=d.get("doris_host"),
            doris_be_port=d.get("doris_be_port"),
            doris_fe_port=d.get("doris_fe_port"),
            doris_database=d.get("doris_database"),
        )

    def copy(self, **kwargs):
        """
        复制一个新的 DorisInfo 对象， 并更新其中的属性
        """
        d = vars(deepcopy(self))
        d.update(kwargs)
        return DorisInfo.from_dict(d)
