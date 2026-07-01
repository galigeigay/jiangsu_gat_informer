from dataclasses import dataclass
from typing import Mapping

from obs import ObsClient


@dataclass(frozen=True)
class ObsInfo:
    server: str
    access_key: str
    secret_key: str
    bucket_name: str

    @classmethod
    def from_dict(cls, d: Mapping[str, str]):
        return cls(
            server=d['server'],
            access_key=d['access_key'],
            secret_key=d['secret_key'],
            bucket_name=d['bucket_name'],
        )


def new_obs_client(obs_info: ObsInfo) -> ObsClient:
    return ObsClient(
        server=obs_info.server,
        access_key_id=obs_info.access_key,
        secret_access_key=obs_info.secret_key,
    )
