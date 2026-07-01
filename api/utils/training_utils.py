from copy import deepcopy
import math
from typing import Type

from torch import optim
import torch.nn as nn


def get_optimizer(model: Type[nn.Module], optim_name: str, **optim_params):
    optimizer = getattr(optim, optim_name)(model.parameters(), **optim_params)
    return optimizer


def get_lr_scheduler(optimizer, lr_sch_type, **lr_sch_params):
    try:
        scheduler = getattr(optim.lr_scheduler, lr_sch_type)(optimizer, **lr_sch_params[lr_sch_type])
    except AttributeError:
        scheduler = CosineAnnealingWarmupPolynomial(optimizer, **lr_sch_params[lr_sch_type])

    return scheduler


class CosineAnnealingWarmupPolynomial(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_epoch, warmup_init_lr, cosine_params: dict, poly_params: dict):
        self.warmup_epoch = warmup_epoch
        self.warmup_init_lr = warmup_init_lr
        self.cosine_params = deepcopy(cosine_params)
        self.poly_params = deepcopy(poly_params)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch=-1, verbose=False)

    def _cosine_lr(self, epoch, base_lr):
        params = self.cosine_params
        return params["eta_min"] + (base_lr - params["eta_min"]) * (
                1 + math.cos(epoch * math.pi / params["T_max"])) / 2

    def _poly_lr(self, epoch, base_lr):
        params = self.poly_params
        return base_lr * (1.0 - min(params["total_epochs"], epoch) / params["total_epochs"]) ** params["power"]

    def _linear_lr(self, epoch, base_lr):
        return self.warmup_init_lr + (base_lr - self.warmup_init_lr) * epoch / self.warmup_epoch

    def get_lr(self):
        if self.last_epoch < self.warmup_epoch:
            return [self._linear_lr(self.last_epoch, base_lr) for base_lr in self.base_lrs]
        else:
            epoch = self.last_epoch - self.warmup_epoch
            return [self._poly_lr(epoch, self._cosine_lr(epoch, base_lr)) for base_lr in self.base_lrs]
