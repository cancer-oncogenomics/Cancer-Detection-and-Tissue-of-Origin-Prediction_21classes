#!/usr/bin/env python
# coding: utf-8
# @Time: 2024/7/20 下午6:01
# @Author：shenny
# @File: algo.py
# @Software: PyCharm


"""优化器调度器"""

import torch


__all__ = ["get_scheduler"]


def get_scheduler(algo: str, optimizer, step_size: int = None, gamma: float = None, mode: str = None):
    """获取优化器调度器

    :param algo: str, 优化器调度器选择。 [steplr, reducelr, none]
    :param optimizer: torch.optim.Optimizer, 优化器
    :param step_size: int, 学习率衰减周期, 仅在algo为steplr时有效
    :param gamma: float, 学习率衰减因子, 仅在algo为steplr时有效
    :param mode: str, 早停的模式. [min, max] 仅在algo为reducelr时有效
    """

    if algo == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif algo == "reducelr":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode)
    elif algo == "none":
        scheduler = FakeScheduler()
    else:
        raise ValueError("优化器调度器选择错误")

    return scheduler


class FakeScheduler(object):
    """一个假的调度器，当没有选择调度器是模拟一个调度器"""

    def __init__(self):
        pass

    def step(self):
        pass
