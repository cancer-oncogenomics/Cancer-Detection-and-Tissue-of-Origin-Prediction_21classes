#!/usr/bin/env python
# coding: utf-8
# @Time: 2024/7/20 下午10:56
# @Author：shenny
# @File: loss_weight_scheduler.py
# @Software: PyCharm

"""loss权重调节器"""

import numpy as np
import torch


def get_loss_weight_scheduler(algo: str, weights: list, end_weights=None, num_epochs=None):
    if algo == "dwa":
        return DynamicWeightAveraging(weights)
    elif algo == "lws":
        return LinearWeightScheduler(weights, weights, end_weights, num_epochs)
    elif algo == "none":
        return FakeScheduler(weights)
    else:
        raise ValueError("loss权重调度器选择错误")


class DynamicWeightAveraging:
    """动态权重平均
    算法：
        1. 计算当前损失值与上一次损失值的比值
        2. 权重 = 1 / 比值
        3. 归一化
    当前损失值小于上一次损失值时，权重增大；反之，权重减小
    """

    def __init__(self, weights: list):
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.previous_losses = None

    def update_weights(self, current_losses, min_value=0.01, **kwargs):
        current_losses = torch.tensor(current_losses, dtype=torch.float32)
        current_losses = torch.where(torch.isnan(current_losses), torch.tensor(min_value), current_losses)

        if self.previous_losses is None:
            self.previous_losses = current_losses
            return self.weights

        loss_ratios = current_losses / self.previous_losses
        self.weights = 1 / loss_ratios
        self.weights = self.weights / self.weights.sum()  # 归一化
        self.weights = torch.clamp(self.weights, min=min_value)
        self.previous_losses = current_losses

        return self.weights


class LinearWeightScheduler:
    """线性权重调度器
    算法:
        1. 从开始权重到结束权重线性调整
    """
    def __init__(self, weights: list, start_weights, end_weights, num_epochs):
        assert len(start_weights) == len(weights), "start_weights长度与num_losses不一致"
        assert len(end_weights) == len(weights), "end_weights长度与num_losses不一致"

        self.weights = weights
        self.start_weights = start_weights
        self.end_weights = end_weights
        self.num_epochs = num_epochs

    def update_weights(self, epoch, **kwargs):
        weights = []
        for i in range(len(self.weights)):
            weight = self.start_weights[i] + (self.end_weights[i] - self.start_weights[i]) * epoch / self.num_epochs
            weights.append(weight)
        return weights


class FakeScheduler(object):
    """一个假的调度器，当没有选择调度器是模拟一个调度器"""

    def __init__(self, weights: list):
        self.weight = weights

    def update_weights(self, **kwargs):
        return self.weight
