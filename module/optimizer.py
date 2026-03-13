#!/usr/bin/env python
# coding: utf-8
# @Time: 2024/7/20 下午5:49
# @Author：shenny
# @File: optimizer.py
# @Software: PyCharm

"""优化器选择"""

import torch

from module import log


def get_optimizer(algo: str, model, lr, weight_decay: float, frozen_layers: list = None, other_params=None):
    """获取优化器

    :param algo: 优化器选择。 [adam, sgd, rmsprop]
    :param model: 模型
    :param lr: 初始学习率
    :param weight_decay: 权重衰减
    :param frozen_layers: 冻结层列表
    :param other_params: 其他模型参数
    """

    # 冻结某些层
    for name, param in model.named_parameters():
        param.requires_grad = True # 默认所有层都训练, 这么设置是为了防止某些层被冻结
        for layer in frozen_layers:
            if name.startswith(layer):
                param.requires_grad = False
                log.info(f"冻结层: {name}", 1)

    if other_params:
        parameters = list(model.parameters()) + list(other_params)
    else:
        parameters = model.parameters()

    # 选择优化器
    if algo == "adam":
        optimizer = torch.optim.Adam(parameters, weight_decay=weight_decay, lr=lr)
    elif algo == "sgd":
        optimizer = torch.optim.SGD(parameters, weight_decay=weight_decay, lr=lr)
    elif algo == "rmsprop":
        optimizer = torch.optim.RMSprop(parameters, weight_decay=weight_decay, lr=lr)
    else:
        raise ValueError("优化器选择错误")

    return optimizer

