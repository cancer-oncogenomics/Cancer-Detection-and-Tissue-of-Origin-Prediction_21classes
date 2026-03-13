#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/28 下午3:58
# @Author  : shenny
# @File    : loss.py
# @Software: PyCharm

"""损失函数"""

from torch import nn
import torch

from module import log

__all__ = ["get_loss"]


def get_loss(algos: str):
    """获取损失函数

    :param algos: 损失函数选择。 ccsa
    """

    if algos == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif algos == "csa_loss":
        return CSALoss()
    else:
        raise ValueError("损失函数选择错误")


class CSALoss(object):
    """CSA损失函数, 主要用于域适应"""

    def __init__(self):
        pass

    def __call__(self, source_features, target_features, source_labels, target_labels, margin=1.0):
        loss = 0

        source_labels = source_labels.argmax(axis=1)
        target_labels = target_labels.argmax(axis=1)

        for sf, tf, sl, tl in zip(source_features, target_features, source_labels, target_labels):
            if sl == tl:  # 如果源域样本与目标域样本的标签相同
                loss += torch.norm(sf - tf)  # 累加它们的欧氏距离
            else:  # 如果源域样本与目标域样本的标签不同
                loss += torch.clamp(margin - torch.norm(sf - tf), min=0)  # 累加 margin 与它们欧氏距离之差的最大值, margin相当于惩罚阈值

        return loss / (len(source_features) * len(target_features))  # 返回平均损失值
