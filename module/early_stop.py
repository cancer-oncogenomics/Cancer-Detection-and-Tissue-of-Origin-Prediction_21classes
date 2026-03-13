#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/29 14:51
# @Author  : shenny
# @File    : early_stopp.py
# @Software: PyCharm

"""早停策略"""


class EarlyStopping(object):

    """  模型早停实例

    每次调用时，传入当前的指标值，与之前的最佳指标值进行比较，判断是否早停

    :param patience: int, 早停的容忍次数
    :param mode: str, 早停的模式. [min, max]
    :param min_step: float, 早停的最小步长
    """

    def __init__(self, patience, mode, min_step):
        self.patience = int(patience)
        self.counter = 0
        self.best_value = None
        self.mode = mode
        self.min_step = float(min_step)

        self.early_stop = False

    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value

        elif self.mode == "min" and value > (self.best_value - self.min_step):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        elif self.mode == "max" and value < (self.best_value + self.min_step):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_value = value
            self.counter = 0
        return self.early_stop


class MultiEarlyStopping(object):
    """  多指标早停策略
    :param stop_methods: list, 早停的方法. ["Valid|Response|acc,40,min,0.001", "Valid|r01b|sens,40,min,0.001"]
    :param stopper: list, 早停的方法. [("loss", EarlyStopping), ("acc", EarlyStopping)]
    """

    def __init__(self, stop_methods: list = None):
        self.stop_methods = stop_methods

        self.stopper = [self.get_stopper(method) for method in stop_methods] if stop_methods else []

    @staticmethod
    def get_stopper(method):
        """  获取早停实例

        :param method:  str, 早停的方法. "Valid|Response|acc|40|max|0.01".
        :return:
        """

        value_tag, patience, mode, min_step = method.split(",")
        stopper = EarlyStopping(int(patience), mode, float(min_step))
        return value_tag, stopper

    @property
    def early_stop(self):
        """ 是否早停

        所有的早停方法都早停了, 则返回True
        :return:
        """

        if not self.stop_methods:
            return False

        for _, stopper in self.stopper:
            if not stopper.early_stop:
                return False
        return True

    def __call__(self, value_dict):
        """记录一次指标值，并判断是否早停"""

        if not self.stop_methods:
            return False

        for value_tag, stopper in self.stopper:
            stopper(value_dict[value_tag])

    def __bool__(self):
        return True

