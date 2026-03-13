#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 9:34
# @Author  : shenny
# @File    : preprocess.py
# @Software: PyCharm

"""特征预处理"""

import logging
import typing as t

import coloredlogs
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import torch

from module import log


__all__ = ["get_scaler"]


def get_scaler(algo: str, na_strategy: str):
    """获取标准化实例

    :param algo: 标准化算法
    :param na_strategy: 缺失值填充策略
    """

    log.info(f"选择标准化算法: {algo}, 空值填充策略: {na_strategy}", 1)
    if algo == "minmax":
        return MinMaxScale(na_strategy=na_strategy)
    elif algo == "zscore":
        return ZScoreScale(na_strategy=na_strategy)
    elif algo == "robust":
        return RobustScale(na_strategy=na_strategy)
    elif algo == "maxabs":
        return MaxAbsScale(na_strategy=na_strategy)
    elif algo == "none":
        return FakeScale(na_strategy=na_strategy)


class Scale(object):
    """数据标准化基本类

    主要功能包括:
        1. 缺失值填充
        2. 数据缩放
        3. Y值按固定格式转换为one-hot

    :param na_strategy: 缺失值填充策略. min, max, mean, median, most_frequent, constant(自定义填充值)
    """

    def __init__(self, na_strategy="mean"):
        self.na_strategy = na_strategy

        self.algo = "scale_base"  # 标准化算法
        self.imputer = SimpleImputer(strategy=na_strategy)  # 缺失值填充实例
        self.scaler = preprocessing.MinMaxScaler()  # 数据缩放实例

        self.c_features = []  # 转换时特征顺序
        self.class_tags = {}  # 相关列的类别标签。{"Domain": ["D1", "D2"]...} 最后一个标签一定是skip

    def fit(self, df_data: pd.DataFrame):

        log.info("开始训练数据标准化器", 1)

        self.c_features = [c for c in df_data.columns if c not in ["SampleID"]]  # 保存特征列顺序

        X = df_data[self.c_features].values
        self.imputer.fit(X)
        self.scaler.fit(X)

    def transform(self, df_data: pd.DataFrame):
        """特征缩放"""

        X = df_data[self.c_features].values
        data = self.imputer.transform(X)
        data = self.scaler.transform(data)

        return data

    def to_one_hot(self, df_data: pd.DataFrame, col):
        """  将Y值转换为one-hot编码

        :param df_data: 数据集
        :param col: 需要转换的列
        """

        if col not in self.class_tags:
            tag_list = sorted(df_data[col].unique().tolist())  # 报证Cancer,Healthy的顺序固定

            if "skip" in tag_list:
                tag_list.remove("skip")
                tag_list.append("skip")
            else:
                tag_list.append("skip")
            log.info(f"列{col}的类别标签固定顺序: {tag_list[:-1]}", 1)
            self.class_tags[col] = tag_list

        df_data[col] = df_data[col].astype(CategoricalDtype(self.class_tags[col], ordered=True))
        y = pd.get_dummies(df_data[col])

        # 删除skip列
        y = y.drop(columns=["skip"])
        y = y.values
        return y

    def __bool__(self):
        return True


class MinMaxScale(Scale):
    """最大最小值缩放"""

    def __init__(self, na_strategy="mean"):
        super(MinMaxScale, self).__init__(na_strategy=na_strategy)

        self.algo = "minmax"
        self.scaler = preprocessing.MinMaxScaler()


class ZScoreScale(Scale):
    """Zscore标准化"""

    def __init__(self, na_strategy="mean"):
        super(ZScoreScale, self).__init__(na_strategy=na_strategy)

        self.algo = "zscore"
        self.scaler = preprocessing.StandardScaler()


class RobustScale(Scale):
    """Robust标准化"""

    def __init__(self, na_strategy="mean"):
        super(RobustScale, self).__init__(na_strategy=na_strategy)

        self.algo = "robust"
        self.scaler = preprocessing.RobustScaler()


class MaxAbsScale(Scale):
    """MaxAbs标准化"""

    def __init__(self, na_strategy="mean"):
        super(MaxAbsScale, self).__init__(na_strategy=na_strategy)

        self.algo = "maxabs"
        self.scaler = preprocessing.MaxAbsScaler()


class FakeScale(Scale):
    """生成一个假的标准化类.虽然没有实现数据标准化，但是数据填充还是必须得"""

    def __init__(self, na_strategy="mean"):
        super(FakeScale, self).__init__(na_strategy=na_strategy)

        self.algo = "none"
        self.scaler = None

    def transform(self, df_data: pd.DataFrame):
        """特征缩放"""

        X = df_data[self.c_features].values
        data = self.imputer.transform(X)

        return data

