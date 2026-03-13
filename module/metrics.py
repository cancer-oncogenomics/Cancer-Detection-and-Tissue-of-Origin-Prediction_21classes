#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/4 5:13

"""模型训练指标"""

import typing as t
import numpy as np
from collections import defaultdict

import pandas as pd


class Metrics(object):
    """记录模型训练过程中的一些指标"""

    def __init__(self, epoch: int):

        self.df_stat = pd.DataFrame()  # 模型的所有指标

        # 当前epoch的指标
        self.epoch = epoch
        self.epoch_data = []


    def report_metric(self):
        """  返回当前epoch的指标
        会以字典的形式输出，主要是用作其他方法的性能指标
        """

        df_stat = self.df_stat.copy()

        # 取最后一个epoch的指标
        epoch = df_stat["epoch"].max()
        df_stat = df_stat[df_stat["epoch"] == epoch]
        df_stat = pd.melt(
            df_stat,
            id_vars=["dataset"],
            value_vars=["loss", "acc", "sens", "spec"],
            var_name="metric",
            value_name="Value"
        ).fillna(0)

        rslt = {f"{s.dataset}_{s.metric}": s.Value for _, s in df_stat.iterrows()}
        return rslt

    def stat_epoch(self):
        """统计当前epoch的性能"""

        df_stat = pd.DataFrame(self.epoch_data)

        # 将epoch每个batch的指标合并
        df_sum = df_stat[df_stat["metric"].isin(["tp", "fp", "tn", "fn"])]
        df_mean = df_stat[df_stat["metric"].isin(["loss"])]

        df_sum = df_sum.groupby(["epoch", "dataset", "metric"])["value"].sum().reset_index()
        df_mean = df_mean.groupby(["epoch", "dataset", "metric"])["value"].mean().reset_index()
        df_stat = pd.concat([df_sum, df_mean], axis=0)

        # 将指标转为列
        df_stat = pd.pivot_table(
            df_stat,
            values='value',
            index=['epoch', 'dataset'],
            columns='metric',
            aggfunc="first").rename_axis(None, axis=1).reset_index()

        # 补全列
        cols = ["tp", "fp", "tn", "fp", "loss"]
        for col in cols:
            if col not in df_stat.columns:
                df_stat[col] = np.nan

        # 统计其他指标
        df_stat["acc"] = (df_stat["tp"] + df_stat["tn"]) / (df_stat["tp"] + df_stat["fp"] + df_stat["tn"] + df_stat["fn"])
        df_stat["f1"] = 2 * df_stat["tp"] / (2 * df_stat["tp"] + df_stat["fp"] + df_stat["fn"])
        df_stat["spec"] = df_stat["tn"] / (df_stat["tn"] + df_stat["fp"])
        df_stat["sens"] = df_stat["tp"] / (df_stat["tp"] + df_stat["fn"])
        df_stat = df_stat[["epoch", "dataset", "loss", "acc", "sens", "spec", "f1", "tp", "fp", "tn", "fn"]]

        self.df_stat = pd.concat([self.df_stat, df_stat], ignore_index=True, sort=False)

    def next_epoch(self, epoch):
        """重置当前epoch的指标"""

        self.stat_epoch()
        self.epoch = epoch
        self.epoch_data = []

    def get_metric(self, dataset, metric):
        """获取一个指标的当前性能"""

        df_stat = self.df_stat.copy()
        value = df_stat.loc[(df_stat.dataset == dataset), metric].iloc[-1]
        return value

    def __call__(self, dataset, Y, target, loss):
        """记录当前epoch的指标"""

        Y = Y.argmax(axis=1)
        target = target.argmax(axis=1)

        tp = ((Y == 0) & (target == 0)).sum().item()
        tn = ((Y == 1) & (target == 1)).sum().item()
        fp = ((Y == 1) & (target == 0)).sum().item()
        fn = ((Y == 0) & (target == 1)).sum().item()

        metrics = [
            {"epoch": self.epoch, "dataset": dataset, "metric": "loss", "value": loss},
            {"epoch": self.epoch, "dataset": dataset, "metric": "tp", "value": tp},
            {"epoch": self.epoch, "dataset": dataset, "metric": "tn", "value": tn},
            {"epoch": self.epoch, "dataset": dataset, "metric": "fp", "value": fp},
            {"epoch": self.epoch, "dataset": dataset, "metric": "fn", "value": fn},
        ]
        self.epoch_data.extend(metrics)

    def __str__(self):

        df_stat = self.df_stat.copy()
        epoch = df_stat["epoch"].max()
        df_stat = df_stat[df_stat["epoch"] == epoch]
        return df_stat.round(4).to_string()
