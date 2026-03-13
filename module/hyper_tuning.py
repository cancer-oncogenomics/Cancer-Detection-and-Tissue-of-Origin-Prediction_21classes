#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/4 13:17
# @Author  : shenny
# @File    : hyper_tuning.py
# @Software: PyCharm

"""超参搜索"""
import os
from functools import partial
import time


import pandas as pd
import ray
from ray import air, tune
from ray import train as ray_train
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import torch
from torch.utils.data import TensorDataset
import yaml

from module.train_dl import DLTrain, get_train_algo
from module.preprocess import MinMaxScale


class HyperDL(object):
    """深度学习通用超参搜索"""

    def __init__(self, num_cpus=1, num_gpus=-1):
        num_gpus = num_gpus if num_gpus != -1 else (1 if torch.cuda.is_available() else 0)
        ray.init(num_cpus=1, num_gpus=num_gpus, ignore_reinit_error=True)

    @staticmethod
    def get_hyper_params(file: str):
        """ 从yaml文件中获取超参搜索的参数，并转换成ray.tune的参数格式
        :param file: str, yaml文件路径
        """

        conf = yaml.load(open(file, "r"), Loader=yaml.FullLoader)
        params = {}
        for k, v in conf.items():
            if v["distribution"] == "uniform":
                params[k] = tune.uniform(v["value"][0], v["value"][1])

            elif v["distribution"] == "loguniform":
                params[k] = tune.loguniform(v["value"][0], v["value"][1])

            elif v["distribution"] == "quniform":
                params[k] = tune.quniform(v["value"][0], v["value"][1], v["step"])

            elif v["distribution"] == "choice":
                params[k] = tune.choice(v["value"])
            else:
                raise ValueError(f"distribution error. {v}")

        return params

    def partial_train(self, **kwargs):
        """ 生成训练函数的偏函数"""

        scaler = MinMaxScale(na_strategy="mean")

        df_feature = pd.read_csv(kwargs["f_feature"], low_memory=False)

        # 训练集准备
        df_train = pd.read_csv(kwargs["f_train"], sep="\t")

        df = df_feature[df_feature.SampleID.isin(df_train.SampleID)]
        scaler.fit(df)


        assert set(df_train.SampleID) - set(df_feature.SampleID) == set(), "训练集中有样本未找到特征"
        df_train = pd.merge(df_train, df_feature, on="SampleID", how="left")
        X_train = torch.tensor(scaler.transform(df_train), dtype=torch.float32)
        Y_train = torch.tensor(scaler.to_one_hot(df_train, col="Response"), dtype=torch.float32)
        ds_train = TensorDataset(X_train, Y_train)

        # 验证集准备
        ds_valid_list = []
        for name, file in kwargs["valid_dict"].items():
            df_valid = pd.read_csv(file, sep="\t")
            assert set(df_valid.SampleID) - set(df_feature.SampleID) == set(), f"{name}集中有样本未找到特征"
            df_valid = pd.merge(df_valid, df_feature, on="SampleID", how="left")
            X_valid = torch.tensor(scaler.transform(df_valid), dtype=torch.float32)
            Y_valid = torch.tensor(scaler.to_one_hot(df_valid, col="Response"), dtype=torch.float32)
            ds_valid = TensorDataset(X_valid, Y_valid)
            ds_valid_list.append((name, ds_valid))

        # 对抗域
        if kwargs["f_target"]:
            df_target = pd.read_csv(kwargs["f_target"], sep="\t")
            assert set(df_target.SampleID) - set(df_feature.SampleID) == set(), "训练集中有样本未找到特征"
            df_target = pd.merge(df_target, df_feature, on="SampleID", how="left")
            X_target = torch.tensor(scaler.transform(df_target), dtype=torch.float32)
            Y_target = torch.tensor(scaler.to_one_hot(df_target, col="Response"), dtype=torch.float32)
            ds_target = TensorDataset(X_target, Y_target)
        else:
            ds_target = None

        # 生成训练函数的偏函数
        func = partial(
            self.train,
            ds_train=ray.put(ds_train),
            ds_valid_list=ray.put(ds_valid_list),
            ds_target=ds_target,
            **kwargs)
        return func


    @staticmethod
    def train(config, ds_train, ds_valid_list, mode, metrics, ds_target=None, **kwargs):
        """ 训练函数"""

        # 训练模型
        try:
            # 前缀加时间戳
            kwargs["prefix"] = f"{kwargs['prefix']}.{time.time()}"
            # dl_train = DLTrain(config=config, **kwargs)
            dl_train = get_train_algo(config["train_algo"])
            dl_train = dl_train(config=config, ds_target=ds_target, **kwargs)
            dl_train.run(ds_train=ray.get(ds_train), ds_valid_list=ray.get(ds_valid_list), hyper=True)

        except Exception as e:
            # 记录异常，但不抛出，以确保Ray Tune继续进行
            print(f"An error occurred: {e}")
            if mode == "max":
                ray_train.report(metrics={metrics: 0})  # 返回默认值，确保继续执行
            else:
                ray_train.report(metrics={metrics: 1})


    def run(self, f_train: str, f_valid_list: str, f_feature: str, f_hyper_params: str,
            metric: str, mode: str, num_samples: int, time_budget_s: int, d_output, prefix, f_model=None, f_target=None):
        """ 使用ray.tune进行超参搜索"""

        valid_dict = {i.split(",")[0]: i.split(",")[1] for i in f_valid_list}

        # 设置超参搜索算法
        search_algo = TuneBOHB(metric=metric, mode=mode)

        # 设置调度器
        scheduler = HyperBandForBOHB(metric=metric, mode=mode)

        # 确定超参搜索空间
        hyper_params = self.get_hyper_params(f_hyper_params)
        # 确定训练函数
        train_func = self.partial_train(
            f_model=f_model,
            f_train=f_train,
            f_target=f_target,
            valid_dict=valid_dict,
            f_feature=f_feature,
            d_output=self.outdir(f"{d_output}/model"),
            prefix=prefix,
            mode=mode,
            metrics=metric
        )

        # 启动超参搜索
        local_dir = self.outdir(f"{d_output}/hyper")
        analysis = tune.run(
            train_func,
            config=hyper_params,
            search_alg=search_algo,
            scheduler=scheduler,
            fail_fast=False,
            raise_on_failed_trial=False,
            resources_per_trial={"gpu": 1, "cpu": 1} if torch.cuda.is_available() else {"cpu": 1},
            num_samples=num_samples,
            time_budget_s=time_budget_s,
            log_to_file=True,
            local_dir=local_dir,
            name=prefix,
            resume="AUTO",
            max_failures=1,
            callbacks=[
                SaveBestModelCallback(metric=metric, mode=mode, filename=f"{local_dir}/best_model_performance.txt"),
                CustomFailureCallback()  # 自定义失败回调
            ]
        )

        # 获取最佳结果
        best_config = analysis.get_best_config(metric=metric, mode="max")
        print(f"best_config: {best_config}")
        with open(f"{local_dir}/best_config.yaml", "w") as fw:
            fw.write(yaml.dump(best_config))

    @staticmethod
    def outdir(path):
        os.makedirs(path, exist_ok=True)
        return path


class SaveBestModelCallback(tune.Callback):
    """保存最佳模型的回调函数"""

    def __init__(self, metric, mode, filename):
        self.metric = metric
        self.mode = mode
        self.best_performance = None
        self.filename = filename

    def on_trial_result(self, iteration, trials, trial, result, **info):
        if self.mode == "max":
            is_better = self.best_performance is None or result[self.metric] > self.best_performance
        else:
            is_better = self.best_performance is None or result[self.metric] < self.best_performance

        if is_better:
            self.best_performance = result[self.metric]
            with open(self.filename, "w") as fw:
                col, data = [], []
                for k, v in result.items():
                    if k != "config":
                        col.append(k)
                        data.append(str(v))
                    else:
                        for k1, v1 in v.items():
                            col.append(k1)
                            data.append(str(v1))
                fw.write("\t".join(col) + "\n")
                fw.write("\t".join(data) + "\n")


from ray.tune import Callback

class CustomFailureCallback(Callback):
    def on_trial_complete(self, iteration, trials, trial, **info):
        if trial.status == "ERROR":
            print(f"Trial {trial.trial_id} failed. Skipping this trial.")
