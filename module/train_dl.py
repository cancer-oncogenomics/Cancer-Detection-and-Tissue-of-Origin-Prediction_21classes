#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/19 下午8:03
# @Author  : shenny
# @File    : train_dl.py
# @Software: PyCharm

"""深度学习模型训练"""

import os.path

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from ray import train as ray_train
import joblib

from estimators.cnn import get_model
from module.early_stop import MultiEarlyStopping
from module import log
from module.metrics import Metrics
from module.optimizer import get_optimizer
from module.opt_scheduler import get_scheduler
from module.loss import get_loss
from module.loss_weight_scheduler import get_loss_weight_scheduler
from module.load_model import load_model


def get_train_algo(algo):
    """获取训练实例"""

    if algo == "default":
        return DLTrain
    elif algo == "dann":
        return DANNTrain
    elif algo == "ccsa":
        return CCSATrain
    else:
        log.error("训练算法选择错误", 1)
        raise ValueError("训练算法选择错误")


class DLTrain(object):
    """ 深度学习模型训练基本类

    :param f_train: 训练集
    :param valid_dict: 验证集字典
    :param f_feature: 特征文件
    :param f_model: 模型文件
    :param config: 模型训练参数
    :param d_output: 输出目录
    :param prefix: 输出文件前缀
    """

    def __init__(self, config: dict, f_train: str, f_feature: str, f_model=None, valid_dict: dict = None,
                 d_output: str = None, prefix: str = None, **kwargs):

        self.config = config
        self.f_train = f_train
        self.valid_dict = valid_dict
        self.f_feature = f_feature
        self.d_output = self.outdir(d_output) if d_output else None
        self.prefix = prefix

        self.model = self.init_model(f_model)  # 模型实例
        self.scaler = self.model.scaler

        self.output_metric = self.config.get("output_metric")

    def run(self, ds_train=None, ds_valid_list=None, hyper=False, **kwargs):
        """模型训练"""

        if not ds_train:
            ds_train = self.get_dataset(self.f_train)
        if not ds_valid_list:
            ds_valid_list = [(name, self.get_dataset(file)) for name, file in self.valid_dict.items()]

        # 模型训练
        self.train(ds_train, ds_valid_list, hyper=hyper)

        # 保存模型
        self.save()

    def train(self, ds_train, ds_valid_list, verbose=1, hyper=False, **kwargs):
        """默认训练方法"""

        device = self.device
        conf = self.config

        # 初始化一个模型，这个模型是根据模型参数生成的
        params = {k.split("__")[-1]: v for k, v in conf.items() if k.startswith("m_update__")}
        model = self.model.model(update_params=params)
        model.to(device)

        optimizer = self.get_optimizer(model)  # 设置优化器
        opt_scheduler = self.get_scheduler(optimizer)  # 设置优化器调度器
        loss_func = self.get_loss_func()  # 设置损失函数
        early_stopper = self.get_early_stopper()  # 设置早停监控

        # 开启训练
        batch_size = conf["batch_size"]
        epochs = conf["num_epochs"]
        metrics = Metrics(epoch=0)

        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader_list = [(name, DataLoader(ds_valid, batch_size=batch_size, shuffle=False)) for name, ds_valid in ds_valid_list]

        # 训练
        for epoch in range(epochs):
            model.train()
            for X, Y in train_loader:
                X, Y = X.to(device), Y.to(device)
                X = X.unsqueeze(1)
                optimizer.zero_grad()  # 梯度清零
                output, _ = model(X)  # 前向传播
                loss = loss_func(output, Y)
                loss.backward()
                optimizer.step()
                metrics("Train", Y, output, loss.item())

            # 验证
            model.eval()
            with torch.no_grad():

                for index, (name, valid_loader) in enumerate(valid_loader_list):
                    val_loss = 0
                    for X, Y in valid_loader:
                        X, Y = X.to(device), Y.to(device)
                        X = X.unsqueeze(1)
                        output, _ = model(X)
                        loss_batch = loss_func(output, Y)
                        metrics(name, Y, output, loss_batch.item())
                        val_loss += loss_batch.item()

                    # 更新学习率
                    if index == 0 and conf["lr_algo"] == "reducelr":
                        avg_val_loss = val_loss / len(valid_loader)
                        opt_scheduler.step(epoch=epoch, metrics=avg_val_loss)

            # 更新以epoch为单位的指标
            if conf["lr_algo"] == "steplr":
                opt_scheduler.step(epoch=epoch)

            # 计算指标
            metrics.next_epoch(epoch=epoch + 1)
            epoch_metric = metrics.report_metric()
            if verbose <= 1:
                print(metrics)
            if hyper:
                ray_metric = 0
                for out_metric in conf.get("output_metric", "").split("|"):
                    metric, _, mode = out_metric.rsplit("_", 2)
                    value = epoch_metric[metric]
                    ray_metric += value if model == "max" else (1 - value)
                epoch_metric["ray_metric"] = ray_metric
                ray_train.report(metrics=epoch_metric)

            # 早停
            early_stopper(epoch_metric)
            if early_stopper.early_stop:
                log.info("早停", verbose)
                break

        # 记录模型参数
        model = model.to("cpu")
        self.model.last_model_state_dict = model.state_dict()
        self.model.metrics = metrics

    @property
    def input_size(self):
        """模型输入大小"""

        df_features = pd.read_csv(self.f_feature, nrows=2)
        return df_features.shape[1] - 1

    def init_model(self, f_model=None):
        """初始化一个模型
        否则会根据网络结构参数，生成一个新的模型，并且做好数据标准化等工作
        """

        # 如果提供了模型路径，那就载入模型
        if f_model:
            log.info("载入已有模型", 1)
            model = load_model(f_model=f_model)
        else:
            df_features = pd.read_csv(self.f_feature, nrows=2)
            input_size = df_features.shape[1] - 1

            model = get_model(
                input_size=input_size,
                num_class=self.config["num_classes"],
                network=self.config["network"],
                params={k.split("__")[-1]: v for k, v in self.config.items() if k.startswith("model__")},
                na_strategy=self.config["na_strategy"],
                scale_algo=self.config["scale_algo"]
            )

            # 设置标准化实例
            df_train = pd.read_csv(self.f_train, sep="\t")
            df_feature = pd.read_csv(self.f_feature)
            df_feature = df_feature[df_feature.SampleID.isin(df_train.SampleID)]
            model.scaler.fit(df_feature)

            # Response标签顺序固定
            _ = model.scaler.to_one_hot(df_train, col="Response")

        return model

    def get_optimizer(self, model, other_params=None):
        """ 获取优化器"""

        frozen_layers = []
        for k, v in self.config.items():
            if k.startswith("frozen_") and v:
                frozen_layers.append(k.replace("frozen_", ""))

        optimizer = get_optimizer(
            model=model,
            algo=self.config["lr_algo"],
            weight_decay=self.config["lr_weight_decay"],
            lr=self.config["lr"],
            frozen_layers=frozen_layers,
            other_params=other_params
        )
        return optimizer

    def get_scheduler(self, optimizer):
        """ 获取优化器调度器

        :param optimizer: 优化器
        :return:
        """

        scheduler = get_scheduler(
            algo=self.config["lr_scheduler_algo"],
            optimizer=optimizer,
            step_size=self.config["steplr_step_size"],
            gamma=self.config["steplr_gamma"],
            mode=self.config["reducelr_mode"]
        )
        return scheduler

    def get_loss_func(self, algo=None):
        """ 获取损失函数

        :return:
        """

        algo = algo or self.config["loss_func"]
        loss_func = get_loss(algo)
        return loss_func

    def get_loss_weight_scheduler(self):
        """ 获取损失函数权重调度器"""

        loss_weight_scheduler = get_loss_weight_scheduler(
            weights=[float(i) for i in self.config["loss_weights"].split(",")],
            algo=self.config["lw_algo"],
            end_weights=[float(i) for i in self.config["lw_end_weights"].split(",")],
            num_epochs=self.config["num_epochs"]
        )
        return loss_weight_scheduler

    def get_early_stopper(self):
        """生成一个早停监控实例"""

        early_stop_strategy = self.config.get("early_stop_strategy", "").split("__")
        return MultiEarlyStopping(early_stop_strategy)

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def outdir(path):
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def is_model_pass(self):
        """模型性能是否通过"""

        metrics = self.config.get("output_metric", "").split("|")
        if metrics:
            for metric in metrics:
                dataset, metric, threshold, mode = metric.split("_")
                threshold = float(threshold)

                value = self.model.metrics.get_metric(dataset, metric)
                if mode == "max" and value < threshold:
                    return False
                if mode == "min" and value > threshold:
                    return False
            return True

    def get_dataset(self, file):
        """ 将数据集转换为TensorDataset

        :param file:
        :return:
        """

        df_feature = pd.read_csv(self.f_feature)
        df_ss = pd.read_csv(file, sep="\t")
        assert set(df_ss.SampleID) - set(df_feature.SampleID) == set(), "特征不全"

        df_ss = pd.merge(df_ss, df_feature, on="SampleID", how="left")
        X_train = torch.tensor(self.scaler.transform(df_ss), dtype=torch.float32)
        Y_train = torch.tensor(self.scaler.to_one_hot(df_ss, col="Response"), dtype=torch.float32)
        ds_train = TensorDataset(X_train, Y_train)
        return ds_train

    def save(self):
        """保存模型结果"""

        # 保存模型训练结果
        model_state_dict = self.model.last_model_state_dict
        self.model.model_state_dict = model_state_dict  # 模型参数更新为最后一次的参数
        self.model.metrics.df_stat.to_csv(f"{self.d_output}/{self.prefix}.Metrics.tsv", index=False, sep="\t")
        df_config = pd.DataFrame([self.config])
        df_config.to_csv(f"{self.d_output}/{self.prefix}.config.tsv", index=False, sep="\t")

        # 如果模型性能通过，则保存模型
        if self.is_model_pass:
            log.info("模型性能通过，保存模型", 1)

            self.model.d_output = self.d_output
            self.model.prefix = self.prefix
            torch.save(self.model.model_state_dict, f"{self.d_output}/{self.prefix}.model.state_dict")
            torch.save(self.model.model(), f"{self.d_output}/{self.prefix}.model")
            joblib.dump(self.model, f"{self.d_output}/{self.prefix}.gsml")

            df_feature = pd.read_csv(self.f_feature)
            df_score = self.model.predict(df_feature)
            df_score.to_csv(f"{self.d_output}/{self.prefix}.Predict.tsv", index=False, sep="\t")

class DANNTrain(DLTrain):
    """ DANN模型训练

    :param f_target: 目标域数据集

    """

    def __init__(self, f_target: str = None, **kwargs):
        super(DANNTrain, self).__init__(**kwargs)

        self.f_target = f_target

    def run(self, ds_train=None, ds_valid_list=None, ds_target=None, hyper=False):
        """模型训练"""

        # 生成数据集
        if not ds_train:
            ds_train = self.get_dataset(self.f_train)
        if not ds_valid_list:
            ds_valid_list = [(name, self.get_dataset(file)) for name, file in self.valid_dict.items()]
        if not ds_target:
            ds_target = self.get_dataset(self.f_target)

            # 模型训练
            self.train(ds_train=ds_train, ds_valid_list=ds_valid_list, ds_target=ds_target, hyper=hyper)

            # 保存模型
            self.save()

    def train(self, ds_train, ds_valid_list, verbose=1, hyper=False, ds_target=None):
        """ DANN训练方法"""

        log.info("DANN训练", verbose)
        device = self.device
        conf = self.config

        # 源域模型
        params = {k.split("__")[-1]: v for k, v in conf.items() if k.startswith("m_update__")}
        model = self.model.model(update_params=params)
        model.to(device)

        # 目标域模型和参数
        dann_alpha = conf["dann_alpha"]
        t_model = get_model(
            input_size=model.feature_size()[-1],
            num_class=self.config["num_classes"],
            network=self.config["m2_network"],
            params={k.split("__")[-1]: v for k, v in self.config.items() if k.startswith("m2__")},
            na_strategy="none",
            scale_algo="none"
        )
        t_model = t_model.model()
        t_model.to(device)
        t_parameters = t_model.parameters()

        optimizer = self.get_optimizer(model, other_params=t_parameters)  # 设置优化器
        opt_scheduler = self.get_scheduler(optimizer)  # 设置优化器调度器
        loss_func = self.get_loss_func()  # 设置损失函数
        early_stopper = self.get_early_stopper()  # 设置早停监控
        lw_scheduler = self.get_loss_weight_scheduler()          # 设置损失函数权重调度器

        # 开启训练
        batch_size = conf["batch_size"]
        epochs = conf["num_epochs"]
        metrics = Metrics(epoch=0)

        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
        target_loader = DataLoader(ds_target, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader_list = [(name, DataLoader(ds_valid, batch_size=batch_size, shuffle=False)) for name, ds_valid in ds_valid_list]

        # 训练
        weight_list = [float(i) for i in conf["loss_weights"].split(",")]  # 损失函数权重
        for epoch in range(epochs):
            model.train()

            for (source_data, source_label), (target_data, _) in zip(train_loader, target_loader):

                # 特征转换
                source_data, source_label = source_data.to(device), source_label.to(device)
                target_data = target_data.to(device)
                source_data, target_data = source_data.unsqueeze(1), target_data.unsqueeze(1)

                optimizer.zero_grad()

                # 提取标签和特征
                class_output, [*_, source_feature] = model(source_data)
                _, [*_, target_feature] = model(target_data)

                # 分类损失
                class_loss = loss_func(class_output, source_label)

                # 域损失
                combined_feature = torch.cat([source_feature, target_feature], 0)
                label1 = torch.stack([torch.tensor([0.0, 1.0]) for _ in range(source_data.size(0))])
                label2 = torch.stack([torch.tensor([1.0, 0.0]) for _ in range(target_data.size(0))])
                domain_labels = torch.cat([label1, label2], 0).to(device)
                domain_output, _ = t_model(combined_feature, reverse=False, alpha=dann_alpha)
                domain_loss = loss_func(domain_output, domain_labels)

                # 总损失
                loss = weight_list[0] * class_loss + weight_list[1] * domain_loss

                # 反向传播
                loss.backward()
                optimizer.step()

                # 损失函数权重调度
                weight_list = lw_scheduler.update_weights(
                    current_losses=[class_loss.item(), domain_loss.item()],
                    epoch=epoch + 1
                )

                # 报告结果
                metrics("Train", source_label, class_output, loss.item())
                metrics("Train_Domain", domain_labels, domain_output, domain_loss.item())
            log.info(f"当前损失函数权重: {weight_list}", verbose)

            # 验证
            model.eval()
            with torch.no_grad():

                for index, (name, valid_loader) in enumerate(valid_loader_list):
                    val_loss = 0
                    for X, Y in valid_loader:
                        X, Y = X.to(device), Y.to(device)
                        X = X.unsqueeze(1)
                        output, _ = model(X)
                        loss_batch = loss_func(output, Y)
                        metrics(name, Y, output, loss_batch.item())
                        val_loss += loss_batch.item()

                    # 更新学习率
                    if index == 0 and conf["lr_algo"] == "reducelr":
                        avg_val_loss = val_loss / len(valid_loader)
                        opt_scheduler.step(epoch=epoch, metrics=avg_val_loss)

            # 更新以epoch为单位的指标
            if conf["lr_algo"] == "steplr":
                opt_scheduler.step(epoch=epoch)

            # 计算指标
            metrics.next_epoch(epoch=epoch + 1)
            epoch_metric = metrics.report_metric()
            if verbose <= 1:
                print(metrics)
            if hyper:
                ray_metric = 0
                for out_metric in conf.get("output_metric", "").split("|"):
                    metric, _, mode = out_metric.rsplit("_", 2)
                    value = epoch_metric[metric]
                    ray_metric += value if model == "max" else (1 - value)
                epoch_metric["ray_metric"] = ray_metric
                ray_train.report(metrics=epoch_metric)

            # 早停
            early_stopper(epoch_metric)
            if early_stopper.early_stop:
                log.info("早停", verbose)
                break

        # 记录模型参数
        model = model.to("cpu")
        self.model.last_model_state_dict = model.state_dict()
        self.model.metrics = metrics


class CCSATrain(DLTrain):
    """ CCSA模型训练

    :param f_target: 目标域数据集

    """

    def __init__(self, f_target: str = None, **kwargs):
        super(CCSATrain, self).__init__(**kwargs)

        self.f_target = f_target

    def run(self, ds_train=None, ds_valid_list=None, ds_target=None, hyper=False):
        """模型训练"""

        # 生成数据集
        if not ds_train:
            ds_train = self.get_dataset(self.f_train)
        if not ds_valid_list:
            ds_valid_list = [(name, self.get_dataset(file)) for name, file in self.valid_dict.items()]
        if not ds_target:
            ds_target = self.get_dataset(self.f_target)

            # 模型训练
            self.train(ds_train=ds_train, ds_valid_list=ds_valid_list, ds_target=ds_target, hyper=hyper)

            # 保存模型
            self.save()

    def train(self, ds_train, ds_valid_list, verbose=1, hyper=False, ds_target=None):
        """ CCSA训练方法"""

        log.info("CCSA训练", verbose)
        device = self.device
        conf = self.config

        # 模型
        params = {k.split("__")[-1]: v for k, v in conf.items() if k.startswith("m_update__")}
        model = self.model.model(update_params=params)
        model.to(device)

        optimizer = self.get_optimizer(model)  # 设置优化器
        opt_scheduler = self.get_scheduler(optimizer)  # 设置优化器调度器
        loss_class_func = self.get_loss_func()  # 设置损失函数
        loss_csa_func = self.get_loss_func(algo="csa_loss")  # 设置损失函数
        early_stopper = self.get_early_stopper()  # 设置早停监控
        lw_scheduler = self.get_loss_weight_scheduler()          # 设置损失函数权重调度器

        # 开启训练
        batch_size = conf["batch_size"]
        epochs = conf["num_epochs"]
        metrics = Metrics(epoch=0)

        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
        target_loader = DataLoader(ds_target, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader_list = [(name, DataLoader(ds_valid, batch_size=batch_size, shuffle=False, drop_last=True)) for name, ds_valid in ds_valid_list]

        # 训练
        weight_list = [float(i) for i in conf["loss_weights"].split(",")]  # 损失函数权重
        for epoch in range(epochs):
            model.train()

            for (source_data, source_label), (target_data, target_label) in zip(train_loader, target_loader):

                # 特征转换
                source_data, source_label = source_data.to(device), source_label.to(device)
                target_data, target_label = target_data.to(device), target_label.to(device)
                source_data, target_data = source_data.unsqueeze(1), target_data.unsqueeze(1)

                optimizer.zero_grad()

                # 提取标签和特征
                class_output, [*_, source_feature] = model(source_data)
                _, [*_, target_feature] = model(target_data)

                # 分类损失
                class_loss = loss_class_func(class_output, source_label)

                # csa损失
                ccsa_loss_value = loss_csa_func(source_feature, target_feature, source_label, target_label, conf["ccsa_margin"])

                # 总损失
                loss = weight_list[0] * class_loss + weight_list[1] * ccsa_loss_value

                # 反向传播
                loss.backward()
                optimizer.step()

                # 损失函数权重调度
                weight_list = lw_scheduler.update_weights(
                    current_losses=[class_loss.item(), ccsa_loss_value],
                    epoch=epoch + 1
                )

                # 报告结果
                metrics("Train", source_label, class_output, loss.item())
                metrics("Train_ccsa", source_label, class_output, ccsa_loss_value)
            log.info(f"当前损失函数权重: {weight_list}", verbose)

            # 验证
            model.eval()
            with torch.no_grad():

                for index, (name, valid_loader) in enumerate(valid_loader_list):
                    val_loss = 0
                    for X, Y in valid_loader:
                        X, Y = X.to(device), Y.to(device)
                        X = X.unsqueeze(1)
                        output, _ = model(X)
                        loss_batch = loss_class_func(output, Y)
                        metrics(name, Y, output, loss_batch.item())
                        val_loss += loss_batch.item()

                    # 更新学习率
                    if index == 0 and conf["lr_algo"] == "reducelr":
                        avg_val_loss = val_loss / len(valid_loader)
                        opt_scheduler.step(epoch=epoch, metrics=avg_val_loss)

            # 更新以epoch为单位的指标
            if conf["lr_algo"] == "steplr":
                opt_scheduler.step(epoch=epoch)

            # 计算指标
            metrics.next_epoch(epoch=epoch + 1)
            epoch_metric = metrics.report_metric()
            if verbose <= 1:
                print(metrics)
            if hyper:
                ray_metric = 0
                for out_metric in conf.get("output_metric", "").split("|"):
                    metric, _, mode = out_metric.rsplit("_", 2)
                    value = epoch_metric[metric]
                    ray_metric += value if model == "max" else (1 - value)
                epoch_metric["ray_metric"] = ray_metric
                ray_train.report(metrics=epoch_metric)

            # 早停
            early_stopper(epoch_metric)
            if early_stopper.early_stop:
                log.info("早停", verbose)
                break

        # 记录模型参数
        model = model.to("cpu")
        self.model.last_model_state_dict = model.state_dict()
        self.model.metrics = metrics
