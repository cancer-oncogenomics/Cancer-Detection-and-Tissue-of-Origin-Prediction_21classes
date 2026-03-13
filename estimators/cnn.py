#!/usr/bin/env python
# coding: utf-8
# @Time: 2024/7/20 下午6:38
# @Author：shenny
# @File: cnn.py
# @Software: PyCharm

"""卷积神经网络模型"""

import typing as t
from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn

from module.preprocess import get_scaler
from model.model_base import GsModelStat
from module import log
from version import __version__

__all__ = ["get_model", "CNNEstimators"]


def get_model(input_size: int, num_class: int, network: str, params: dict = None, na_strategy="mean", scale_algo=None):
    """获取模型

    :param input_size: int, 输入特征的维度
    :param num_class: int, 分类的类别数
    :param network: str, 模型名称
    :param params: dict, 模型参数
    :param na_strategy: str, 缺失值填充策略
    :param scale_algo: str, 数据标准化算法
    """

    log.info(f"选择模型网络结构: {network}", 1)
    log.info(f"模型参数: {params}", 1)
    log.info(f"数据标准化算法: {scale_algo}", 1)
    log.info(f"数据填充算法: {na_strategy}", 1)

    if network == "cnn_2conv_3fc":
        return CNN2Conv3Fc(input_size, num_class, params, na_strategy, scale_algo)
    elif network == "cnn_2conv_2fc":
        return CNN2Conv2Fc(input_size, num_class, params, na_strategy, scale_algo)
    elif network == "cnn_2conv_1fc":
        return CNN2Conv1Fc(input_size, num_class, params, na_strategy, scale_algo)
    elif network == "cnn_1conv_3fc":
        return CNN1Conv3Fc(input_size, num_class, params, na_strategy, scale_algo)
    elif network == "cnn_1conv_2fc":
        return CNN1Conv2Fc(input_size, num_class, params, na_strategy, scale_algo)
    elif network == "cnn_1conv_1fc":
        return CNN1Conv1Fc(input_size, num_class, params, na_strategy, scale_algo)
    elif network == "cnn_3fc":
        return CNN3Fc(input_size, num_class, params, na_strategy, scale_algo)
    elif network == "cnn_2fc":
        return CNN2Fc(input_size, num_class, params, na_strategy, scale_algo)
    elif network == "cnn_1fc":
        return CNN1Fc(input_size, num_class, params, na_strategy, scale_algo)
    else:
        raise ValueError(f"网络结构{network}不支持")


class CNNEstimators(GsModelStat):
    """ 卷积神经网络基本类

    :param input_size: 输入数据的大小
    :param num_class: 分类的数量
    :param params: 模型参数
    """

    def __init__(self, input_size: int, num_class: int, params: dict = None, na_strategy="mean", scale_algo="none"):
        super(GsModelStat, self).__init__()

        # 基本参数
        self.input_size = input_size
        self.num_class = num_class
        self.params = params or {}
        self.scaler = get_scaler(algo=scale_algo, na_strategy=na_strategy)  # 数据处理器,会根据列名自动处理数据并标准化
        self.d_output = None  # 输出目录, 这个参数在load_model中会根据实际路径进行设置
        self.prefix = None  # 输出文件前缀, 这个参数在load_model中会根据实际路径进行设置

        # 原始模型信息，主要记录神经网络结构和初始化参数
        self.model_state_dict = None  # 模型参数, 模型训练后需要更新这个参数
        self.last_model_state_dict = None  # 模型最后一次训练的得到的参数

        # 模型训练过程中的信息，包括数据预处理
        self.model_state_dict = None  # 模型参数, 模型训练后需要更新这个参数
        self.metrics = {}  # 模型训练loss等的信息

        # 版本信息
        self.gsml_version = __version__  # gsml版本
        self.pytorch_version = torch.__version__  # pytorch版本
        self.algorithm = "Pytorch--cnn_2conv_3fc"

    def predict(self, df_feature: pd.DataFrame, device="cpu", scale=True, f_output=None, precise=True) -> pd.DataFrame:
        """ 模型预测
        是在forward的基础上，增加了样本号，最后Score等信息。使其兼容之前的模块

        :param df_feature:
        :param device:
        :param scale: 是否对数据进行标准化
        :param f_output: 输出文件
        :param precise: 是否精确输出
        :return:
        """

        if scale:
            X = self.scaler.transform(df_feature)
        else:
            X = df_feature[self.scaler.c_features].values

        X = torch.tensor(X, dtype=torch.float32, device=device)
        X = X.unsqueeze(1)  # 增加一个维度
        model = torch.load(f"{self.d_output}/{self.prefix}.model")
        model.eval()
        output = model(X, softmax=True)[0]
        output = output.cpu().detach().numpy()
        df_score = pd.DataFrame(output, columns=self.scaler.class_tags["Response"][:-1])

        if "Cancer" in df_score.columns:
            df_score["Score"] = df_score.apply(lambda x: x.Cancer, axis=1)
        else:
            df_score["Score"] = -1
        df_score.insert(0, "SampleID", df_feature["SampleID"])
        df_score["PredType"] = "predict"

        train_ids = list(self._score.loc[self._score.PredType == "train", "SampleID"])
        df_out_train = df_score[~df_score.SampleID.isin(train_ids)].copy()
        df_all_score = pd.concat([self._score, df_out_train], ignore_index=True, sort=False)
        df_all_score = df_all_score.drop_duplicates(subset=["SampleID"], keep="last")

        if f_output:
            if precise:
                df_score.to_csv(f_output, sep="\t", index=False)
            else:
                df_all_score.to_csv(f_output, sep="\t", index=False)

        if precise:
            return df_score.round(8)
        else:
            return df_all_score.round(8)

    def model(self, update_params: dict = None):
        """初始化一个模型"""

        log.info(f"初始化模型: {self.input_size}, {self.num_class}, {self.params}", 1)
        model = _CNN2Conv3Fc(self.input_size, self.num_class, self.params)

        if self.model_state_dict:  # 如果模型参数存在，则加载模型参数。这样在模型二次训练时，可以继续训练
            model.load_state_dict(self.model_state_dict)
        else:
            self.model_state_dict = model.state_dict()
        return model

    def save_model(self, d_output, prefix, only_stat_dict=False):
        """  保存pytorch模型

        :param d_output: 输出结果
        :param prefix: 文件前缀
        :param only_stat_dict: 是否只保存模型参数
        """

        model = self.model()
        torch.save(model.state_dict(), f"{d_output}/{prefix}_model.pth")
        if not only_stat_dict:
            torch.save(model, f"{d_output}/{prefix}_model.model")

    def __bool__(self):
        return True


class CNN2Conv3Fc(CNNEstimators):
    """ 两层卷积层，三层全连接层的CNN模型

    :param input_size: 输入数据的大小
    :param num_class: 分类的数量
    :param params: 模型参数
    """

    def __init__(self, input_size: int, num_class: int, params: dict = None, na_strategy="mean", scale_algo=None):
        super(CNN2Conv3Fc, self).__init__(input_size, num_class, params, na_strategy, scale_algo)

        self.algorithm = "Pytorch--cnn_2conv_3fc"
        self._score = pd.DataFrame(columns=["SampleID", "PredType", "Score"])

    def model(self, update_params: dict = None):
        """ 初始化一个模型
        逻辑是：
            1. 首先通过原始参数，生成一个最原始的模型
            2. 如果存储了模型参数，则加载模型参数
            3. 如果是分布式训练，第二步想修改某些参数的，就提供update_params参数

        :param update_params: 需要更新的模型参数
        :return:
        """

        model = _CNN2Conv3Fc(self.input_size, self.num_class, self.params)

        if self.model_state_dict:  # 如果模型参数存在，则加载模型参数。这样在模型二次训练时，可以继续训练
            model.load_state_dict(self.model_state_dict)
        else:
            self.model_state_dict = model.state_dict()

        if update_params:
            log.info(f"分步训练，更新模型参数: {update_params}", 1)
            model.update_params(update_params)

        return model


class CNN2Conv2Fc(CNNEstimators):
    """ 两层卷积层，三层全连接层的CNN模型

    :param input_size: 输入数据的大小
    :param num_class: 分类的数量
    :param params: 模型参数
    """

    def __init__(self, input_size: int, num_class: int, params: dict = None, na_strategy="mean", scale_algo=None):
        super(CNN2Conv2Fc, self).__init__(input_size, num_class, params, na_strategy, scale_algo)

        self.algorithm = "Pytorch--cnn_2conv_2fc"
        self._score = pd.DataFrame(columns=["SampleID", "PredType", "Score"])

    def model(self, update_params: dict = None):
        """ 初始化一个模型
        逻辑是：
            1. 首先通过原始参数，生成一个最原始的模型
            2. 如果存储了模型参数，则加载模型参数
            3. 如果是分布式训练，第二步想修改某些参数的，就提供update_params参数

        :param update_params: 需要更新的模型参数
        :return:
        """

        model = _CNN2Conv2Fc(self.input_size, self.num_class, self.params)

        if self.model_state_dict:  # 如果模型参数存在，则加载模型参数。这样在模型二次训练时，可以继续训练
            model.load_state_dict(self.model_state_dict)
        else:
            self.model_state_dict = model.state_dict()

        if update_params:
            log.info(f"分步训练，更新模型参数: {update_params}", 1)
            model.update_params(update_params)

        return model


class CNN2Conv1Fc(CNNEstimators):
    """ 两层卷积层，三层全连接层的CNN模型

    :param input_size: 输入数据的大小
    :param num_class: 分类的数量
    :param params: 模型参数
    """

    def __init__(self, input_size: int, num_class: int, params: dict = None, na_strategy="mean", scale_algo=None):
        super(CNN2Conv1Fc, self).__init__(input_size, num_class, params, na_strategy, scale_algo)

        self.algorithm = "Pytorch--cnn_2conv_1fc"
        self._score = pd.DataFrame(columns=["SampleID", "PredType", "Score"])

    def model(self, update_params: dict = None):
        """ 初始化一个模型
        逻辑是：
            1. 首先通过原始参数，生成一个最原始的模型
            2. 如果存储了模型参数，则加载模型参数
            3. 如果是分布式训练，第二步想修改某些参数的，就提供update_params参数

        :param update_params: 需要更新的模型参数
        :return:
        """

        model = _CNN2Conv1Fc(self.input_size, self.num_class, self.params)

        if self.model_state_dict:  # 如果模型参数存在，则加载模型参数。这样在模型二次训练时，可以继续训练
            model.load_state_dict(self.model_state_dict)
        else:
            self.model_state_dict = model.state_dict()

        if update_params:
            log.info(f"分步训练，更新模型参数: {update_params}", 1)
            model.update_params(update_params)

        return model


class CNN1Conv3Fc(CNNEstimators):
    """ 两层卷积层，三层全连接层的CNN模型

    :param input_size: 输入数据的大小
    :param num_class: 分类的数量
    :param params: 模型参数
    """

    def __init__(self, input_size: int, num_class: int, params: dict = None, na_strategy="mean", scale_algo=None):
        super(CNN1Conv3Fc, self).__init__(input_size, num_class, params, na_strategy, scale_algo)

        self.algorithm = "Pytorch--cnn_1conv_3fc"
        self._score = pd.DataFrame(columns=["SampleID", "PredType", "Score"])

    def model(self, update_params: dict = None):
        """ 初始化一个模型
        逻辑是：
            1. 首先通过原始参数，生成一个最原始的模型
            2. 如果存储了模型参数，则加载模型参数
            3. 如果是分布式训练，第二步想修改某些参数的，就提供update_params参数

        :param update_params: 需要更新的模型参数
        :return:
        """

        model = _CNN1Conv3Fc(self.input_size, self.num_class, self.params)

        if self.model_state_dict:  # 如果模型参数存在，则加载模型参数。这样在模型二次训练时，可以继续训练
            model.load_state_dict(self.model_state_dict)
        else:
            self.model_state_dict = model.state_dict()

        if update_params:
            log.info(f"分步训练，更新模型参数: {update_params}", 1)
            model.update_params(update_params)

        return model


class CNN1Conv2Fc(CNNEstimators):
    """ 两层卷积层，三层全连接层的CNN模型

    :param input_size: 输入数据的大小
    :param num_class: 分类的数量
    :param params: 模型参数
    """

    def __init__(self, input_size: int, num_class: int, params: dict = None, na_strategy="mean", scale_algo=None):
        super(CNN1Conv2Fc, self).__init__(input_size, num_class, params, na_strategy, scale_algo)

        self.algorithm = "Pytorch--cnn_1conv_2fc"
        self._score = pd.DataFrame(columns=["SampleID", "PredType", "Score"])

    def model(self, update_params: dict = None):
        """ 初始化一个模型
        逻辑是：
            1. 首先通过原始参数，生成一个最原始的模型
            2. 如果存储了模型参数，则加载模型参数
            3. 如果是分布式训练，第二步想修改某些参数的，就提供update_params参数

        :param update_params: 需要更新的模型参数
        :return:
        """

        model = _CNN1Conv2Fc(self.input_size, self.num_class, self.params)

        if self.model_state_dict:  # 如果模型参数存在，则加载模型参数。这样在模型二次训练时，可以继续训练
            model.load_state_dict(self.model_state_dict)
        else:
            self.model_state_dict = model.state_dict()

        if update_params:
            log.info(f"分步训练，更新模型参数: {update_params}", 1)
            model.update_params(update_params)

        return model


class CNN1Conv1Fc(CNNEstimators):
    """ 两层卷积层，三层全连接层的CNN模型

    :param input_size: 输入数据的大小
    :param num_class: 分类的数量
    :param params: 模型参数
    """

    def __init__(self, input_size: int, num_class: int, params: dict = None, na_strategy="mean", scale_algo=None):
        super(CNN1Conv1Fc, self).__init__(input_size, num_class, params, na_strategy, scale_algo)

        self.algorithm = "Pytorch--cnn_1conv_1fc"
        self._score = pd.DataFrame(columns=["SampleID", "PredType", "Score"])

    def model(self, update_params: dict = None):
        """ 初始化一个模型
        逻辑是：
            1. 首先通过原始参数，生成一个最原始的模型
            2. 如果存储了模型参数，则加载模型参数
            3. 如果是分布式训练，第二步想修改某些参数的，就提供update_params参数

        :param update_params: 需要更新的模型参数
        :return:
        """

        model = _CNN1Conv1Fc(self.input_size, self.num_class, self.params)

        if self.model_state_dict:  # 如果模型参数存在，则加载模型参数。这样在模型二次训练时，可以继续训练
            model.load_state_dict(self.model_state_dict)
        else:
            self.model_state_dict = model.state_dict()

        if update_params:
            log.info(f"分步训练，更新模型参数: {update_params}", 1)
            model.update_params(update_params)

        return model


class CNN3Fc(CNNEstimators):
    """ 两层卷积层，三层全连接层的CNN模型

    :param input_size: 输入数据的大小
    :param num_class: 分类的数量
    :param params: 模型参数
    """

    def __init__(self, input_size: int, num_class: int, params: dict = None, na_strategy="mean", scale_algo=None):
        super(CNN3Fc, self).__init__(input_size, num_class, params, na_strategy, scale_algo)

        self.algorithm = "Pytorch--cnn_3fc"
        self._score = pd.DataFrame(columns=["SampleID", "PredType", "Score"])

    def model(self, update_params: dict = None):
        """ 初始化一个模型
        逻辑是：
            1. 首先通过原始参数，生成一个最原始的模型
            2. 如果存储了模型参数，则加载模型参数
            3. 如果是分布式训练，第二步想修改某些参数的，就提供update_params参数

        :param update_params: 需要更新的模型参数
        :return:
        """

        model = _CNN3Fc(self.input_size, self.num_class, self.params)

        if self.model_state_dict:  # 如果模型参数存在，则加载模型参数。这样在模型二次训练时，可以继续训练
            model.load_state_dict(self.model_state_dict)
        else:
            self.model_state_dict = model.state_dict()

        if update_params:
            log.info(f"分步训练，更新模型参数: {update_params}", 1)
            model.update_params(update_params)

        return model


class CNN2Fc(CNNEstimators):
    """ 两层卷积层，三层全连接层的CNN模型

    :param input_size: 输入数据的大小
    :param num_class: 分类的数量
    :param params: 模型参数
    """

    def __init__(self, input_size: int, num_class: int, params: dict = None, na_strategy="mean", scale_algo=None):
        super(CNN2Fc, self).__init__(input_size, num_class, params, na_strategy, scale_algo)

        self.algorithm = "Pytorch--cnn_2fc"
        self._score = pd.DataFrame(columns=["SampleID", "PredType", "Score"])

    def model(self, update_params: dict = None):
        """ 初始化一个模型
        逻辑是：
            1. 首先通过原始参数，生成一个最原始的模型
            2. 如果存储了模型参数，则加载模型参数
            3. 如果是分布式训练，第二步想修改某些参数的，就提供update_params参数

        :param update_params: 需要更新的模型参数
        :return:
        """

        model = _CNN2Fc(self.input_size, self.num_class, self.params)

        if self.model_state_dict:  # 如果模型参数存在，则加载模型参数。这样在模型二次训练时，可以继续训练
            model.load_state_dict(self.model_state_dict)
        else:
            self.model_state_dict = model.state_dict()

        if update_params:
            log.info(f"分步训练，更新模型参数: {update_params}", 1)
            model.update_params(update_params)

        return model


class CNN1Fc(CNNEstimators):
    """ 两层卷积层，三层全连接层的CNN模型

    :param input_size: 输入数据的大小
    :param num_class: 分类的数量
    :param params: 模型参数
    """

    def __init__(self, input_size: int, num_class: int, params: dict = None, na_strategy="mean", scale_algo=None):
        super(CNN1Fc, self).__init__(input_size, num_class, params, na_strategy, scale_algo)

        self.algorithm = "Pytorch--cnn_1fc"
        self._score = pd.DataFrame(columns=["SampleID", "PredType", "Score"])

    def model(self, update_params: dict = None):
        """ 初始化一个模型
        逻辑是：
            1. 首先通过原始参数，生成一个最原始的模型
            2. 如果存储了模型参数，则加载模型参数
            3. 如果是分布式训练，第二步想修改某些参数的，就提供update_params参数

        :param update_params: 需要更新的模型参数
        :return:
        """

        model = _CNN1Fc(self.input_size, self.num_class, self.params)

        if self.model_state_dict:  # 如果模型参数存在，则加载模型参数。这样在模型二次训练时，可以继续训练
            model.load_state_dict(self.model_state_dict)
        else:
            self.model_state_dict = model.state_dict()

        if update_params:
            log.info(f"分步训练，更新模型参数: {update_params}", 1)
            model.update_params(update_params)

        return model


class _Model(object):
    def __init__(self, input_size: int, num_class: int, params: dict):
        self.input_size = int(input_size)
        self.num_class = int(num_class)
        self.params = params

    def forward(self, x, softmax=False, reverse=False, alpha=None):
        pass

    def _get_fc_input_size(self, input_size):
        pass

    def update_params(self, params: dict):
        """ 更新模型参数

        :param params: 需要更新的模型参数
        :return:
        """

        pass


class _CNN2Conv3Fc(nn.Module, _Model):
    def __init__(self, input_size: int, num_class: int, params: dict):
        super(_CNN2Conv3Fc, self).__init__()
        self.input_size = int(input_size)
        self.num_class = int(num_class)
        self.params = params

        # 初始化模型结构
        p = SimpleNamespace(**params)

        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=int(p.conv1_out), kernel_size=int(p.conv1_ks), stride=int(p.conv1_st), bias=p.conv1_bias)
        self.conv1_relu = nn.ReLU()
        self.conv1_bn = nn.BatchNorm1d(num_features=int(p.conv1_out))
        self.conv1_drop = nn.Dropout(p.conv1_drop)
        self.conv1_pool = nn.MaxPool1d(kernel_size=int(p.pool1_ks), stride=int(p.pool1_st))

        # 第二层卷积
        self.conv2 = nn.Conv1d(in_channels=int(p.conv1_out), out_channels=int(p.conv2_out), kernel_size=int(p.conv2_ks), stride=int(p.conv2_st), bias=p.conv2_bias)
        self.conv2_relu = nn.ReLU()
        self.conv2_bn = nn.BatchNorm1d(int(p.conv2_out))
        self.conv2_drop = nn.Dropout(p.conv2_drop)
        self.conv2_pool = nn.MaxPool1d(kernel_size=int(p.pool2_ks), stride=int(p.pool2_st))

        # 第一层全连接
        self.fc_input_size = self._get_fc_input_size(int(self.input_size))
        self.fc1 = nn.Linear(int(self.fc_input_size), int(p.fc1_size))
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(p.fc1_drop)

        # 第二层全连接
        self.fc2 = nn.Linear(int(p.fc1_size), int(p.fc2_size))
        self.fc2_relu = nn.ReLU()
        self.fc2_drop = nn.Dropout(p.fc2_drop)

        # 第三层全连接
        self.fc3 = nn.Linear(int(p.fc2_size), int(p.fc3_size))
        self.fc3_relu = nn.ReLU()
        self.fc3_drop = nn.Dropout(p.fc3_drop)

        # 分类器
        self.class_classifier = nn.Linear(int(p.fc3_size), int(self.num_class))

    def forward(self, x, softmax=False, reverse=False, alpha=None):
        """ 前向传播

        :param x: 特征张量
        :param softmax: 是否使用softmax
        :param reverse: 是否梯度反转
        :param alpha: 梯度反转的系数
        :return:
        """

        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)
        x = self.conv1_pool(x)

        x = self.conv2(x)
        x = self.conv2_relu(x)
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        x = self.conv2_pool(x)

        x = x.view(-1, int(self.fc_input_size))
        x = self.fc1(x)
        x = self.fc1_relu(x)
        fc1_features = self.fc1_drop(x)

        x = self.fc2(fc1_features)
        x = self.fc2_relu(x)
        fc2_features = self.fc2_drop(x)

        x = self.fc3(fc2_features)
        x = self.fc3_relu(x)
        fc3_features = self.fc3_drop(x)

        if reverse:
            fc3_features = ReverseLayerF.apply(fc3_features, alpha)
        class_output = self.class_classifier(fc3_features)

        if softmax:
            class_output = nn.functional.softmax(class_output, dim=1)

        return [class_output, [fc1_features, fc2_features, fc3_features]]

    def feature_size(self):
        """全连接层输出的特征数量"""

        size_list = [self.fc1.out_features, self.fc2.out_features, self.fc3.out_features]
        return size_list

    def _get_fc_input_size(self, input_size):
        """ 返回特征提取器的输入大小

        :param input_size: 输入数据的大小
        :return: 特征提取器的输入大小
        """

        x = torch.randn(1, 1, input_size)
        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)
        x = self.conv1_pool(x)

        x = self.conv2(x)
        x = self.conv2_relu(x)
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        x = self.conv2_pool(x)
        return x.shape[1] * x.shape[2]


class _CNN2Conv2Fc(nn.Module, _Model):
    def __init__(self, input_size: int, num_class: int, params: dict):
        super(_CNN2Conv2Fc, self).__init__()
        self.input_size = int(input_size)
        self.num_class = int(num_class)
        self.params = params

        # 初始化模型结构
        p = SimpleNamespace(**params)

        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=int(p.conv1_out), kernel_size=int(p.conv1_ks), stride=int(p.conv1_st), bias=p.conv1_bias)
        self.conv1_relu = nn.ReLU()
        self.conv1_bn = nn.BatchNorm1d(num_features=int(p.conv1_out))
        self.conv1_drop = nn.Dropout(p.conv1_drop)
        self.conv1_pool = nn.MaxPool1d(kernel_size=int(p.pool1_ks), stride=int(p.pool1_st))

        # 第二层卷积
        self.conv2 = nn.Conv1d(in_channels=int(p.conv1_out), out_channels=int(p.conv2_out), kernel_size=int(p.conv2_ks), stride=int(p.conv2_st), bias=p.conv2_bias)
        self.conv2_relu = nn.ReLU()
        self.conv2_bn = nn.BatchNorm1d(int(p.conv2_out))
        self.conv2_drop = nn.Dropout(p.conv2_drop)
        self.conv2_pool = nn.MaxPool1d(kernel_size=int(p.pool2_ks), stride=int(p.pool2_st))

        # 第一层全连接
        self.fc_input_size = self._get_fc_input_size(int(self.input_size))
        self.fc1 = nn.Linear(int(self.fc_input_size), int(p.fc1_size))
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(p.fc1_drop)

        # 第二层全连接
        self.fc2 = nn.Linear(int(p.fc1_size), int(p.fc2_size))
        self.fc2_relu = nn.ReLU()
        self.fc2_drop = nn.Dropout(p.fc2_drop)

        # 分类器
        self.class_classifier = nn.Linear(int(p.fc2_size), int(self.num_class))

    def forward(self, x, softmax=False, reverse=False, alpha=None):
        """ 前向传播

        :param x: 特征张量
        :param softmax: 是否使用softmax
        :param reverse: 是否梯度反转
        :param alpha: 梯度反转的系数
        :return:
        """

        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)
        x = self.conv1_pool(x)

        x = self.conv2(x)
        x = self.conv2_relu(x)
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        x = self.conv2_pool(x)

        x = x.view(-1, int(self.fc_input_size))
        x = self.fc1(x)
        x = self.fc1_relu(x)
        fc1_features = self.fc1_drop(x)

        x = self.fc2(fc1_features)
        x = self.fc2_relu(x)
        fc2_features = self.fc2_drop(x)

        if reverse:
            fc2_features = ReverseLayerF.apply(fc2_features, alpha)
        class_output = self.class_classifier(fc2_features)

        if softmax:
            class_output = nn.functional.softmax(class_output, dim=1)

        return [class_output, [fc1_features, fc2_features]]

    def feature_size(self):
        """全连接层输出的特征数量"""

        size_list = [self.fc1.out_features, self.fc2.out_features]
        return size_list

    def _get_fc_input_size(self, input_size):
        """ 返回特征提取器的输入大小

        :param input_size: 输入数据的大小
        :return: 特征提取器的输入大小
        """

        x = torch.randn(1, 1, input_size)
        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)
        x = self.conv1_pool(x)

        x = self.conv2(x)
        x = self.conv2_relu(x)
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        x = self.conv2_pool(x)
        return x.shape[1] * x.shape[2]


class _CNN2Conv1Fc(nn.Module, _Model):
    def __init__(self, input_size: int, num_class: int, params: dict):
        super(_CNN2Conv1Fc, self).__init__()
        self.input_size = int(input_size)
        self.num_class = int(num_class)
        self.params = params

        # 初始化模型结构
        p = SimpleNamespace(**params)

        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=int(p.conv1_out), kernel_size=int(p.conv1_ks), stride=int(p.conv1_st), bias=p.conv1_bias)
        self.conv1_relu = nn.ReLU()
        self.conv1_bn = nn.BatchNorm1d(num_features=int(p.conv1_out))
        self.conv1_drop = nn.Dropout(p.conv1_drop)
        self.conv1_pool = nn.MaxPool1d(kernel_size=int(p.pool1_ks), stride=int(p.pool1_st))

        # 第二层卷积
        self.conv2 = nn.Conv1d(in_channels=int(p.conv1_out), out_channels=int(p.conv2_out), kernel_size=int(p.conv2_ks), stride=int(p.conv2_st), bias=p.conv2_bias)
        self.conv2_relu = nn.ReLU()
        self.conv2_bn = nn.BatchNorm1d(int(p.conv2_out))
        self.conv2_drop = nn.Dropout(p.conv2_drop)
        self.conv2_pool = nn.MaxPool1d(kernel_size=int(p.pool2_ks), stride=int(p.pool2_st))

        # 第一层全连接
        self.fc_input_size = self._get_fc_input_size(int(self.input_size))
        self.fc1 = nn.Linear(int(self.fc_input_size), int(p.fc1_size))
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(p.fc1_drop)

        # 分类器
        self.class_classifier = nn.Linear(int(p.fc1_size), int(self.num_class))

    def forward(self, x, softmax=False, reverse=False, alpha=None):
        """ 前向传播

        :param x: 特征张量
        :param softmax: 是否使用softmax
        :param reverse: 是否梯度反转
        :param alpha: 梯度反转的系数
        :return:
        """

        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)
        x = self.conv1_pool(x)

        x = self.conv2(x)
        x = self.conv2_relu(x)
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        x = self.conv2_pool(x)

        x = x.view(-1, int(self.fc_input_size))
        x = self.fc1(x)
        x = self.fc1_relu(x)
        fc1_features = self.fc1_drop(x)

        if reverse:
            fc1_features = ReverseLayerF.apply(fc1_features, alpha)
        class_output = self.class_classifier(fc1_features)

        if softmax:
            class_output = nn.functional.softmax(class_output, dim=1)

        return [class_output, [fc1_features]]

    def feature_size(self):
        """全连接层输出的特征数量"""

        size_list = [self.fc1.out_features]
        return size_list

    def _get_fc_input_size(self, input_size):
        """ 返回特征提取器的输入大小

        :param input_size: 输入数据的大小
        :return: 特征提取器的输入大小
        """

        x = torch.randn(1, 1, input_size)
        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)
        x = self.conv1_pool(x)

        x = self.conv2(x)
        x = self.conv2_relu(x)
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        x = self.conv2_pool(x)
        return x.shape[1] * x.shape[2]


class _CNN1Conv3Fc(nn.Module, _Model):
    def __init__(self, input_size: int, num_class: int, params: dict):
        super(_CNN1Conv3Fc, self).__init__()
        self.input_size = int(input_size)
        self.num_class = int(num_class)
        self.params = params

        # 初始化模型结构
        p = SimpleNamespace(**params)

        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=int(p.conv1_out), kernel_size=int(p.conv1_ks), stride=int(p.conv1_st), bias=p.conv1_bias)
        self.conv1_relu = nn.ReLU()
        self.conv1_bn = nn.BatchNorm1d(num_features=int(p.conv1_out))
        self.conv1_drop = nn.Dropout(p.conv1_drop)
        self.conv1_pool = nn.MaxPool1d(kernel_size=int(p.pool1_ks), stride=int(p.pool1_st))

        # 第一层全连接
        self.fc_input_size = self._get_fc_input_size(int(self.input_size))
        self.fc1 = nn.Linear(int(self.fc_input_size), int(p.fc1_size))
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(p.fc1_drop)

        # 第二层全连接
        self.fc2 = nn.Linear(int(p.fc1_size), int(p.fc2_size))
        self.fc2_relu = nn.ReLU()
        self.fc2_drop = nn.Dropout(p.fc2_drop)

        # 第三层全连接
        self.fc3 = nn.Linear(int(p.fc2_size), int(p.fc3_size))
        self.fc3_relu = nn.ReLU()
        self.fc3_drop = nn.Dropout(p.fc3_drop)

        # 分类器
        self.class_classifier = nn.Linear(int(p.fc3_size), int(self.num_class))

    def forward(self, x, softmax=False, reverse=False, alpha=None):
        """ 前向传播

        :param x: 特征张量
        :param softmax: 是否使用softmax
        :param reverse: 是否梯度反转
        :param alpha: 梯度反转的系数
        :return:
        """

        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)
        x = self.conv1_pool(x)

        x = x.view(-1, int(self.fc_input_size))
        x = self.fc1(x)
        x = self.fc1_relu(x)
        fc1_features = self.fc1_drop(x)

        x = self.fc2(fc1_features)
        x = self.fc2_relu(x)
        fc2_features = self.fc2_drop(x)

        x = self.fc3(fc2_features)
        x = self.fc3_relu(x)
        fc3_features = self.fc3_drop(x)

        if reverse:
            fc3_features = ReverseLayerF.apply(fc3_features, alpha)
        class_output = self.class_classifier(fc3_features)

        if softmax:
            class_output = nn.functional.softmax(class_output, dim=1)

        return [class_output, [fc1_features, fc2_features, fc3_features]]

    def feature_size(self):
        """全连接层输出的特征数量"""

        size_list = [self.fc1.out_features, self.fc2.out_features, self.fc3.out_features]
        return size_list

    def _get_fc_input_size(self, input_size):
        """ 返回特征提取器的输入大小

        :param input_size: 输入数据的大小
        :return: 特征提取器的输入大小
        """

        x = torch.randn(1, 1, input_size)
        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)
        x = self.conv1_pool(x)
        return x.shape[1] * x.shape[2]


class _CNN1Conv2Fc(nn.Module, _Model):
    def __init__(self, input_size: int, num_class: int, params: dict):
        super(_CNN1Conv2Fc, self).__init__()
        self.input_size = int(input_size)
        self.num_class = int(num_class)
        self.params = params

        # 初始化模型结构
        p = SimpleNamespace(**params)

        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=int(p.conv1_out), kernel_size=int(p.conv1_ks),
                               stride=int(p.conv1_st), bias=p.conv1_bias)
        self.conv1_relu = nn.ReLU()
        self.conv1_bn = nn.BatchNorm1d(num_features=int(p.conv1_out))
        self.conv1_drop = nn.Dropout(p.conv1_drop)
        self.conv1_pool = nn.MaxPool1d(kernel_size=int(p.pool1_ks), stride=int(p.pool1_st))

        # 第一层全连接
        self.fc_input_size = self._get_fc_input_size(int(self.input_size))
        self.fc1 = nn.Linear(int(self.fc_input_size), int(p.fc1_size))
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(p.fc1_drop)

        # 第二层全连接
        self.fc2 = nn.Linear(int(p.fc1_size), int(p.fc2_size))
        self.fc2_relu = nn.ReLU()
        self.fc2_drop = nn.Dropout(p.fc2_drop)

        # 分类器
        self.class_classifier = nn.Linear(int(p.fc2_size), int(self.num_class))

    def forward(self, x, softmax=False, reverse=False, alpha=None):
        """ 前向传播

        :param x: 特征张量
        :param softmax: 是否使用softmax
        :param reverse: 是否梯度反转
        :param alpha: 梯度反转的系数
        :return:
        """

        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)
        x = self.conv1_pool(x)

        x = x.view(-1, int(self.fc_input_size))
        x = self.fc1(x)
        x = self.fc1_relu(x)
        fc1_features = self.fc1_drop(x)

        x = self.fc2(fc1_features)
        x = self.fc2_relu(x)
        fc2_features = self.fc2_drop(x)

        if reverse:
            fc2_features = ReverseLayerF.apply(fc2_features, alpha)
        class_output = self.class_classifier(fc2_features)

        if softmax:
            class_output = nn.functional.softmax(class_output, dim=1)

        return [class_output, [fc1_features, fc2_features]]

    def feature_size(self):
        """全连接层输出的特征数量"""

        size_list = [self.fc1.out_features, self.fc2.out_features]
        return size_list

    def _get_fc_input_size(self, input_size):
        """ 返回特征提取器的输入大小

        :param input_size: 输入数据的大小
        :return: 特征提取器的输入大小
        """

        x = torch.randn(1, 1, input_size)
        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)
        x = self.conv1_pool(x)
        return x.shape[1] * x.shape[2]


class _CNN1Conv1Fc(nn.Module, _Model):
    def __init__(self, input_size: int, num_class: int, params: dict):
        super(_CNN1Conv1Fc, self).__init__()
        self.input_size = int(input_size)
        self.num_class = int(num_class)
        self.params = params

        # 初始化模型结构
        p = SimpleNamespace(**params)

        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=int(p.conv1_out), kernel_size=int(p.conv1_ks),
                               stride=int(p.conv1_st), bias=p.conv1_bias)
        self.conv1_relu = nn.ReLU()
        self.conv1_bn = nn.BatchNorm1d(num_features=int(p.conv1_out))
        self.conv1_drop = nn.Dropout(p.conv1_drop)
        self.conv1_pool = nn.MaxPool1d(kernel_size=int(p.pool1_ks), stride=int(p.pool1_st))

        # 第一层全连接
        self.fc_input_size = self._get_fc_input_size(int(self.input_size))
        self.fc1 = nn.Linear(int(self.fc_input_size), int(p.fc1_size))
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(p.fc1_drop)

        # 分类器
        self.class_classifier = nn.Linear(int(p.fc1_size), int(self.num_class))

    def forward(self, x, softmax=False, reverse=False, alpha=None):
        """ 前向传播

        :param x: 特征张量
        :param softmax: 是否使用softmax
        :return:
        """

        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)
        x = self.conv1_pool(x)

        x = x.view(-1, int(self.fc_input_size))
        x = self.fc1(x)
        x = self.fc1_relu(x)
        fc1_features = self.fc1_drop(x)

        if reverse:
            fc1_features = ReverseLayerF.apply(fc1_features, alpha)
        class_output = self.class_classifier(fc1_features)

        if softmax:
            class_output = nn.functional.softmax(class_output, dim=1)

        return [class_output, [fc1_features]]

    def feature_size(self):
        """全连接层输出的特征数量"""

        size_list = [self.fc1.out_features]
        return size_list

    def _get_fc_input_size(self, input_size):
        """ 返回特征提取器的输入大小

        :param input_size: 输入数据的大小
        :return: 特征提取器的输入大小
        """

        x = torch.randn(1, 1, input_size)
        x = self.conv1(x)
        x = self.conv1_relu(x)
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)
        x = self.conv1_pool(x)
        return x.shape[1] * x.shape[2]


class _CNN3Fc(nn.Module, _Model):
    def __init__(self, input_size: int, num_class: int, params: dict):
        super(_CNN3Fc, self).__init__()
        self.input_size = int(input_size)
        self.num_class = int(num_class)
        self.params = params

        # 初始化模型结构
        p = SimpleNamespace(**params)

        # 第一层全连接
        self.fc1 = nn.Linear(int(self.input_size), int(p.fc1_size))
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(p.fc1_drop)

        # 第二层全连接
        self.fc2 = nn.Linear(int(p.fc1_size), int(p.fc2_size))
        self.fc2_relu = nn.ReLU()
        self.fc2_drop = nn.Dropout(p.fc2_drop)

        # 第三层全连接
        self.fc3 = nn.Linear(int(p.fc2_size), int(p.fc3_size))
        self.fc3_relu = nn.ReLU()
        self.fc3_drop = nn.Dropout(p.fc3_drop)

        # 分类器
        self.class_classifier = nn.Linear(int(p.fc3_size), int(self.num_class))

    def forward(self, x, softmax=False, reverse=False, alpha=None):
        """ 前向传播

        :param x: 特征张量
        :param softmax: 是否使用softmax
        :param reverse: 是否梯度反转
        :param alpha: 梯度反转的系数
        :return:
        """

        x = x.view(-1, int(self.input_size))
        x = self.fc1(x)
        x = self.fc1_relu(x)
        fc1_features = self.fc1_drop(x)

        x = self.fc2(fc1_features)
        x = self.fc2_relu(x)
        fc2_features = self.fc2_drop(x)

        x = self.fc3(fc2_features)
        x = self.fc3_relu(x)
        fc3_features = self.fc3_drop(x)

        if reverse:
            fc3_features = ReverseLayerF.apply(fc3_features, alpha)
        class_output = self.class_classifier(fc3_features)

        if softmax:
            class_output = nn.functional.softmax(class_output, dim=1)

        return [class_output, [fc1_features, fc2_features, fc3_features]]

    def feature_size(self):
        """全连接层输出的特征数量"""

        size_list = [self.fc1.out_features, self.fc2.out_features, self.fc3.out_features]
        return size_list


class _CNN2Fc(nn.Module, _Model):
    def __init__(self, input_size: int, num_class: int, params: dict):
        super(_CNN2Fc, self).__init__()
        self.input_size = int(input_size)
        self.num_class = int(num_class)
        self.params = params

        # 初始化模型结构
        p = SimpleNamespace(**params)

        # 第一层全连接
        self.fc1 = nn.Linear(int(self.input_size), int(p.fc1_size))
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(p.fc1_drop)

        # 第二层全连接
        self.fc2 = nn.Linear(int(p.fc1_size), int(p.fc2_size))
        self.fc2_relu = nn.ReLU()
        self.fc2_drop = nn.Dropout(p.fc2_drop)

        # 分类器
        self.class_classifier = nn.Linear(int(p.fc2_size), int(self.num_class))

    def forward(self, x, softmax=False, reverse=False, alpha=None):
        """ 前向传播

        :param x: 特征张量
        :param softmax: 是否使用softmax
        :param reverse: 是否梯度反转
        :param alpha: 梯度反转的系数
        :return:
        """

        x = x.view(-1, int(self.input_size))
        x = self.fc1(x)
        x = self.fc1_relu(x)
        fc1_features = self.fc1_drop(x)

        x = self.fc2(fc1_features)
        x = self.fc2_relu(x)
        fc2_features = self.fc2_drop(x)

        if reverse:
            fc2_features = ReverseLayerF.apply(fc2_features, alpha)
        class_output = self.class_classifier(fc2_features)

        if softmax:
            class_output = nn.functional.softmax(class_output, dim=1)

        return [class_output, [fc1_features, fc2_features]]

    def feature_size(self):
        """全连接层输出的特征数量"""

        size_list = [self.fc1.out_features, self.fc2.out_features]
        return size_list


class _CNN1Fc(nn.Module, _Model):
    def __init__(self, input_size: int, num_class: int, params: dict):
        super(_CNN1Fc, self).__init__()
        self.input_size = int(input_size)
        self.num_class = int(num_class)
        self.params = params

        # 初始化模型结构
        p = SimpleNamespace(**params)

        # 第一层全连接
        self.fc1 = nn.Linear(int(self.input_size), int(p.fc1_size))
        self.fc1_relu = nn.ReLU()
        self.fc1_drop = nn.Dropout(p.fc1_drop)

        # 分类器
        self.class_classifier = nn.Linear(int(p.fc1_size), int(self.num_class))

    def forward(self, x, softmax=False, alpha: float = None, reverse=False):
        """ 前向传播

        :param x: 特征张量
        :param softmax: 是否使用softmax
        :param alpha: 梯度反转的参数
        :param reverse: 是否梯度反转
        :return:
        """

        x = x.view(-1, int(self.input_size))
        x = self.fc1(x)
        x = self.fc1_relu(x)
        fc1_features = self.fc1_drop(x)

        # 梯度反转
        if reverse:
            fc1_features = ReverseLayerF.apply(fc1_features, alpha)
        class_output = self.class_classifier(fc1_features)

        if softmax:
            class_output = nn.functional.softmax(class_output, dim=1)

        return [class_output, [fc1_features]]

    def feature_size(self):
        """全连接层输出的特征数量"""

        size_list = [self.fc1.out_features]
        return size_list


class Lung(CNNEstimators):
    """肺癌模型"""

    def __init__(self, input_size: int, num_class: int, params: dict = None, na_strategy="mean", scale_algo=None):
        super(Lung, self).__init__(input_size, num_class, params, na_strategy, scale_algo)

        self.algorithm = "Pytorch--lung"
        self._score = pd.DataFrame(columns=["SampleID", "PredType", "Score"])

    def model(self, update_params: dict = None):
        """ 初始化一个模型
        逻辑是：
            1. 首先通过原始参数，生成一个最原始的模型
            2. 如果存储了模型参数，则加载模型参数
            3. 如果是分布式训练，第二步想修改某些参数的，就提供update_params参数

        :param update_params: 需要更新的模型参数
        :return:
        """

        model = _Lung(self.input_size, self.num_class, **self.params)

        if self.model_state_dict:  # 如果模型参数存在，则加载模型参数。这样在模型二次训练时，可以继续训练
            model.load_state_dict(self.model_state_dict)
        else:
            self.model_state_dict = model.state_dict()

        if update_params:
            log.info(f"分步训练，更新模型参数: {update_params}", 1)
            model.update_params(update_params)

        return model


class _Lung(nn.Module):
    """肺癌模型"""

    def __init__(self, input_size: int, num_class: int,
                 out1: int, conv1: int, pool1: int, drop1: float,
                 out2: int, conv2: int, pool2: int, drop2: float,
                 fc1: int, fc2: int, fc3: int, fc4: int, drop3: float, **kwargs):
        super(_Lung, self).__init__()

        input_size = int(input_size)
        num_class = int(num_class)
        out1 = int(out1)
        conv1 = int(conv1)
        pool1 = int(pool1)
        drop1 = float(drop1)
        out2 = int(out2)
        conv2 = int(conv2)
        pool2 = int(pool2)
        drop2 = float(drop2)
        fc1 = int(fc1)
        fc2 = int(fc2)
        fc3 = int(fc3)
        fc4 = int(fc4)
        drop3 = float(drop3)

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=out1, kernel_size=conv1, stride=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(out1),  # 规范化激活，将激活值规范到均值为0，方差为1的正态分布
            nn.Dropout(drop1),  # 随机失活，防止过拟合
            nn.MaxPool1d(kernel_size=pool1, stride=2),  # 最大池化，降低维度

            nn.Conv1d(in_channels=out1, out_channels=out2, kernel_size=conv2, stride=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(out2),
            nn.Dropout(drop2),
            nn.MaxPool1d(kernel_size=pool2, stride=2),
        )

        self.fc_input_size = self._get_fc_input_size(input_size)  # 特征提取器的输入大小
        self.feature_extractor2 = nn.Sequential(
            nn.Linear(self.fc_input_size, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
        )

        # 分类器
        self.class_classifier = nn.Sequential(
            nn.Linear(fc2, fc3),
            nn.ReLU(),
            nn.Dropout(drop3),
            nn.Linear(fc3, fc4),
            nn.Linear(fc4, num_class),
        )

    def forward(self, x, softmax=False, alpha: float = None, reverse=False):
        """前向传播"""

        features = self.feature_extractor(x)
        features = features.view(-1, self.fc_input_size)  # 将特征展平

        features2 = self.feature_extractor2(features)  # 特征值

        # 梯度反转
        if reverse:
            features2 = ReverseLayerF.apply(features2, alpha)
        class_output = self.class_classifier(features2)

        if softmax:
            class_output = nn.functional.softmax(class_output, dim=1)

        return [class_output, [features2]]

    def _get_fc_input_size(self, input_size):
        """ 返回特征提取器的输入大小

        :param input_size: 输入数据的大小
        :return: 特征提取器的输入大小
        """

        x = torch.randn(1, 1, input_size)
        x = self.feature_extractor(x)
        return x.shape[1] * x.shape[2]


class ReverseLayerF(torch.autograd.Function):
    """反向传播时，将梯度反转的函数"""

    @staticmethod
    def forward(ctx, x, alpha):
        """前向传播"""

        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        output = grad_output.neg() * ctx.alpha

        return output, None