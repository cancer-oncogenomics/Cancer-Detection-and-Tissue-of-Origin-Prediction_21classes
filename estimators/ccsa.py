#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/28 下午3:46
# @Author  : shenny
# @File    : ccsa.py
# @Software: PyCharm

"""CCSA模型"""

from collections import defaultdict
import os
import typing as t
import tempfile

import numpy as np
import pandas as pd
from ray import train as ray_train
from ray.tune.schedulers import AsyncHyperBandScheduler
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from module.early_stop import MultiEarlyStopping
from module.preprocess import MinMaxScale
from module.frame import TorchFrame
from module.loss import CSALoss
from module import log
from version import __version__


class CCSA(nn.Module):
    """ CCSA模型

    :param input_size: 输入数据的大小
    :param num_class: 分类数
    :param out1: 卷积层1输出通道数
    :param conv1: 卷积层1卷积核大小
    :param pool1: 卷积层1池化核大小
    :param drop1: 卷积层1随机失活率
    :param out2: 卷积层2输出通道数
    :param conv2: 卷积层2卷积核大小
    :param pool2: 卷积层2池化核大小
    :param drop2: 卷积层2随机失活率
    :param fc1: 全连接层1大小
    :param fc2: 全连接层2大小
    :param fc3: 全连接层3大小
    :param fc4: 全连接层4大小
    :param drop3: 全连接层3随机失活率
    """

    def __init__(self, input_size: int, num_class: int,
                 out1: int, conv1: int, pool1: int, drop1: float,
                 out2: int, conv2: int, pool2: int, drop2: float,
                 fc1: int, fc2: int, fc3: int, fc4: int, drop3: float, **kwargs):
        super(CCSA, self).__init__()

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

    def forward(self, x):
        """前向传播"""

        features = self.feature_extractor(x)
        features = features.view(-1, self.fc_input_size)  # 将特征展平

        features2 = self.feature_extractor2(features)  # 特征值
        output = self.class_classifier(features2)  # 分类器
        return output, features2

    def _get_fc_input_size(self, input_size):
        """ 返回特征提取器的输入大小

        :param input_size: 输入数据的大小
        :return: 特征提取器的输入大小
        """

        x = torch.randn(1, 1, input_size)
        x = self.feature_extractor(x)
        return x.shape[1] * x.shape[2]


class GsCCSA(object):
    """封装后的CCSA模型

    主要是为了方便调用，保存模型的基本信息

    :param input_size: 输入数据的大小
    :param num_class: 分类数
    :param out1: 卷积层1输出通道数
    :param conv1: 卷积层1卷积核大小
    :param pool1: 卷积层1池化核大小
    :param drop1: 卷积层1随机失活率
    :param out2: 卷积层2输出通道数
    :param conv2: 卷积层2卷积核大小
    :param pool2: 卷积层2池化核大小
    :param drop2: 卷积层2随机失活率
    :param fc1: 全连接层1大小
    :param fc2: 全连接层2大小
    :param fc3: 全连接层3大小
    :param fc4: 全连接层4大小
    :param drop3: 全连接层3随机失活率
    """

    def __init__(self, verbose=1):
        self.verbose = verbose

        # 模型训练后，需要存的结果
        self.is_trained = False
        self.d_output = None
        self.model_name = None
        self.framer = None
        self.init_params = None  # 模型初始化参数
        self.nfold = None  # 交叉验证折数
        self._score = pd.DataFrame()
        self.metrics = pd.DataFrame()  # 训练过程中的指标

        self.verbose = verbose
        self.gsml_version = __version__  # gsml版本
        self.pytorch_version = torch.__version__  # pytorch版本
        self.algorithm = "Pytorch--CCSA"

    def init_framer(self, f_feature, f_train, scale_method, na_strategy):
        """初始化数据处理器"""

        log.info(f"初始化数据处理器", self.verbose)
        df_feature = pd.read_csv(f_feature, low_memory=False)
        df_train = pd.read_csv(f_train, sep="\t", low_memory=False)

        self.framer = TorchFrame()
        self.framer.fit(df_feature, df_train, ["Response"], scale_method, na_strategy)
        log.info(f"Response: {self.framer.classes['Response']}", self.verbose)

    def train(self, f_feature, f_train, f_valid, f_test, d_output, model_name, init_params: dict, lr: float,
              weight_decay: float, batch_size: int, epochs: int, retrain=False, early_strategies: str = None,
              nfold=5):
        """模型训练"""

        assert self.framer, "数据处理器未初始化"

        self.is_trained = self.is_trained if not retrain else False
        self.d_output = self.d_output or self._outdir(d_output)
        self.model_name = model_name
        self.init_params = init_params
        self.init_params["input_size"] = len(self.framer.features)
        self.init_params["num_class"] = len(self.framer.classes["Response"])

        log.info("读取数据", self.verbose)
        df_feature = pd.read_csv(f_feature, low_memory=False)
        df_train = pd.read_csv(f_train, sep="\t", low_memory=False)
        df_valid = pd.read_csv(f_valid, sep="\t", low_memory=False)
        df_test = pd.read_csv(f_test, sep="\t", low_memory=False)

        log.info("数据处理", self.verbose)
        ds_train = self.framer.create_tensor_dataset(df_feature, df_train, ["Response"])
        ds_valid = self.framer.create_tensor_dataset(df_feature, df_valid, ["Response"])
        ds_test = self.framer.create_tensor_dataset(df_feature, df_test, ["Response"])

        # # todo R01B单独提出来
        # df_train_r01b = df_train[df_train.ProjectID == "R01B"]
        # df_valid_r01b = df_valid[df_valid.ProjectID == "R01B"]
        # ds_train_r01b = self.framer.create_tensor_dataset(df_feature, df_train_r01b, ["Response"])
        # ds_valid_r01b = self.framer.create_tensor_dataset(df_feature, df_valid_r01b, ["Response"])

        log.info("初始化模型训练参数", self.verbose)
        f_model_params = self.f_model_params if self.is_trained and not retrain else None
        f_opt_params = self.f_opt_params if self.is_trained and not retrain else None
        train_args = {"init_params": self.init_params, "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
                      "model_state_dict": f_model_params, "opt_state_dict": f_opt_params, "device": self.device,
                      "epochs": epochs, "early_strategies": early_strategies, "verbose": self.verbose,
                      "ds_test": ds_test
                      }

        # 交叉验证
        if nfold > 1:
            log.info("交叉验证", self.verbose)
            self.train_by_nfold(nfold, f_feature, f_train, f_valid, d_output, train_args)

        # 模型训练
        log.info("模型训练", self.verbose)
        model, optimizer, metrics = train_ccsa(
            ds_train=ds_train, ds_valid=ds_valid, init_params=self.init_params, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, model_state_dict=f_model_params, opt_state_dict=f_opt_params, device=self.device,
            epochs=epochs, early_strategies=early_strategies, ds_test=ds_test, verbose=self.verbose,
        )

        # 保存预测得分
        log.info("保存预测得分", self.verbose)
        self.is_trained = True
        self.predict(model, f_feature=f_feature, pred_type="predict", save=True)
        df_score = self._score.copy()
        if nfold < 1:
            df_score.loc[df_score.SampleID.isin(df_train.SampleID), "PredType"] = "train"
        df_score.to_csv(self.f_predict, sep="\t", index=False)

        # 保存结果
        log.info("保存结果", self.verbose)
        model.to("cpu")
        model.eval()
        torch.save(model, self.f_model)
        torch.save(model.state_dict(), self.f_model_params)
        torch.save(optimizer.state_dict(), self.f_opt_params)
        self.metrics = pd.DataFrame(metrics.data)
        self.metrics.to_csv(self.f_train_process, sep="\t", index=False)

    def predict(self, model=None, f_feature=None, f_output=None, pred_type="predict", save = False):

        # assert self.is_trained, "模型未训练"

        model = model or torch.load(self.f_model)
        model = model.to("cpu")
        model.eval()

        df_feature = pd.read_csv(f_feature, low_memory=False)
        X = self.framer.transform_x(df_feature)
        X = torch.tensor(X, dtype=torch.float32)
        X = X.unsqueeze(1).to("cpu")
        class_output, _ = model(X)
        class_output = nn.Softmax(dim=1)(class_output)
        df_score = pd.DataFrame(class_output.detach().numpy(), columns=self.framer.classes["Response"])
        df_score.insert(0, "SampleID", df_feature.SampleID)
        df_score.insert(1, "PredType", pred_type)
        if "Cancer" in self.framer.classes["Response"]:
            df_score.insert(2, "Score", df_score.Cancer)

        if f_output:
            df_score.to_csv(f_output, sep="\t")

        if save:
            if len(self._score):
                train_ids = list(self._score.loc[self._score.PredType == "train", "SampleID"])
            else:
                train_ids = []
            df_out_train = df_score[~df_score.SampleID.isin(train_ids)].copy()  # 去除训练集中的样本
            self._score = pd.concat([self._score, df_out_train], ignore_index=True, sort=False)
            self._score = self._score.drop_duplicates(subset=["SampleID"], keep="last")  # 去除重复的样本, 保留最后一次预测的结果

        return df_score

    def train_by_nfold(self, nfold, f_feature, f_train, f_valid, d_output, train_args):
        assert nfold > 1, "nfold must > 1"

        df_feature = pd.read_csv(f_feature, low_memory=False)
        df_train = pd.read_csv(f_train, sep="\t", low_memory=False)
        df_valid = pd.read_csv(f_valid, sep="\t", low_memory=False)

        kf_train = KFold(n_splits=nfold, shuffle=True, random_state=42)
        kf_valid = KFold(n_splits=nfold, shuffle=True, random_state=42)

        # 预先生成不同nfold的训练和测试数据
        for i, ((train_index, test_index1), (valid_index, test_index2)) in enumerate(zip(kf_train.split(df_train), kf_valid.split(df_valid))):
            log.info(f"第{i + 1}折交叉验证", self.verbose)

            ds_train = self.framer.create_tensor_dataset(df_feature, df_train.iloc[train_index], ["Response"])
            ds_valid = self.framer.create_tensor_dataset(df_feature, df_valid.iloc[valid_index], ["Response"])
            model, _, _ = train_ccsa(ds_train=ds_train, ds_valid=ds_valid, **train_args)

            with tempfile.NamedTemporaryFile(dir=d_output) as f_nfold_feature:
                df_test = pd.concat([df_train.iloc[test_index1], df_valid.iloc[test_index2]], ignore_index=True, sort=False)
                df_test_feature = df_feature[df_feature.SampleID.isin(df_test.SampleID)]
                df_test_feature.to_csv(f_nfold_feature.name, index=False)
                self.predict(model, f_feature=f_nfold_feature.name, pred_type="train", save=True)

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def f_model(self):
        """模型实例"""

        return f"{self.d_output}/{self.model_name}.model.pt"

    @property
    def f_predict(self):
        """预测得分文件"""

        return f"{self.d_output}/{self.model_name}.Predict.tsv"

    @property
    def f_model_params(self):
        """模型参数文件"""

        return f"{self.d_output}/{self.model_name}.model.params.pt"

    @property
    def f_opt_params(self):
        """优化器参数文件"""

        return f"{self.d_output}/{self.model_name}.opt.params.pt"

    @property
    def f_train_process(self):
        """训练过程文件"""

        return f"{self.d_output}/{self.model_name}.train.process.tsv"

    @staticmethod
    def _outdir(p):
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)
        return p


class Metrics(object):
    """记录模型训练过程中的一些指标

    :param metrics:  需要记录的性能指标，report_metrics时会输出这些指标
    """

    def __init__(self, epoch: int):

        self.metrics = ["loss", "acc", "recall", "f1"] # 所有的指标
        self.data = []  # 所有epoch的指标，包括metrics和tp,fp,tn,fn等

        # 当前epoch的指标
        self.epoch = epoch
        self.epoch_data = dict()  # {"train": {"tp": [1, 2]}}

    def report_metric(self, precision=4):
        """  返回当前epoch的指标

        :param precision: 小数位数
        """
        skip_metrics = ["test__loss"]
        return {k: round(v, precision) if k != "epoch" else v for k, v in self.data[-1].items() if k not in skip_metrics}

    def stat_epoch(self):
        """统计当前epoch的性能"""

        rslt = {"epoch": self.epoch}

        all_tp, all_fp, all_tn, all_fn = 0, 0, 0, 0
        for ds, v in self.epoch_data.items():
            tp = sum(v.get("tp", np.nan))
            fp = sum(v.get("fp", np.nan))
            tn = sum(v.get("tn", np.nan))
            fn = sum(v.get("fn", np.nan))
            rslt[f"{ds}__accuracy"] = (tp + tn) / (tp + fp + tn + fn)
            rslt[f"{ds}__recall"] = tp / (tp + fn)
            rslt[f"{ds}__f1"] = 2 * tp / (2 * tp + fp + fn)
            rslt[f"{ds}__loss"] = np.mean(v.get("loss", np.nan))

            all_tp += tp
            all_fp += fp
            all_tn += tn
            all_fn += fn

        self.data.append(rslt)

    def next_epoch(self, epoch):
        """重置当前epoch的指标"""



        self.stat_epoch()
        self.epoch = epoch
        self.epoch_data = dict()

    def __call__(self, value: t.Union[int, float], name: str, dataset: str):
        """记录当前epoch的指标"""

        if dataset not in self.epoch_data:
            self.epoch_data[dataset] = defaultdict(list)
        self.epoch_data[dataset][name].append(value)


def train_ccsa(ds_train, ds_valid, ds_test, init_params: dict, lr, weight_decay, batch_size, model_state_dict: str = None,
               opt_state_dict: str = None, device: str = None, epochs: int = None, early_strategies: str = None,
               verbose: int = 1, ds_train_r01b=None, ds_valid_r01b=None
               ):
    """  CCSA模型训练

    :param ds_train:  训练集。格式为TensorDataset
    :param ds_valid:  验证集。格式为TensorDataset。这个验证集其实也参与了训练，是与训练集配对训练
    :param init_params:  CCSA模型初始化参数
    :param lr:  学习率
    :param weight_decay: 权重衰减
    :param batch_size: 批次大小
    :param model_state_dict: 模型参数
    :param opt_state_dict: 优化器参数
    :param device: 设备
    :param epochs: 训练轮数
    :param early_strategies: 早停策略
    :param ds_test: 测试集。格式为{"数据集名称": TestDataset}
    :param verbose: 详细程度
    :return:
    """

    # 确认模型,优化器,损失函数
    log.info("初始化模型,优化器,损失函数", verbose)
    model = CCSA(**init_params)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if model_state_dict:
        log.warning(f"加载模型参数: {model_state_dict}", verbose)
        model.load_state_dict(torch.load(model_state_dict, map_location=device))
    if opt_state_dict:
        log.warning(f"加载优化器参数: {opt_state_dict}", verbose)
        optimizer.load_state_dict(torch.load(opt_state_dict, map_location=device))

    loss_func = nn.CrossEntropyLoss()  # 二分类交叉熵
    loss_csa = CSALoss()  # CCSA损失函数

    # 数据加载器
    log.info("初始化数据加载器", verbose)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(ds_valid, batch_size=batch_size, shuffle=True, drop_last=True)
    test_iter = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    train_r01b_loader = DataLoader(ds_train_r01b, batch_size=8, shuffle=True, drop_last=True)
    valid_r01b_loader = DataLoader(ds_valid_r01b, batch_size=8, shuffle=True, drop_last=True)

    # 初始化早停策略
    log.info("初始化早停策略", verbose)
    early_stopper = MultiEarlyStopping(early_strategies) if early_strategies else None

    # 训练模型
    log.info("训练模型", verbose)
    metrics = Metrics(0)

    for epoch in range(epochs):
        model.train()  # 训练模式
        for i, (X, Y, T) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            X = X.unsqueeze(1)  # 增加一个维度
            output, features = model(X)

            # 随机取valid一个batch，计算CCSA损失
            X_valid, Y_valid, _ = next(iter(valid_loader))
            X_valid, Y_valid = X_valid.to(device), Y_valid.to(device)
            X_valid = X_valid.unsqueeze(1)
            output_valid, features_valid = model(X_valid)

            # 计算R01B的CCSA损失
            X_r01b, Y_r01b, _ = next(iter(train_r01b_loader))
            X_r01b, Y_r01b = X_r01b.to(device), Y_r01b.to(device)
            X_r01b = X_r01b.unsqueeze(1)
            output_r01b, features_r01b = model(X_r01b)
            X_valid_r01b, Y_valid_r01b, _ = next(iter(valid_r01b_loader))
            X_valid_r01b, Y_valid_r01b = X_valid_r01b.to(device), Y_valid_r01b.to(device)
            X_valid_r01b = X_valid_r01b.unsqueeze(1)
            output_valid_r01b, features_valid_r01b = model(X_valid_r01b)
            loss_r01b = loss_csa(features_r01b, features_valid_r01b, Y_r01b.argmax(dim=1), Y_valid_r01b.argmax(dim=1))

            # 所有的R01B + 随机20例健康，再计算一个loss
            X_all_r01b = torch.cat([X_r01b, X_valid_r01b, X[0: 20]], dim=0)
            Y_all_r01b = torch.cat([Y_r01b, Y_valid_r01b, Y[0: 20]], dim=0)
            output_all_r01b, features_all_r01b = model(X_all_r01b)
            loss_all_r01b = loss_func(output_all_r01b, Y_all_r01b)

            # 计算损失
            loss_class = loss_func(output, Y)
            loss_csa_value = loss_csa(features, features_valid, Y.argmax(dim=1), Y_valid.argmax(dim=1))
            loss = loss_class + loss_csa_value + loss_r01b * 0.5 + loss_all_r01b

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录指标
            pos_index = torch.nonzero(T).squeeze()
            neg_index = torch.nonzero(1 - T).squeeze()
            pos_index = pos_index if pos_index.ndim else pos_index.unsqueeze(0)
            neg_index = neg_index if neg_index.ndim else neg_index.unsqueeze(0)
            tp = (output[pos_index].argmax(dim=1) == Y[pos_index].argmax(dim=1)).sum().item() if len(
                pos_index) else 0
            tn = (output[neg_index].argmax(dim=1) == Y[neg_index].argmax(dim=1)).sum().item() if len(
                neg_index) else 0
            fp = len(neg_index) - tn
            fn = len(pos_index) - tp

            metrics(loss.item(), "loss", "Train")
            metrics(tp, "tp", "Train")
            metrics(fp, "fp", "Train")
            metrics(tn, "tn", "Train")
            metrics(fn, "fn", "Train")

        # 预测test数据集
        model.eval()
        for i, (X, Y, T) in enumerate(test_iter):
            X, Y = X.to(device), Y.to(device)
            X = X.unsqueeze(1)
            output, features = model(X)

            # 记录指标
            pos_index = torch.nonzero(T).squeeze()
            neg_index = torch.nonzero(1 - T).squeeze()
            pos_index = pos_index if pos_index.ndim else pos_index.unsqueeze(0)
            neg_index = neg_index if neg_index.ndim else neg_index.unsqueeze(0)
            tp = (output[pos_index].argmax(dim=1) == Y[pos_index].argmax(dim=1)).sum().item() if len(
                pos_index) else 0
            tn = (output[neg_index].argmax(dim=1) == Y[neg_index].argmax(dim=1)).sum().item() if len(
                neg_index) else 0
            fp = len(neg_index) - tn
            fn = len(pos_index) - tp

            metrics(tp, "tp", "test")
            metrics(fp, "fp", "test")
            metrics(tn, "tn", "test")
            metrics(fn, "fn", "test")

        # 计算指标
        metrics.next_epoch(epoch=epoch + 1)
        print(metrics.report_metric(4))
        ray_train.report(metrics=metrics.report_metric(4))

        # 早停
        if early_strategies:
            early_stopper(metrics.data[-1])
            if early_stopper.early_stop:
                break

    return model, optimizer, metrics




















