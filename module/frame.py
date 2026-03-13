#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/4 5:13

from functools import reduce
import random
import typing as t

import h2o
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import TensorDataset, DataLoader

from module.preprocess import Scale

__all__ = ["GsFrame", "TorchFrame", "DLFrame"]


class GsFrame(object):
    """ 将input数据转换成GsML格式的数据"""

    def __init__(self, dataset_list=None, feature_list=None, axis=0):

        self.axis = axis
        self.dataset = self._dataset(dataset_list)
        self.feature = self._feature(feature_list, axis=axis)
        self.data = self._data()

    @property
    def c_dataset(self):
        return list(self.dataset.columns)

    @property
    def c_features(self):

        return [c for c in self.feature.columns if c != "SampleID"]

    @property
    def samples(self):

        return list(self.data["SampleID"])

    @property
    def as_pd(self):
        return self.data.copy()

    @property
    def as_h2o(self):
        col_types = {c: "float" for c in self.c_features}
        data = h2o.H2OFrame(self.data.copy(), column_types=col_types)
        return data

    @property
    def as_tensor_by_dann(self):
        """ 将数据转换成tensor格式"""

        df_data = self.data.copy()
        X = torch.sentor(df_data[self.c_features].values, dtype=torch.float32)
        y = torch.sentor(df_data["Response"].values, dtype=torch.float32)
        d = torch.sentor(df_data["Domain"].values, dtype=torch.float32)
        return X, y, d

    @staticmethod
    def _dataset(dataset_list):

        if dataset_list:
            df_dataset = pd.concat([pd.read_csv(f, sep="\t", low_memory=False) for f in dataset_list], ignore_index=True, sort=False)
        else:
            df_dataset = pd.DataFrame(columns=["SampleID", "Response"])
        return df_dataset

    @staticmethod
    def _feature(feature_list, axis):
        if feature_list and axis == 0:
            df_feature = pd.concat([pd.read_csv(f, low_memory=False) for f in feature_list], ignore_index=True, sort=False)
        elif feature_list and axis == 1:
            df_feature = reduce(lambda x, y: pd.merge(x, y, on="SampleID", how="outer"), [pd.read_csv(f) for f in feature_list])
        else:
            df_feature = pd.DataFrame(columns=["SampleID"])
        return df_feature

    def _data(self):
        """合并info和feature结果。作为最原始的数据集"""

        if len(self.feature) and len(self.dataset):
            df_final = pd.merge(self.feature, self.dataset, on="SampleID", how="inner")
        elif len(self.feature):
            df_final = self.feature.copy()
        elif len(self.dataset):
            df_final = self.dataset.copy()
        else:
            df_final = pd.DataFrame()

        return df_final


class DLFrame(object):
    """生成深度学习数据集

    cols会有多列，则会生成多个数据集，对应不同的loss算法。
    首先会记录每个cols对应非skip行的索引，
    每一个batch_size 则会分别去从各自索引取相同数量的索引。然后去重后生成一个特征矩阵

    """

    def __init__(self, scaler: Scale):

        self.scaler = scaler

        self.index_list = []  # 记录每个cols对应非skip行的索引
        self.start_index_list = []  # 每次取数据时的起始索引

        self.y_list = []  # 记录每个cols对应的Y值
        self.features = None  # 特征矩阵
        self.batch_size_list = []  # batch_size列表
        self.cols = []  # 记录列名

    def data_loader(self, dataset: pd.DataFrame, feature: pd.DataFrame, cols: list, batch_size_list: list):

        assert len(set(dataset.SampleID) - set(feature.SampleID)) == 0, "训练集样本特征缺失"
        assert len(cols) == len(batch_size_list), "cols和batch_size_list长度不一致"

        self.batch_size_list = batch_size_list
        self.cols = cols  # 列明

        # 合并数据集
        df_ss = pd.merge(dataset, feature, on="SampleID", how="inner")
        df_ss = df_ss.sample(frac=1).reset_index(drop=True)  # 打乱数据

        feature = self.scaler.transform(df_ss)
        self.features = torch.tensor(feature, dtype=torch.float32)  # 保存特征矩阵

        # 生成索引列表
        self.start_index_list = [0 for _ in cols]
        for col in cols:
            self.index_list.append(df_ss[df_ss[col] != "skip"].index.tolist())
            y = self.scaler.to_one_hot(df_ss, col)
            y = torch.tensor(y, dtype=torch.float32)
            self.y_list.append(y)

    def __iter__(self):
        return self

    def __next__(self):
        """生成一个batch的数据.
        对每个cols，取一个batch_size的数据，然后合并成一个batch_size的数据。
        当第一个col，也就是Response迭代完。任务终止
        """

        index_list = []
        for i in range(len(self.start_index_list)):
            start = self.start_index_list[i]
            end = start + self.batch_size_list[i]
            # 如果end超过了索引列表的长度，则重新打乱索引列表
            if end > len(self.index_list[i]):
                if i == 0:
                    # 重置
                    self.start_index_list = [0 for _ in self.cols]
                    # random.shuffle(self.index_list[0])
                    raise StopIteration

                random.shuffle(self.index_list[i])
                start = 0
                end = self.batch_size_list[i]
            index = self.index_list[i][start: end]
            index_list.extend(index)
            self.start_index_list[i] = end  # 更新起始索引
        index_list = list(set(index_list))
        return self.features[index_list], [y[index_list] for y in self.y_list], self.cols



    # def data_loader(self, dataset: pd.DataFrame, feature: pd.DataFrame, cols: list, batch_size_list: list, shuffle: True):
    #     """生成数据加载器
    #
    #     遍历所有要转换的列，然后生成对应的数据加载器
    #
    #     :param dataset: 数据集
    #     :param feature: 特征集
    #     :param cols: 需要转换的列.
    #     :param batch_size_list: batch_size列表
    #     :param shuffle: 是否打乱数据
    #     """
    #
    #     data_loader_list = []
    #     for col, batch_size in zip(cols, batch_size_list):
    #         X = self.scaler.transform(feature)
    #         Y = self.scaler.to_one_hot(dataset[dataset[col] != "skip"], col)  # skip标签说明该样本不参与这个算法的训练
    #         data = TensorDataset(X, Y)
    #         data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    #         data_loader_list.append(data_loader)
    #     return data_loader_list


class TorchFrame(object):
    """将数据转换成pytorch适用的格式"""

    def __init__(self):

        self.is_fit = False  # 是否已经fit过
        self.imputer = None  # NaN填充实例
        self.scaler = None  # 数据缩放实例
        self.features = None  # 所有特征名的集合
        self.classes = dict()  # 所有类别名的集合. {"Response": ["Cancer", "Healthy"], "Domain": ["D1", "D2"]...}

    def fit(self, df_feature: pd.DataFrame, df_dataset: pd.DataFrame = None, class_cols: list = None,
            scale_method: list = None, na_strategy: list = None):
        """ fit数据，获取特征名和类别名

        :param df_feature: train数据集的特征，用于fit缩放和填充实例
        :param df_dataset: train数据集信息，用于获取类别名
        :param class_cols: 需要存储类别信息的类。一般为Response，DANN模型需要Domain列
        :param scale_method: 特征缩放方法。 [minmax]
        :param na_strategy: na填充策略。 [mean]
        :return:
        """

        # 特征及特征处理实例fit
        self.features = [c for c in df_feature.columns if c != "SampleID"]
        if scale_method and na_strategy:
            self.imputer = SimpleImputer(strategy=na_strategy)
            self.scaler = self._get_scaler(scale_method)
            self.imputer.fit(df_feature[self.features])
            self.scaler.fit(self.imputer.transform(df_feature[self.features]))
        elif na_strategy:
            self.imputer = SimpleImputer(strategy=na_strategy)
            self.imputer.fit(df_feature[self.features])
        elif scale_method:
            self.scaler = self._get_scaler(scale_method)
            self.scaler.fit(df_feature[self.features])

        # 类别名
        if class_cols:
            for col in class_cols:
                self.classes[col] = df_dataset[col].unique().tolist()

        self.is_fit = True

    def transform_x(self, df_feature: pd.DataFrame):
        """特征值转换。填充缩放"""

        assert self.is_fit, "fit方法未执行"

        if self.imputer and self.scaler:
            X = self.scaler.transform(self.imputer.transform(df_feature[self.features]))
        elif self.imputer:
            X = self.imputer.transform(df_feature[self.features])
        elif self.scaler:
            X = self.scaler.transform(df_feature[self.features])
        else:
            X = df_feature[self.features]
        return X

    def transform_y(self, df_dataset: pd.DataFrame, class_cols: list):
        """ 类别值转换。one-hot编码

        :param df_dataset: 数据集
        :param class_cols: 待转换的列名
        :return:
        """
        from pandas.api.types import CategoricalDtype

        assert self.is_fit, "fit方法未执行"

        rslt = []
        for col in class_cols:
            assert col in df_dataset.columns, f"{col} not in df_dataset"
            assert col in self.classes, f"{col} not in self.classes. {self.classes.keys()}"
            if set(df_dataset[col].unique()) != set(self.classes[col]):
                print(f"{col}类别数不一致。{self.classes[col]}")

            df_dataset[col] = df_dataset[col].astype(CategoricalDtype(self.classes[col], ordered=True))
            rslt.append(pd.get_dummies(df_dataset[col]).astype(int))

        return rslt

    def create_tensor_dataset(self, df_feature: pd.DataFrame, df_dataset: pd.DataFrame, class_cols: list):
        """将数据转换成tensor格式"""

        df_data = pd.merge(df_feature, df_dataset, on="SampleID", how="inner")
        df_data["Positive"] = df_data["Response"].apply(lambda x: 1 if x == "Cancer" else 0)  # 记录阴阳性

        X = self.transform_x(df_data[self.features])
        X = torch.tensor(X, dtype=torch.float32)

        Y_list = self.transform_y(df_data, class_cols)
        Y_list = [torch.tensor(y.values, dtype=torch.float32) for y in Y_list]
        Y_list.append(torch.tensor(df_data["Positive"].values, dtype=torch.float32))

        dataset = TensorDataset(X, *Y_list)
        return dataset

    @staticmethod
    def _get_scaler(scale_method):
        if scale_method == "minmax":
            return preprocessing.MinMaxScaler()
        else:
            raise ValueError(f"scale_method must be in ['minmax']")


if __name__ == '__main__':

    f_feature = "/dssg/home/sheny/MyProject/gsml/demo/feature.csv"
    f_ds = f"/dssg/home/sheny/MyProject/gsml/demo/dataset.tsv"

    frame = GsDLFrame(dataset_list=[f_ds], feature_list=[f_feature])
    print(f"scaler algo: {frame.scaler.algo}")
    frame.fit(scale_method="minmax", na_strategy="mean")
    print(f"scaler algo: {frame.scaler.algo}")
    print(frame.data)
