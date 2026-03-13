#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/17 15:32

import os

import h2o
import pandas as pd

from module import cluster
from module.load_model import load_model
from module.save_model import save_model
from module.frame import GsFrame

__all__ = ["predict"]


def predict(f_model, feature, dataset=None, skip_in_model=False, f_output=None, submit_shell=False,
            precise=False):
    """ 模型得分预测

    :param f_model: 模型路径
    :param feature: 特征文件路径
    :param dataset:  数据集文件路径
    :param skip_in_model: 得分是否保存到模型内部
    :param f_output: 保存得分文件路径
    :param submit_shell: 是否提交到shell
    :param precise: 是否只输出特征文件中的样本得分
    :return:
    """

    prefix = os.path.basename(f_model).replace(".gsml", "")
    model = load_model(f_model, use_predict=True)

    if not model.algorithm.startswith("Pytorch"):
        gf_pred = GsFrame(feature_list=feature, dataset_list=dataset)
        if not submit_shell:
            score = model.predict(predict_frame=gf_pred)
        else:
            score = model.predict(predict_frame=gf_pred, submit="none")

        # 深度学习模型，会返回多个结果，但是第一个肯定是预测得分
        if type(score) == tuple:
            score = score[0]

        if f_output:
            if not precise:
                score.to_csv(f_output, sep="\t", index=False)
            else:
                df_feature = pd.concat([pd.read_csv(i) for i in feature], ignore_index=True, sort=False)
                score[score.SampleID.isin(df_feature.SampleID)].to_csv(f_output, sep="\t", index=False)

        if not skip_in_model:
            save_model(model=model, path=os.path.dirname(f_model), prefix=prefix, skip_h2o=True)
        print(score)

    else:
        df_feature = pd.read_csv(feature[0])
        score = model.predict(df_feature, f_output=f_output, precise=precise)
        print(score)


