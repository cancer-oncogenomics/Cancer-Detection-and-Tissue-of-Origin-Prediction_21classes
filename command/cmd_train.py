#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/7/1 21:01
# @File     : cmd_train.py
# @Project  : gsml


"""模型训练"""

import click
import yaml

from module.train import train_h2o_base, train_h2o_stack, train_gs_stack
from module import cluster
from module.train_dl import get_train_algo

__all__ = ["cli_train"]


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli_train():
    """Command line tool for model training"""

    pass


@cli_train.command("Train_H2oDeepLearning", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-r", "--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("-p", "--pred_info",
              required=True,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("-f", "--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("-a", "--prefix",
              required=True,
              default="H2oDeepLearning",
              help="Prefix of output files"
              )
@click.option("-o", "--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("-w", "--weights_column",
              help="The name or index of the column in training_frame that holds per-row weights."
              )
@click.option("-n", "--nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="nfolds"
              )
@click.option("-d", "--fold_assignment",
              type=click.Choice(["auto", "random", "modulo", "stratified"]),
              default="stratified",
              show_default=True,
              help="fold_assignment"
              )
@click.option("-t", "--threads",
              type=click.INT,
              default=10,
              show_default=True,
              help="nthreads"
              )
@click.option("--epochs",
              type=click.INT,
              default=50,
              show_default=True,
              help="How many times the dataset should be iterated (streamed), can be fractional."
              )
@click.option("--reproducible",
              is_flag=True,
              default=True,
              show_default=True,
              help="How many times the dataset should be iterated (streamed), can be fractional."
              )
def cmd_h2o_deep_learning(threads, **kwargs):
    """used H2oDeepLearning algorithm to model training"""

    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")
    try:
        train_h2o_base(algorithm="H2oDeepLearning", **kwargs)
    finally:
        cluster.close()


@cli_train.command("Train_H2OGradientBoosting", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-r", "--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("-p", "--pred_info",
              required=True,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("-f", "--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("-a", "--prefix",
              required=True,
              default="H2OGradientBoosting",
              help="Prefix of output files"
              )
@click.option("-o", "--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("-w", "--weights_column",
              help="The name or index of the column in training_frame that holds per-row weights."
              )
@click.option("-n", "--nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="nfolds"
              )
@click.option("-d", "--fold_assignment",
              type=click.Choice(["auto", "random", "modulo", "stratified"]),
              default="stratified",
              show_default=True,
              help="fold_assignment"
              )
@click.option("-t", "--threads",
              type=click.INT,
              default=10,
              show_default=True,
              help="nthreads"
              )
def cmd_h2o_gbm(threads, **kwargs):
    """used H2OGradientBoosting algorithm to model training"""

    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")
    try:
        train_h2o_base(algorithm="H2OGradientBoosting", **kwargs)
    finally:
        cluster.close()


@cli_train.command("Train_H2OGeneralizedLinear", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-r", "--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("-p", "--pred_info",
              required=True,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("-f", "--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("-a", "--prefix",
              required=True,
              default="H2OGeneralizedLinear",
              help="Prefix of output files"
              )
@click.option("-o", "--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("-w", "--weights_column",
              help="The name or index of the column in training_frame that holds per-row weights."
              )
@click.option("-n", "--nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="nfolds"
              )
@click.option("-d", "--fold_assignment",
              type=click.Choice(["auto", "random", "modulo", "stratified"]),
              default="stratified",
              show_default=True,
              help="fold_assignment"
              )
@click.option("-t", "--threads",
              type=click.INT,
              default=10,
              show_default=True,
              help="nthreads"
              )
def cmd_h2o_glm(threads, **kwargs):
    """used H2OGeneralizedLinear algorithm to model training"""

    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")
    try:
        train_h2o_base(algorithm="H2OGeneralizedLinear", **kwargs)
    finally:
        cluster.close()


@cli_train.command("Train_H2ORandomForest", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-r", "--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("-p", "--pred_info",
              required=True,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("-f", "--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("-a", "--prefix",
              required=True,
              default="H2ORandomForest",
              help="Prefix of output files"
              )
@click.option("-o", "--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("-w", "--weights_column",
              help="The name or index of the column in training_frame that holds per-row weights."
              )
@click.option("-n", "--nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="nfolds"
              )
@click.option("-d", "--fold_assignment",
              type=click.Choice(["auto", "random", "modulo", "stratified"]),
              default="stratified",
              show_default=True,
              help="fold_assignment"
              )
@click.option("-t", "--threads",
              type=click.INT,
              default=10,
              show_default=True,
              help="nthreads"
              )
def cmd_h2o_rm(threads, **kwargs):
    """used H2ORandomForest algorithm to model training"""

    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")
    try:
        train_h2o_base(algorithm="H2ORandomForest", **kwargs)
    finally:
        cluster.close()


@cli_train.command("Train_H2OXGBoost", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-r", "--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("-p", "--pred_info",
              required=True,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("-f", "--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("-a", "--prefix",
              required=True,
              default="H2OXGBoost",
              help="Prefix of output files"
              )
@click.option("-o", "--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("-w", "--weights_column",
              help="The name or index of the column in training_frame that holds per-row weights."
              )
@click.option("-n", "--nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="nfolds"
              )
@click.option("-d", "--fold_assignment",
              type=click.Choice(["auto", "random", "modulo", "stratified"]),
              default="stratified",
              show_default=True,
              help="fold_assignment"
              )
@click.option("-t", "--threads",
              type=click.INT,
              default=10,
              show_default=True,
              help="nthreads"
              )
def cmd_h2o_xgboost(threads, **kwargs):
    """used H2OXGBoost algorithm to model training"""

    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")
    try:
        train_h2o_base(algorithm="H2OXGBoost", **kwargs)
    finally:
        cluster.close()


@cli_train.command("Train_H2OStackedEnsemble", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-r", "--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("-p", "--pred_info",
              required=True,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("-f", "--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("-a", "--prefix",
              required=True,
              default="H2OXGBoost",
              help="Prefix of output files"
              )
@click.option("-o", "--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("-b", "--d_base_models",
              multiple=True,
              show_default=True,
              help="The directory of base models"
              )
@click.option("--model_list",
              multiple=True,
              show_default=True,
              help="The path of base models"
              )
@click.option("-n", "--metalearner_nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="nfolds"
              )
@click.option("-d", "--metalearner_fold_assignment",
              type=click.Choice(["auto", "random", "modulo", "stratified"]),
              default="stratified",
              show_default=True,
              help="fold_assignment"
              )
@click.option("-t", "--threads",
              type=click.INT,
              default=10,
              show_default=True,
              help="nthreads"
              )
@click.option("--metalearner_nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="metalearner_nfolds"
              )
@click.option("--seed",
              type=click.INT,
              default=10,
              show_default=True,
              help="seed"
              )
@click.option("--metalearner_algorithm",
              default="auto",
              show_default=True,
              help="metalearner_algorithm"
              )
def cmd_h2o_stacked(threads, **kwargs):
    """Train by H2OStackedEnsemble"""

    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")
    try:
        train_h2o_stack(**kwargs)
    finally:
        cluster.close()


@cli_train.command("Train_GsStacked", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-r", "--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("-p", "--pred_info",
              required=False,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("-f", "--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("-a", "--prefix",
              required=True,
              default="H2OXGBoost",
              help="Prefix of output files"
              )
@click.option("-o", "--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("-b", "--d_base_models",
              multiple=True,
              show_default=True,
              help="The directory of base models"
              )
@click.option("--model_list",
              multiple=True,
              show_default=True,
              help="The path of base models"
              )
@click.option("-n", "--nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="nfolds. The MEAN algorithm is invalid."
              )
@click.option("-t", "--threads",
              type=click.INT,
              default=4,
              show_default=True,
              help="nthreads"
              )
@click.option("--seed",
              type=click.INT,
              default=10,
              show_default=True,
              help="seed"
              )
@click.option("--metalearner_algorithm",
              default="mean",
              show_default=True,
              type=click.Choice(["mean", "glm"]),
              help="metalearner_algorithm"
              )
@click.option("--re_pred",
              is_flag=True,
              show_default=True,
              help="The base model repredicts each sample"
              )
@click.option("--too_class",
              show_default=True,
              help="Too categories,like 'Breast,Colorectal,Gastric,Liver,Lung'"
              )
def cmd_gs_stacked(threads, **kwargs):
    """Train by Gs Stacked"""

    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")
    try:
        train_gs_stack(**kwargs)
    finally:
        cluster.close()


@cli_train.command("train_dl", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--f_model", help="model")
@click.option("--f_train", help="训练集")
@click.option("--f_target", help="训练集")
@click.option("--f_valid", multiple=True, help="验证集")
@click.option("--f_feature", help="特征文件")
@click.option("--f_config", help="配置文件路径")
@click.option("--d_output", help="结果文件路径")
@click.option("--prefix", help="结果文件前缀")
def cmd_train_dl(f_model, f_train, f_target, f_valid, f_feature, f_config, d_output, prefix):
    """深度学习模型训练"""

    config = yaml.load(open(f_config, "r"), Loader=yaml.FullLoader)
    valid_dict = {data.split(",")[0]: data.split(",")[1] for data in f_valid}

    train = get_train_algo(config["train_algo"])
    train = train(
        config=config,
        f_model=f_model,
        f_train=f_train,
        valid_dict=valid_dict,
        f_feature=f_feature,
        d_output=d_output,
        prefix=prefix,
        f_target=f_target
    )
    train.run()
