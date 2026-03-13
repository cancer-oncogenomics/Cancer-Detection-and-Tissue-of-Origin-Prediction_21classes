#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/4 14:40
# @Author  : shenny
# @File    : cmd_hyper_tuning.py
# @Software: PyCharm


import click

from module.hyper_tuning import HyperDL


__all__ = ["cli_hyper_tuning"]


@click.group()
def cli_hyper_tuning():
    pass


@cli_hyper_tuning.command("HyperDL")
@click.option("--f_model", help="模型路径，如果是模型继续训练，需要使用这个参数")
@click.option("--f_train", required=True, help="训练数据集")
@click.option("--f_target", help="训练数据集")
@click.option("--f_valid_list", required=True, help="验证数据集", multiple=True)
@click.option("--f_feature", required=True, help="测试数据集")
@click.option("--f_hyper_params", required=True, help="超参数文件")
@click.option("--metric", default="ray_metric", show_default=True, help=f"监控指标.")
@click.option("--mode", default="max", type=click.Choice(["min", "max"]), show_default=True, help="mode of hyper tuning")
@click.option("--num_samples", default=100000, type=click.INT, show_default=True, help="最大实验次数")
@click.option("--time_budget_s", default=360000, type=click.INT, show_default=True, help="最大实验时间")
@click.option("--d_output", required=True, help="实验名称")
@click.option("--prefix", required=True, help="实验名称")
def cmd_hyper_dl(**kwargs):
    """DANN模型超参数调优"""

    tun = HyperDL()
    tun.run(**kwargs)
