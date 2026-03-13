# gsml-命令行工具

## 1. H2oAutoML

使用H2o的AutoML模块，对数据集进行训练和预测，得到多个base model

+ **快速使用**

    ```bash
    /dssg/home/sheny/software/own/gsml/gsml H2oAutoML \
        --feature /dssg/home/sheny/MyProject/gsml/demo/dataset/Train.cnv.csv \
        --feature /dssg/home/sheny/MyProject/gsml/demo/dataset/Valid1.cnv.csv \
        --feature /dssg/home/sheny/MyProject/gsml/demo/dataset/Valid2.cnv.csv \
        --train_info /dssg/home/sheny/MyProject/gsml/demo/dataset/Train.info.list \
        --pred_info /dssg/home/sheny/MyProject/gsml/demo/dataset/Valid1.info.list \
        --pred_info /dssg/home/sheny/MyProject/gsml/demo/dataset/Valid2.info.list \
        --leaderboard /dssg/home/sheny/MyProject/gsml/demo/dataset/Valid1.info.list \
        --d_output /dssg/home/sheny/test/automl/ \
        --prefix demo
    ```

    

+ **注意事项**

    1. feature可以指定多个，但是必须是同一种feature。（流程内部对feature做按列合并）

+ **参数说明**

    | 参数                       | 类型  | 默认值 | 说明                                                         |
    | -------------------------- | ----- | ------ | ------------------------------------------------------------ |
    | d_output                   | str   | -      | 结果输出目录                                                 |
    | prefix                     | str   | -      | 结果文件的前缀，仅是统计结果文件，模型文件的名称仍旧使用系统自带名称。 |
    | feature                    | str   | -      | 特征文件路径                                                 |
    | train_info                 | str   | -      | train数据集文件路径，文件必须包含Response列                  |
    | pred_info                  | str   | -      | 需要进行预测的数据集文件路径，文件必须包含Response列         |
    | leaderboard                | str   | -      | 用于进行指导的数据集文件路径，文件必须包含Response列         |
    | nthreads                   | int   | 10     | 模型训练使用到的线程数                                       |
    | max_models                 | int   | 200    | 最大训练的base model数量                                     |
    | max_runtime_secs_per_model | int   | 1800   | 单模型最大训练时间（秒）                                     |
    | max_runtime_secs           | int   | 0      | AutoML最大总运行时间（秒），0表示无限制                      |
    | nfolds                     | int   | 5      | 交叉验证的份数                                               |
    | seed                       | int   | -1     | 交叉验证拆分seed                                             |
    | stopping_metric            | str   | aucpr  | 模型终止指标。["AUTO", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr",                    "lift_top_group", "misclassification", "mean_per_class_error", "r2"] |
    | sort_metric                | str   | aucpr  | 模型性能排序指标。["auc", "aucpr", "logloss", "mean_per_class_error", "rmse", "mse"] |
    | stopping_tolerance         | float | 0.001  | 模型终止的最小步长                                           |
    | weights_column             | str   | -      | 样本权重的列名。(列名需要在train info中)                     |

    

## 2. ModelStat(模型统计)

+ **快速使用**

    ```bash
    /dssg/home/sheny/software/own/gsml/gsml ModelStat \
        --f_model demo.gbm.gsml \
        --d_output ./ \
        --model_name test \
        --dataset Train,Train.info.list \
        --dataset Valid1,Valid1.info.list \
        --dataset Valid2,Valid2.info.list \
        --optimize KAG9_new_rep,Optimize_KAG9_new_rep.tsv \
        --optimize KAG9,Optimize_KAG9.tsv \
        --optimize Rep69,Optimize_Rep69.tsv \
        --optimize TubeTest_new,Optimize_TubeTest_new.tsv \
        --optimize TubeTest,Optimize_TubeTest.tsv \
        --optimize YM10,Optimize_YM10.tsv \
        --optimize INR30,Optimize_INR30.tsv \
        --optimize KAG9_new,Optimize_KAG9_new.tsv \
        --optimize KY249_rep,Optimize_KY249_rep.tsv \
        --optimize Rep95,Optimize_Rep95.tsv \
        --optimize TubeTest_NJ-BJ-GZ,Optimize_TubeTest_NJ-BJ-GZ.tsv \
        --optimize Valid1_Partial,Optimize_Valid1_Partial.tsv \
        --cs_conf /dssg/home/sheny/MyProject/gsml/config/combine_score.yaml
    ```

    

+ **注意事项**

    1. 如果不需要统计CombineScore的话，就无需提供--optimize和--cs_conf参数。

+ **参数说明**

    | 参数               | 类型  | 默认值                                                       | 说明                                                         |
    | ------------------ | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | f_model            | str   | -                                                            | 模型路径，后缀为.gsml，后续统计只看模型内部保存的样本预测得分 |
    | f_score            | str   | -                                                            | 预测得分文件路径                                             |
    | d_output           | str   | -                                                            | 结果输出文件路径                                             |
    | model_name         | str   | -                                                            | 模型的名称，对分析没影响，就是输出的表格中会增加ModelID列，用于后续多表合并的 |
    | dataset            | str   | -                                                            | 数据集文件路径和数据集名称。                                 |
    | optimize           | str   | -                                                            | 优化项目文件路径和优化项目名称。                             |
    | cs_conf            | str   | -                                                            | 计算CombineScore时用到的                                     |
    | gs_dataset_conf    | str   | -                                                            | （后面不用了，麻烦）                                         |
    | spec_list          | float | [0.85, 0.9, 0.95]                                            | 按多少特异性去划cutoff，可以指定多个                         |
    | cutoff_dataset     | str   | ["Train"]                                                    | 按那个数据集去划cutoff，可以指定多个                         |
    | skip_auc           | bool  | False                                                        | 不统计auc                                                    |
    | skip_performance   | bool  | False                                                        | 不统计性能指标                                               |
    | skip_combine_score | bool  | False                                                        | 不统计combine score                                          |
    | skip_by_subgroup   | bool  | False                                                        | 不统计各个分组的性能                                         |
    | stat_cols          | str   | Train_Group,Detail_Group,Stage_TNM,<br />Stage_E_L,Sex,Lib,Project,Response | 分组性能统计时，分析的具体的列名信息                         |

    

## 3. Predict(模型预测)

+ **快速使用**

    ```bash
    gsml.py Predict \
        --f_model model_52.gsml \
        --feature cnv.csv \
        --dataset Test.info.list
    ```

+ **注意事项**

+ **参数说明**

    