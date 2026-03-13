# 神经网络超参搜索

1. **预训练模型超参**

    多个超参任务，不要放在同一个目录下.

    cln19环境有点问题。

    ```bash
    /dssg/home/sheny/anaconda3/envs/gsml/bin/python /dssg/NGSPipeline/Mercury/gsml_DL/gsml HyperDL \
        --f_train /dssg/NGSPipeline/Mercury/gsml_DL/demo/data/lung.Train.info.list \
        --f_valid_list Valid,/dssg/NGSPipeline/Mercury/gsml_DL/demo/data/lung.Valid.info.list \
        --f_valid_list R01B,/dssg/NGSPipeline/Mercury/gsml_DL/demo/data/lung.R01B.Valid.info.list \
        --f_feature /dssg/NGSPipeline/Mercury/gsml_DL/demo/data/feature1.csv \
        --f_hyper_params /dssg/NGSPipeline/Mercury/gsml_DL/demo/train_dl/hyper_conf.yaml \
        --d_output ~/test/train_dl_hyper/ \
        --prefix lung
    ```

2. **对已有模型再超参**

    解决R01B问题

    ```bash
    /dssg/home/sheny/anaconda3/envs/gsml/bin/python /dssg/NGSPipeline/Mercury/gsml_DL/gsml HyperDL \
        --f_model ~/test/train_dl/test.gsml  \
        --f_train /dssg/NGSPipeline/Mercury/gsml_DL/demo/data/lung.Train.info.list \
        --f_valid_list Valid,/dssg/NGSPipeline/Mercury/gsml_DL/demo/data/lung.Valid.info.list \
        --f_valid_list R01B,/dssg/NGSPipeline/Mercury/gsml_DL/demo/data/lung.R01B.Valid.info.list \
        --f_feature /dssg/NGSPipeline/Mercury/gsml_DL/demo/data/feature1.csv \
        --f_hyper_params /dssg/NGSPipeline/Mercury/gsml_DL/demo/train_dl/hyper_conf_step2.yaml \
        --d_output ~/test/train_dl_hyper/ \
        --prefix step2
    ```

