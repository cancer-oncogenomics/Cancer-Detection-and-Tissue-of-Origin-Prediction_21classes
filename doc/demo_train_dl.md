# 使用神经网络做模型训练的示例

1. **创建分析目录**

    ```bash
    mkdir -p ~/test/train_dl && cd -
    ```

2. **预训练模型**

    模型训练目前是Valid_acc好就行

    ```bash
    /dssg/NGSPipeline/Mercury/gsml_DL/gsml train_dl \
        --f_train /dssg/NGSPipeline/Mercury/gsml_DL/demo/data/lung.Train.info.list \
        --f_valid Valid,/dssg/NGSPipeline/Mercury/gsml_DL/demo/data/lung.Valid.info.list \
        --f_valid R01B,/dssg/NGSPipeline/Mercury/gsml_DL/demo/data/lung.R01B.Valid.info.list \
        --f_feature /dssg/NGSPipeline/Mercury/gsml_DL/demo/data/feature1.csv \
        --f_config /dssg/NGSPipeline/Mercury/gsml_DL/demo/train_dl/dl_train.yaml \
        --d_output ./ \
        --prefix test
    ```

3. **第二步训练**

    解决R01B问题

    ```bash
    /dssg/NGSPipeline/Mercury/gsml_DL/gsml train_dl \
        --f_model ./test.gsml \
        --f_train /dssg/NGSPipeline/Mercury/gsml_DL/demo/data/lung.Train.info.list \
        --f_valid Valid,/dssg/NGSPipeline/Mercury/gsml_DL/demo/data/lung.Valid.info.list \
        --f_valid R01B,/dssg/NGSPipeline/Mercury/gsml_DL/demo/data/lung.R01B.Valid.info.list \
        --f_feature /dssg/NGSPipeline/Mercury/gsml_DL/demo/data/feature1.csv \
        --f_config /dssg/NGSPipeline/Mercury/gsml_DL/demo/train_dl/dl_train_step2.yaml \
        --d_output ./ \
        --prefix step2
    ```

