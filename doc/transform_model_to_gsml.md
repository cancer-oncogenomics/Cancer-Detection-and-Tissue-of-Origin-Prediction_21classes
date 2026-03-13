# 将已有的pytorch model转换为gsml格式的文件

## 前提要求

1. pytorch版本必须小于等于2.3.1

2. forward函数需满足如下格式

    + 输出结果为一个嵌套列表，如果没有特征可以输出[output, []]
    + 内部不要做标准化等格式转换，只需要根据输出的x，用模型预测得到得分。

    ```python
        def forward(self, x, **kwargs):
            """前向传播"""
    
            features = self.feature_extractor(x)
            features = features.view(-1, self.fc_input_size)  # 将特征展平
    
            features2 = self.feature_extractor2(features)  # 特征值
            output = self.class_classifier(features2)  # 分类器
            return [output, [features2]]
    ```

    

## 生成gsml格式文件

首先，将pytorch model保存为{name}.model格式。然后通过如下脚本，在同一目录下生成{name}.gsml文件，name保持一致

```python
import sys
sys.path.append("/dssg/NGSPipeline/Mercury/gsml_DL")

from module.preprocess import get_scaler
from estimators.cnn import CNNEstimators
import joblib

# 生成数据标准化模块,标准化过程跟模型之前标准化流程要一致
df_train = pd.read_csv("~/test/train.tsv", sep="\t")
scaler = get_scaler(algo="minmax", na_strategy="mean")
scaler.c_features = ["cnv1", "cn2"]  # 这里填之前标准化时的特征顺序
scaler.class_tags = {"Response": ["Cancer", "Healthy"]}  # 这里填你标签分类的顺序

gsml_model = CNNEstimators(input_size=None, num_class=None)
gsml_model.scaler = scaler
gsml_model._score = pd.DataFrame(columns=["SampleID", "PredType"])
gsml_model.algorithm = "Pytorch--other-lung"  # 这里一定要以Pytorch开头
joblib.dump(gsml_model, f"{d_rslt}/{name}.gsml")
```







