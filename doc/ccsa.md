# CCSA模型训练示例

原始数据路径：/dssg05/InternalResearch05/sheny/Mercury/2024-06-27_LungDL

```python
path = outdir("/dssg05/InternalResearch05/sheny/Mercury/2024-06-27_LungDL")
d_rawtable = outdir(f"{path}/RawTable")
d_rawdata = outdir(f"{path}/RawData")
d_analyze = outdir(f"{path}/Analyze")
d_script = outdir(f"{path}/Script")
d_rslt = outdir(f"{path}/Result")
d_cleantable = outdir(f"{path}/CleanTable")
d_tmp = outdir(f"{path}/Tmp")

```



## 1. 数据准备

### 1.1 将train数据集拆分为training和validation

```python
from sklearn.model_selection import train_test_split

df_train = pd.read_csv(f"{d_dataset}/lung.Train.info.list", sep="\t")
df_train["stratify"] = df_train.ProjectID.apply(lambda x: f"{x}" if not x in ["20240301_R007_T7_QiagenEctractSOP", "KZ36"] else f"other")

df_training, df_validation, _, _ = train_test_split(df_train, df_train.Response, test_size=0.4, random_state=1, stratify=df_train.stratify)
df_training["Dataset"] = "training"
df_validation["Dataset"] = "validation"

df_test = pd.read_csv(f"{d_dataset}/lung.Test.info.list", sep="\t")
df_test2 = pd.read_csv(f"{d_dataset}/lung.Test2.info.list", sep="\t")
df_r01b = pd.read_csv(f"{d_dataset}/lung.R01B.info.list", sep="\t")

df_training.to_csv(f"{d_cleantable}/lung.Train.training.info.list", sep="\t", index=False)
df_validation.to_csv(f"{d_cleantable}/lung.Train.validation.info.list", sep="\t", index=False)
df_test.to_csv(f"{d_cleantable}/lung.Test.info.list", sep="\t", index=False)
df_test2.to_csv(f"{d_cleantable}/lung.Test2.info.list", sep="\t", index=False)
df_r01b.to_csv(f"{d_cleantable}/lung.R01B.info.list", sep="\t", index=False)
```

## 2. 特征预处理

将NA填充为0

```
df_cnv = pd.read_csv(f"{d_dataset}/Feature/lung.cnv.LE5X.csv")
df_cnv.fillna(0, inplace=True)
df_cnv.to_csv(f"{d_dataset}/Feature/lung.cnv.LE5X.csv", index=False)
```



## 3. 超参搜索

```python
cmd = (f"python /dssg/home/sheny/MyProject/gsml/gsml HyperCCSA "
			   f"--f_train {d_cleantable}/lung.Train.training.info.list "
			   f"--f_valid {d_cleantable}/lung.Train.validation.info.list "
			   f"--f_test {d_cleantable}/lung.Test.info.list "
			   f"--f_feature {d_dataset}/Feature/lung.{feature_name}.LE5X.csv "
			   f"--f_hyper_params ~/MyProject/gsml/config/hyper_turning/hyper_conf_ccsa.yaml "
			   f"--num_samples -1 "
			   f"--local_dir {d_output}/ "
			   f"--name {feature_name} "
			   f"--early_strategies test__accuracy,40,min,0.01")
```



## 4. 模型训练

```python
best_conf = pd.read_csv(f"{d_analyze}/hyper/{feature}/best_model_performance.txt", sep="\t").to_dict(orient="records")[0]
best_conf['batch_size'] = max(40, best_conf['batch_size'])
cmd = (f"python ~/MyProject/gsml/gsml Train_CCSA "
       f"--f_train {d_cleantable}/lung.Train.training.info.list "
       f"--f_valid {d_cleantable}/lung.Train.validation.info.list "
       f"--f_test {d_cleantable}/lung.Test.info.list "
       f"--f_feature {d_dataset}/Feature/lung.{feature_name}.LE5X.csv "
       f"--d_output {d_output} "
       f"--model_name {feature} "
       f"--early_strategies Train__loss,60,min,0.005 "
       f"--lr {best_conf['lr']} "
       f"--weight_decay {best_conf['weight_decay']} "
       f"--batch_size {best_conf['batch_size']} "
       f"--epochs 2000 "
       f"--out1 {int(best_conf['out1'])} "
       f"--conv1 {int(best_conf['conv1'])} "
       f"--pool1 {best_conf['pool1']} "
       f"--drop1 {best_conf['drop1']} "
       f"--out2 {int(best_conf['out2'])} "
       f"--conv2 {int(best_conf['conv2'])} "
       f"--pool2 {best_conf['pool2']} "
       f"--drop2 {best_conf['drop2']} "
       f"--fc1 {int(best_conf['fc1'])} "
       f"--fc2 {int(best_conf['fc2'])} "
       f"--fc3 {int(best_conf['fc3'])} "
       f"--fc4 {int(best_conf['fc4'])} "
       f"--drop3 {best_conf['drop3']} "
       f"--nfold 10"
       )
```















