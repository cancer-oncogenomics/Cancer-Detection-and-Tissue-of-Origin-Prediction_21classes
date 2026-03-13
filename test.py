
import os
import sys

path = "/dssg/home/sheny/MyProject/gsml"
sys.path.insert(0, os.path.abspath(path))

from module.train_dl import DLTrain
from module.hyper_tuning import HyperDL
from module.metrics import Metrics

f_train = f"/mnt/e/shenyi/projects/projects/gsml/demo/data/lung.Train.info.list"
f_target = f"/mnt/e/shenyi/projects/projects/gsml/demo/data/lung.Train.info.list"
f_feature = f"/mnt/e/shenyi/projects/projects/gsml/demo/data/feature1.csv"
valid_dict = {
    "Valid": f"/mnt/e/shenyi/projects/projects/gsml/demo/data/lung.Valid.info.list",
    "R01B": f"/mnt/e/shenyi/projects/projects/gsml/demo/data/lung.R01B.Valid.info.list",
}
d_output = "/home/shenyi/test/train"
f_config = "/mnt/e/shenyi/projects/projects/gsml/config/dl_train/dl_train_dann.yaml"
# f_model = "/dssg06/InternalResearch06/sheny/Mercury/2024-07-23_LungDL/Analyze/train5/lung.1721868715.0770388.gsml"

train = DLTrain(
    f_train=f_train,
    f_feature=f_feature,
    valid_dict=valid_dict,
    d_output=d_output,
    prefix="test",
    f_config=f_config,
    f_target=f_target,
)

train.run()