from dataset import Dataset
from runner import Runner
from model_lgbm import ModelLGBM

prms = {
  'num_round': 100,
  'early_stopping_rounds': 10,

  'task': 'train',
  'boosting_type': 'gbdt',
  'objective': 'regression',
  'num_leaves': 30,
  'learning_rate': 0.3,
  'verbosity': -1,
  'verbose_eval': -1,
}


features = ['target', '最高気温', '平均気温', '最低気温']




runner = Runner('test', ModelLGBM, features, prms)
runner.run_train_cv()





