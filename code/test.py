from dataset import Dataset
from runner import Runner
from model_lgbm import ModelLGBM
import matplotlib.pyplot as plt

prms = {
  'num_round': 10000,
  'early_stopping_rounds': 10,

  'task': 'train',
  'boosting_type': 'gbdt',
  'objective': 'regression',
  'metrics': 'huber',
  'num_leaves': 30,
  'learning_rate': 0.3,
  'verbosity': -1,
  'verbose_eval': -1,
  'verbose_early': 0,
}
features = [
    'target','最高気温', '平均気温', '最低気温', '平均湿度',
    '平均現地気圧', '平均蒸気圧', '平均雲量', '平均風速', '日照時間']

train_years = [
    #[2008, 2009],
    [2010, 2011],
    [2012, 2013],
    [2014, 2015],
    [2016, 2017],
    ]
test_years = [2018, 2019]


runner = Runner('test', ModelLGBM, features, train_years, test_years, prms)

runner.run_train_cv()
runner.run_predict_cv()

pred, correct,_ = runner.get_predict_cv()
plt.figure(figsize=(30, 30))
plt.plot(pred, color='r')
plt.plot(correct.values, color='black')
plt.show()
