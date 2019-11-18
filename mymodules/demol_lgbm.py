import os

import numpy as np
import pandas as pd
import lightgbm as lgb

from model import Model
from util import Util

class ModelLGBM(Model):
  def train(self, tr_X, tr_Y, va_X=None, va_Y=None):
    # データセットの作成
    validation = va_x is not None
    dtrain = lgb.Dataset(tr_X, label=tr_Y)
    if validation:
      dvalid = lgb.Dataset(va_X, label=va_Y, reference=dtrain)

    # ハイパーパラメーターの設定
    params = dict(self.params)
    num_round = params.pop('num_round')

    # 学習
    if validation:
      early_stopping_rounds = params.pop('early_stopping_rounds')
      self.model = lgb.train(
          params,
          dtrain,
          num_boost_round=num_round,
          valid_sets=dvalid,
          early_stopping_rounds=early_stopping_rounds)
    else:
      self.model = lgb.train(
          params,
          dtrain,
          num_boost_round=num_round)

  def predict(self, te_X):
    return self.model.predict(te_X)

  def save_model(self):
    model_path = os.path.join('../model/model', f'{self.run_hold_name}.model')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    Util.dump(self.model, model_path)

  def load_model(self):
    model_path = os.path.join('../model/model', f'{self.run_hold_name}.model')
    self.model = Util.load(model_path)
