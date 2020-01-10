import os

import lightgbm as lgb
import shap
from model import Model
from util import Util


class ModelLGB(Model):

    def train(self, tr_X, tr_Y, va_X=None, va_Y=None):
        self.lgb = lgb
        # データセット
        validation = va_X is not None
        dtrain = lgb.Dataset(tr_X, label=tr_Y)
        if validation:
            dvalid = lgb.Dataset(va_X, label=va_Y, reference=dtrain)

        # ハイパーパラメーターの設定
        params = dict(self.prms)
        num_round = params.pop('num_round')
        verbose = params.pop('verbose')

        # 学習
        if validation:
            early_rounds = params.pop('early_stopping_rounds')
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

            self.model = lgb.train(params,
                                   dtrain,
                                   num_boost_round=num_round,
                                   valid_sets=dvalid,
                                   verbose_eval=verbose_eval,
                                   early_stopping_round=early_rounds)
            model_array.append(self.model)
        else:
            watchlist = [(dtrain, 'train')]
            self.model = lgb.train(
                params,
                dtrain,
                num_boost_round=num_round,
                evals=watchlist
            )
            model_array.append(self.model)

    def predict(self, te_X):
        return self.model.predict(te_X, num_iteration=self.model.best_iteration)

    def predict_and_shap(self, te_X):
        fold_instance = shap.TreeExplainer(
            self.model).shap_values(te_X[:shap_sampling])
        valid_prediction = self.model.predict(
            te_X, num_iteration=self.model.best_iteration)
        return valid_prediction, fold_importance

    def save_model(self):
        here = os.path.dirname(__file__)
        model_path = os.path.join('here' + '../model/model',
                                  f'{self.run_hold_name}.lgbm')
        Util.dump(self.model, model_path)

    def load_model(self):
        model_path = os.path.join(
            '../model/model', f'{self.run_hold_name}.lgbm')
        self.model = Util.load(model_path)
