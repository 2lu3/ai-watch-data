from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import mean_absolute_error
from model import Model
from dataset import Dataset
from util import Util, Logger
import pandas as pd

logger = Logger()

# クロスバリデーションを含めた学習・評価・予測
class Runner:

  def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model],
      dataset, train_years: list, test_years: list, prms: dict, n_fold=None):
    """ コンストラクタ

    :param run_name: ランの名前
    :param model_cls: モデルのクラス
    :param params: ハイパーパラメーター
    """
    self.run_name = run_name
    self.model_cls = model_cls
    self.train_years = train_years
    self.params = prms
    if n_fold is None:
      n_fold = len(self.train_years)
    self.n_fold = n_fold
    self.dataset = dataset


  def train_fold(self, i_fold: Union[str, int]) -> Tuple[
      Model, Optional[np.array], Optional[np.array], Optional[float]]:
    """ クロスバリデーションでのfoldを指定して学習・評価を行う

    他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

    :param i_fold: foldの番号(すべてのときには'all'とする)
    :return: (モデル、レコードのindex、予測値、評価によるスコア)のタプル
    """

    # 学習データの読み込み
    validation = i_fold != 'all'

    if validation:
      # 学習データ・バリデーションデータをセットする

      va_idx = self.train_years[i_fold]
      tr_idx = self.train_years.copy()
      tr_idx.remove(va_idx)
      # 1次元配列に
      tr_idx = sum(tr_idx, [])

      va_data = self.dataset.get_data(va_idx, 'tokyo')
      tr_data = self.dataset.get_data(tr_idx, 'tokyo')

      tr_Y = tr_data['target']
      tr_X = tr_data.drop('target', axis=1)
      va_Y = va_data['target']
      va_X = va_data.drop('target', axis=1)

      # 学習を行う
      model = self.build_model(i_fold)
      model.train(tr_X, tr_Y, va_X, va_Y)

      # バリデーションデータへの予測・評価を行う
      va_pred = model.predict(va_X)
      score = mean_absolute_error(va_Y, va_pred)

      return model, va_idx, va_pred, score
    else:
      tr_data = self.dataset.get_data([i for i in range(2008, 2018)], 'tokyo')
      tr_Y = tr_data['target']
      tr_X = tr_data.drop('target')

      model = self.build_model(i_fold)
      model.train(train_x, train_y)
      return model, None, None, None

  def get_train_cv(self):
    logger.info(f'{self.run_name} - start training cv')

    scores = []
    va_idxes = []
    preds = []

    # 各foldで学習を行う
    for i_fold in range(self.n_fold):
      logger.info(f'{self.run_name} fold {i_fold} - start training')
      model, va_idx, va_pred, score = self.train_fold(i_fold)
      logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

      # モデルを保存する
      model.save_model()

      # 結果を保持する
      va_idxes.append(va_idx)
      scores.append(score)
      preds.append(va_pred)

    # 各foldの結果をまとめる
    va_idxes = np.concatenate(va_idxes)
    order = np.argsort(va_idxes)
    preds = np.concatenate(preds, axis=0)
    preds = preds[order]

    logger.info(f'{self.run_name} - end training cv')
    return preds, scores


  def run_train_cv(self) -> None:
    """ cvでの学習・評価を行う

    学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
    """
    preds, scores = self.get_train_cv()
    # 予測結果の保存
    Util.dump(preds, f'../model/pred/{self.run_name}-train.pkl')

    # 評価結果の保存
    logger.result_scores(self.run_name, scores)


  def get_predict_cv(self):
    """ cvで学習した各foldのモデルの平均により、テストデータの予測を行う

    あらかじめ、run_train_cvを実行しておく必要がある
    """
    logger.info(f'{self.run_name} - start predicting cv')

    te_data = self.dataset.get_data([2018, 2019], 'tokyo')
    te_Y = te_data['target']
    te_X = te_data.drop('target', axis=1)

    preds = []
    scores = []

    # 各foldのモデルで予測を行う
    for i_fold in range(self.n_fold):
      logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
      model = self.build_model(i_fold)
      model.load_model()
      pred = model.predict(te_X)
      preds.append(pred)
      score = mean_absolute_error(pred, te_Y)
      scores.append(score)
      logger.info(f'{self.run_name} - end prediction fold:{i_fold} - score {score}')

    logger.info(f'{self.run_name} - end prediction cv')
    # 予測の平均値を出力する
    pred_avg = np.mean(preds, axis=0)

    return pred_avg, te_Y, scores

  def run_predict_cv(self) -> None:
    pred_avg, _, scores = self.get_predict_cv()

    # 予測結果の保存
    Util.dump(pred_avg, f'../model/pred/{self.run_name}-test.pkl')

    # スコアの算出
    logger.result_scores(self.run_name, scores)

  def run_train_all(self) -> None:
    """ 学習データ全てで学習し、そのモデルを保存する """
    logger.info(f'{self.run_name} - start training all')

    # 学習データすべてで学習を行う
    i_fold = 'all'
    model, _, _, _ = self.train_fold(i_fold)
    model.save_model()

    logger.info(f'{self.run_name} - end training all')

  def run_predict_all(self) -> None:
    """ 学習データすべてで学習したモデルにより、テストデータの予測を行う

    あらかじめrun_train_allを実行しておく必要がある
    """
    logger.info(f'{self.run_name} - start prediction all')

    te_X = self.dataset.get_data([2018, 2019], 'tokyo')

    # 学習データ全てで学習したモデルで予測を行う
    i_fold = 'all'
    model = self.build_model(i_fold)
    model.load_model()
    pred = model.predict(te_X)

    # 予測結果の保存
    Util.dump(pred, f'../model/pred/{self.run_name}-test.pkl')

    logger.info(f'{self.run_name} - end prediction all')

  def build_model(self, i_fold: Union[int, str]) -> Model:
    """ cvでのfoldを指定し、モデルの作成を行う """

    run_fold_name = f'{self.run_name}-{i_fold}'
    return self.model_cls(run_fold_name, self.params)

