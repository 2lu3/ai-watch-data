from typing import Callable, List, Optional, Tuple, Union
from model import Model
from dataset import Dataset
import numpy as np


# クロスバリデーションを含めた学習・評価・予測
class Runner:

  def __init__(self, run_name: str, model_cls: Callable[[str, dict], Model],
      features: List[str], prms: dict, n_hold=None):
    """ コンストラクタ

    :param run_name: ランの名前
    :param model_cls: モデルのクラス
    :param features: 特徴量のリスト
    :param params: ハイパーパラメーター
    """
    self.run_name = run_name
    self.model_cls = model_cls
    self.features = features
    self.params = prms
    if n_hold is None:
      self.n_hold = 5
    self.dataset = Dataset(self.features)


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

      va_idx = [2008 + i_fold * 2, 2009 + i_fold * 2]
      print('val', va_idx)
      tr_idx = [y for y in range(2008, 2018)]
      for idx in va_idx:
        tr_idx.remove(idx)
      print('trn', tr_idx)

      va_data = self.dataset.get_data(va_idx, 'tokyo')
      tr_data = self.dataset.get_data(tr_idx, 'tokyo')

      tr_Y = tr_data['target']
      tr_X = tr_data.drop('target', axis=1)
      va_Y = va_data['target']
      va_X = va-data.drop('target', axis=1)

  def run_train_fold(i_fold):
    pass
  def run_train_cv():
    pass
  def run_predict_cv():
    pass
  def run_train_all():
    pass
  def run_predict_all():
    pass

  def build_model(i_fold):
    pass
  def load_x_train():
    pass
  def load_y_train():
    pass
  def load_x_test():
    pass
  def load_y_test():
    pass
  def load_index_fold(i_fold):
    pass

