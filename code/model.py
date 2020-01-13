from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

# 学習・予測・モデルの保存/読み込み


class Model(metaclass=ABCMeta):
    def __init__(self, run_hold_name: str, prms: dict) -> None:
        """ コンストラクタ

        :param run_hold_name: モデルを保存するpathに使用(例:lgbm-param1-fold1)
        :param params: ハイパーパラメーター
        """
        self.run_hold_name = run_hold_name
        self.prms = prms
        self.model = None

    @abstractmethod
    def train(
        self,
        tr_X: pd.DataFrame,
        tr_Y: pd.Series,
        va_X: Optional[pd.DataFrame] = None,
        va_Y: Optional[pd.Series] = None,
    ) -> None:
        # モデルの学習を行い、学習済みのモデルを保存する
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.array:
        # 学習済みのモデルでの予測値を返す
        pass

    @abstractmethod
    def save_model(self) -> None:
        # モデルの保存を行う
        pass

    @abstractmethod
    def load_model(self) -> None:
        # モデルの読み込みを行う
        pass
