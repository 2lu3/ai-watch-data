""" Copyright 2lu3  """
from typing import Callable

from dataset import Dataset

from model import Model

from sklearn.metrics import mean_absolute_error


# クロスバリデーションを含めた学習・評価・予測
class Runner:
    def __init__(
        self,
        run_name: str,
        model_cls: Callable[[str, dict], Model],
        features: dict,
        prms: dict,
    ):
        """ コンストラクタ

        :run_name: ランの名前
        :model_cls: モデルのクラス
        :features: 特徴
        :param: パラメーター
            :target: 予測するカラム名
            :cities: データで使用する県
            :cv: クロスバリデーションの方法
                :manual: CVなし。訓練、バリデーションは手動指定
                :no valid: CVなし。訓練データは手動指定、バリデーションもなし
            :その他: LightGBMやXGBoostなどのパラメーターとして使用する
        """
        self.run_name = run_name
        self.model_cls = model_cls
        self.features = features

        # prms
        self.target_name = prms.pop("target")

        self.cities = prms.pop("cities")

        self.cv = prms.pop("cv")

        if self.cv == "manual":  # 手動指定
            self.train_years = prms.pop("train_years")
            self.valid_years = prms.pop("valid_years")
            self.test_years = prms.pop("test_years")
        elif self.cv == "no valid":  # validationなし
            self.train_years = prms.pop("train_years")
            self.valid_years = None
            self.test_years = prms.pop("test_years")
        else:
            # TODO write
            pass
        self.params = prms

        # データセットの作成
        self.dataset = Dataset(features, self.target_name)

    def train(self, tr_X, tr_Y, va_X=None, va_Y=None, i_fold=None):
        """
        input : train data, valid data
        output : model, score
        """
        # train
        model = self.__build_model(i_fold)
        model.train(tr_X, tr_Y, va_X, va_Y)

        # evaluate score of validation
        if va_X is not None:
            va_pred = model.predict(va_X)
            score = mean_absolute_error(va_Y, va_pred)
        else:
            score = None
        return model, score

    def run(self):
        if self.cv == "manual":
            tr_X, tr_Y = self.__get_train_data()
            va_X, va_Y = self.__get_valid_data()
            te_X, te_Y = self.__get_test_data()
        elif self.cv == "no valid":
            tr_X, tr_Y = self.__get_train_data()
            va_X = va_Y = None
            te_X, te_Y = self.__get_test_data()
        else:
            # TODO: write
            pass

        # 訓練
        model, valid_score = self.train(tr_X, tr_Y, va_X, va_Y)

        # 評価
        test_score = self.evaluate(model, te_X, te_Y)

        return valid_score, test_score

    def evaluate(self, model, X, Y):
        pred_y = model.predict(X)

        return mean_absolute_error(Y, pred_y)

    def __build_model(self, i_fold=None):
        """ cvでのfoldを指定し、モデルの作成を行う """

        if i_fold is None:
            run_fold_name = f"{self.run_name}"
        else:
            run_fold_name = f"{self.run_name}-{i_fold}"
        return self.model_cls(run_fold_name, self.params)

    def __split_x_y(self, dataset):
        Y = dataset[self.target_name]
        X = dataset.drop(self.target_name, axis=1)
        return X, Y

    def __get_train_data(self):
        return self.__split_x_y(
                self.dataset.get_data(self.train_years, self.cities))

    def __get_valid_data(self):
        return self.__split_x_y(
                self.dataset.get_data(self.valid_years, self.cities))

    def __get_test_data(self):
        return self.__split_x_y(
                self.dataset.get_data(self.test_years, self.cities))