import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error

from dataset import Dataset


# クロスバリデーションを含めた学習・評価・予測
class Runner:
    def __init__(self, **kwargs):
        """ コンストラクタ

        :run_name str: ランの名前
        :model_cls model: モデルのクラス
        :features list: 特徴
        :param dict: パラメーター
            :target: 予測するカラム名
            :cities: データで使用する県
            :cv: クロスバリデーションの方法
                :manual: CVなし。訓練、バリデーションは手動指定
                :no valid: CVなし。訓練データは手動指定、バリデーションもなし
            :その他: LightGBMやXGBoostなどのパラメーターとして使用する
        """
        self.run_name = kwargs.pop("run_name")
        self.model_cls = kwargs.pop("model_cls")
        self.features = kwargs.pop("features")

        # prms
        prms = kwargs.pop("prms").copy()
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
        self.dataset = Dataset(self.features, self.target_name)

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
        self.model, valid_score = self.train(tr_X, tr_Y, va_X, va_Y)

        # 評価
        test_score, _, _ = self.evaluate(self.model, te_X, te_Y)

        self.model.save_model()
        return valid_score, test_score

    def evaluate(self, model, X, Y):
        pred_y = model.predict(X)

        return mean_absolute_error(Y, pred_y), pred_y, Y.values

    def __convert_featurename_to_number(self, df: pd.DataFrame) -> None:
        df_copy = df.copy()
        df_copy.columns = [str(i) for i in range(len(df.columns))]
        return df_copy

    def feature_importance(self):
        feature_names = self.features.copy()
        feature_names.remove(self.target_name)
        importance = self.model.feature_importance()
        plt.figure(figsize=(20, 10))
        plt.bar(feature_names, importance)
        plt.show()

    def plot_prediction(self, train=False, valid=False, test=True, figsize=(20, 10)):
        if train + valid + test != 1:
            print('Only one dataset could be choosed to plot')
        if train is True:
            x, y = self.__get_train_data()
            _, pred_y, correct_y = self.evaluate(self.model, x, y)
        elif valid is True:
            x, y = self.__get_valid_data()
            _, pred_y, correct_y = self.evaluate(self.model, x, y)
        elif test is True:
            x, y = self.__get_test_data()
            _, pred_y, correct_y = self.evaluate(self.model, x, y)
        plt.figure(figsize=figsize)
        plt.plot(pred_y, color='r')
        plt.plot(correct_y, color='k')
        plt.legend(['prediction', 'correct'])
        plt.show()

    def __build_model(self, i_fold=None):
        """ cvでのfoldを指定し、モデルの作成を行う """

        if i_fold is None:
            run_fold_name = f"{self.run_name}"
        else:
            run_fold_name = f"{self.run_name}-{i_fold}"
        return self.model_cls(run_fold_name, self.params)

    def __split_x_y(self, dataset):
        target_index = str(self.features.index(self.target_name))
        Y = dataset[target_index]
        X = dataset.drop(target_index, axis=1)
        return X, Y

    def __get_train_data(self):
        data = self.dataset.get_data(self.train_years, self.cities)
        data = self.__convert_featurename_to_number(data)
        return self.__split_x_y(data)

    def __get_valid_data(self):
        data = self.dataset.get_data(self.valid_years, self.cities)
        data = self.__convert_featurename_to_number(data)
        return self.__split_x_y(data)

    def __get_test_data(self):
        data = self.dataset.get_data(self.test_years, self.cities)
        data = self.__convert_featurename_to_number(data)
        return self.__split_x_y(data)
