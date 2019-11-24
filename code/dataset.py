import pandas as pd
from util import Util
import pathlib
import glob
from typing import Union

# 指定した条件のPdを返す
class Dataset:

  def __init__(self, feature_names, target_name='target', train_years=None, test_years=None, cities = None):
    # 目的変数名
    self.target_name = target_name

    self.train_years = train_years
    self.test_years = test_years
    self.cities = cities

    # Datasetの中でのみ使用するカラム
    self.secret_feature_names =['県名', 'date']
    self.feature_names = feature_names.copy()

    for name in self.secret_feature_names:
      if name in feature_names:
        self.secret_feature_names.remove(name)
      else:
        self.feature_names.append(name)

    base_dataset = Util.load_feature('basic_data')
    datasets_list = []
    for name in self.feature_names:
      if name in base_dataset.columns:
        datasets_list.append(base_dataset[name])
      else:
        feature = Util.load_feature(name)
        datasets_list.append(feature)
    self.dataset = pd.DataFrame().join(datasets_list, how='outer')

  @classmethod
  def get_all_feature_names(cls):
    # すべての特徴の名前を取得する
    data = []
    basic_data = Util.load_feature('basic_data')
    globbed_files = pathlib.Path('./../features/').glob('*.pkl')
    for globbed_file in globbed_files:
      file_name = globbed_file.name
      if file_name == 'basic_data.pkl':
        continue
      data.append(Util.load_feature(file_name[:-4]))
    data = basic_data.join(data, how='outer')
    return data.columns

  # 年度を条件にして絞り込む
  def __select_by_year(self, years, data=None):
    def __to_year(data):
      return data.year

    if data is None:
      data = self.dataset.copy()

    if type(years) == int:
      years = [years]

    # 年度情報がないデータは削除
    data = data.dropna(subset=['date'])
    adopted_index = False
    for year in years:
      adopted_index = ((adopted_index) |
          (data['date'].map(__to_year) == year))
    return data[adopted_index]

  # 県名を条件にして絞り込む
  def __select_by_city(self, city_names, data=None):
    if type(city_names) == str:
      city_names = [city_names]
    if data is None:
      data = self.dataset.copy()

    # 県名情報がないデータは削除
    data = data.dropna(subset=['県名'])
    return data[data['県名'].isin(city_names)]

  # 年度と県名を条件にして絞り込み、コピーを返す
  def get_data(self, year, city):
    data = self.__select_by_year(year)
    data = self.__select_by_city(city, data)
    data = data.drop(self.secret_feature_names, axis=1)
    data = data.dropna(subset=[self.target_name])
    data = data.dropna()
    return data

  # 2008 ~ 2017年度のデータ
  def get_train(self):
    if self.train_years is not None and self.cities is not None:
      return self.get_data(self.train_years, self.cities)
    else:
      return self.get_data([y for y in range(2008, 2018)], 'tokyo')

  # 2018, 2019年度のデータ
  def get_test(self, option=None):
    if self.test_years is not None and self.cities is not None:
      return self.get_data(self.test_years, self.cities)
    else:
      return self.get_data([2018, 2019], 'tokyo')

  def add_past_day_data(self, days_ago, features = None):
    if features is None:
      features = list(self.dataset.columns.copy())
    for name in self.secret_feature_names:
      features.remove(name)

    if type(days_ago) == int:
      days_ago = [days_ago]

    for i in days_ago:
      for name in features:
        self.dataset['p' + str(i) + name] = self.dataset[name].shift(-i)
