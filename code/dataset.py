import pandas as pd
from util import Util

# 指定した条件のPdを返す
class Dataset:

  def __init__(self, feature_names):

    if len(feature_names) == 0:
      self.dataset = pd.DataFrame()
      return

    # Datasetの中でのみ使用するカラム
    self.local_feature_names =['県名', 'date']
    for name in self.local_feature_names:
      if name in feature_names:
        self.local_feature_names.remove(name)
      else:
        feature_names.append(name)

    if 'target' not in feature_names:
      feature_names.append('target')

    base_dataset = Util.load_feature('basic_data')
    datasets_list = []
    for name in feature_names:
      if name in base_dataset.columns:
        datasets_list.append(base_dataset[name])
      else:
        feature = Util.load_feature(name)
        datasets_list.append(feature)
    self.dataset = pd.DataFrame().join(datasets_list, how='outer')
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
    data = data.drop(self.local_feature_names, axis=1)
    data = data.dropna(subset=['target'])
    data = data.dropna()
    return data

  # 2008 ~ 2017年度のデータ
  def get_train(self):
    return self.get_data([y for y in range(2008, 2018)], 'tokyo')

  # 2018, 2019年度のデータ
  def get_test(self, option=None):
    return self.get_data([2018, 2019], 'tokyo')
