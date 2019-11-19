import pandas as pd
from util import Util

# 指定した条件のPdを返す
class Dataset:
  def __init__(self, feature_names):
    self.dataset = Util.load_feature('basic_data')
    for name in feature_names:
      feature = Util.load_feature(name)
      self.dataset = pd.concat([self.dataset, feature], axis=1)

  # 年度を条件にして絞り込む
  def select_by_year(self, years, data=None):
    def __to_year(data):
      return data.year
    if data is None:
      data = self.dataset.copy()

    if type(years) == int:
      years = [years, years+1]

    # 年度情報がないデータは削除
    data = data.dropna(subset=['date'])
    adopted_index = ((self.dataset['date'].map(__to_year) >= years[0]) &
        (self.dataset['date'].map(__to_year) < years[0]))
    return self.dataset[adopted_index]

  # 県名を条件にして絞り込む
  def select_by_city(self, city_names,data=None):
    if type(city_names) == str:
      city_names = [city_names]
    if data is None:
      data = self.dataset.copy()
    data = data.dropna(subset=['県名'])
    return self.dataset[self.dataset['県名'].isin(city_names)]

  # 年度と県名を条件にして絞り込み、コピーを返す
  def get_data(self, year, city):
    data = self.select_by_year(year)
    data = self.select_by_city(city, data)
    return data

  # 2008 ~ 2017年度のデータ
  def get_train(self):
    return self.get_data([y for y in range(2008, 2018)], 'tokyo')

  # 2018, 2019年度のデータ
  def get_test(self, option=None):
    return self.get_data([2018, 2019], 'tokyo')
