import pandas as pd
from util import Util

class Dataset():
  def __init__(self, feature_names):
    self.root_dir = '../input/'
    self.dataset = Util.load(self.root_dir + 'base_dataset.pickle')
    for name in feature_names:
      features = Util.load(self.root_dir + name + '.pickle')
      self.dataset = pd.concat([self.dataset, features], axis=1)

  def get_data(self, option):
    option = [[['tokyo'], option, [5, 9]]]
    return createDataSet(
        self.heat_dic,
        self.weather_dic,
        self.pref_pd,
        option,
        verbose=0)

