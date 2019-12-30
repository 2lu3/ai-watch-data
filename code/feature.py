import pandas as pd
import numpy as np
import time
from dataset import Dataset
from util import Util
from sklearn import preprocessing

class Feature():
  def __init__(self, use_features):
    self.dataset = Dataset(use_features)
    years = [y for y in range(2008, 2020)]
    self.data = self.dataset.get_data(years, 'tokyo')


  def get_dataset(self):
    return self.data.copy()

  def register_feature(self, feature, feature_name, feature_describe):
    Util.dump_feature(feature, feature_name)

  def standarlization(self):
    for name in self.data.columns:
      if self.data[name][0] is int or self.data[name][0] is float:
        self.data[name] = ((self.data[name] - self.data[name].mean())
            / self.data[name].std(ddof=0))


