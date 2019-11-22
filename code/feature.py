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

  def register_feature(self, feature, feature_name, description):
    Util.dump_feature(feature, feature_name)

  def standarlization(self):
    self.data = ((self.data - self.data.mean())
        / self.data.std(ddof=0))

