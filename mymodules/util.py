import os
from skelarn.externals import joblib


class Util:

  @classmethod
  def dump(cls, value, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(value, path, compress=True)
  @classmethod
  def load(cls, path):
    return joblib.load(path)

