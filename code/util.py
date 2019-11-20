import os
import datetime
import joblib
import logging
import numpy as np

class Util:
  @classmethod
  def dump(cls, value, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(value, path, compress=3)

  @classmethod
  def dump_feature(cls, value, name):
    cls.dump(value, './../input/' + name + '.pkl')

  @classmethod
  def load(cls, path):
    return joblib.load(path)

  @classmethod
  def load_feature(cls, name):
    return cls.load('./../input/' + name + '.pkl')

  @classmethod
  def reduce_mem_usage(df, logger=None, level=logging.DEBUG):
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    start_mem = df.memory_usage().sum() / 1024**2
    print_('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
      col_type = df[col].dtype
      if col_type != 'object' and col_type != 'datetime64[ns]':
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == 'int':
          if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
          elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
          elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
          elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
            df[col] = df[col].astype(np.int64)
        else:
          if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
            df[col] = df[col].astype(np.float32)  # feather-format cannot accept float16
          elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            df[col] = df[col].astype(np.float32)
          else:
            df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print_('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print_('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


class Logger:
  def __init__(self):
    self.general_logger = logging.getLogger('general')
    self.result_logger = logging.getLogger('result')

    stream_handler = logging.StreamHandler()

    file_general_handler = logging.FileHandler('../model/general.log')
    file_result_handler = logging.FileHandler('../model/result.log')

    if len(self.general_logger.handlers) == 0:
      self.general_logger.addHandler(stream_handler)
      self.general_logger.addHandler(file_general_handler)
      self.general_logger.setLevel(logging.INFO)
      self.result_logger.addHandler(stream_handler)
      self.result_logger.addHandler(file_result_handler)
      self.result_logger.setLevel(logging.INFO)

  def info(self, message):
    # 時刻をつけてコンソールとログに出力
    self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

  def result(self, message):
    self.result_logger.info(message)

  def result_ltsv(self, dic):
    self.result(self.to_ltsv(dic))

  def result_scores(self, run_name, scores):
    # 計算結果をコンソールと計算結果用ログに出力
    dic = dict()
    dic['name'] = run_name
    dic['score'] = np.mean(scores)
    for i, score in enumerate(scores):
      dic[f'score{i}'] = score
    self.result(self.to_ltsv(dic))

  def now_string(self):
    return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

  def to_ltsv(self, dic):
    return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])
