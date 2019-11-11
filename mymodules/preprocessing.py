# https://amalog.hateblo.jp/entry/kaggle-snippets
import numpy as np
import logging
import pandas as pd

def Preprocessing(datasets_origin, verbose, convert, option):
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
  def convertDate2OneHotWeekDay(data):
      def date2WeekDayNumber(date):
        return pd.to_datetime(date).day_name()
      weekday_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'] 
      for name in weekday_names:
        # 曜日が一致するか(true or false).astype(float)
        data[name] = (data['日付'].map(date2WeekDayNumber) == name).astype('float32')
      return data
  def addPreviousDayWeatherData(data):
    # 日付以外のすべてのカラムのコピーを作成
    data_columns = data.columns
    for name in data_columns:
      if name == '日付':
        continue
      data['p'+name] = data[name].copy().shift(-1)
    return data
  def splitY(Y_np, split_num, dimension):
    convert_Y_np = Y_np.copy() 
    if dimension == 1:
      for i in range(len(split_num) - 1):
        change_index = (split_num[i] <= convert_Y_np) & (convert_Y_np < split_num[i+1])
        convert_Y_np[change_index] = i
    elif dimension == 2:
      convert_Y_np = convert_Y_np.reshape((convert_Y_np.shape[0], 1))
      for i in range(len(split_num) - 1):
        convert_Y_np = np.insert(convert_Y_np, -1, np.logical_and(split_num[i] <= convert_Y_np[:,-1], convert_Y_np[:,-1] < split_num[i+1]), axis=1)
      convert_Y_np = np.delete(convert_Y_np, -1, axis=1)
    return convert_Y_np
  def createFeatureValue(data_pd_origin):
    data_pd = data_pd_origin.copy()
    # 曜日をone-hotで追加
    if option['add_onehot_day'] == True:
      data_pd = convertDate2OneHotWeekDay(data_pd)

    # いらないパラメーターの削除
    drop_list_weather = option['drop_list_weather']
    drop_list_injury = option['drop_list_injury']
    drop_list_years = option['drop_list_years']
    drop_list_place = option['drop_list_place']
    drop_list_others = option['drop_list_others']

    drop_list_injury = ['傷病程度：' + drop_injury for drop_injury in drop_list_injury]
    drop_list_years = ['年齢区分：' + drop_years for drop_years in drop_list_years]
    drop_list_place = ['発生場所：' + drop_place for drop_place in drop_list_place]

    drop_list = drop_list_weather + drop_list_injury + drop_list_years + drop_list_place + drop_list_others
    for drop_name in drop_list:
      try:
        data_pd = data_pd.drop(drop_name, axis=1)
      except KeyError as e:
        print(e)
    data_pd = data_pd.dropna()

    # 前日データを追加
    if option['add_previous_day'] == True:
      data_pd = addPreviousDayWeatherData(data_pd)
    return data_pd
  def convertDataRange(datasets):
    # 1つのpandasを標準化
    def standardizationData(data, data_std = None, data_mean = None):
      if data_std is None:
        data_std = data.std(ddof=False)
      if data_mean is None:
        data_mean = data.mean()
      return (data - data_mean) / data_std
    # 複数のpandasをまとめて加工
    def normalizationDataSets(datasets, label, min_value=0, max_value=None):
      max_value = 0
      min_value = 10000
      epsilon = 10**-10
      for i in range(len(datasets)):
        max_value = max(max_value, datasets[i][label].max())
        min_value = min(min_value, datasets[i][label].min())
      for i in range(len(datasets)):
        datasets[i][label] = (datasets[i][label] - min_value) / (max_value - min_value + epsilon)
      return datasets

    # 標準化
    data_std = None
    data_mean = None
    standardization_name_list = ['最高気温', '最低気温', '平均気温', '平均風速', '平均蒸気圧', '平均雲量']
    for convert_name in standardization_name_list:
      for i in range(len(datasets)):
        if convert_name in datasets[i].columns:
          datasets[i][convert_name] = standardizationData(datasets[i][convert_name], data_std, data_mean)

    # 正規化
    normalization_name_list = ['人口', '降水量']
    for convert_label in normalization_name_list:
      if convert_label in datasets[0].columns:
        datasets = normalizationDataSets(datasets, convert_label)
    return datasets
  def convertDataSets2Np(datasets, label):
    x_datas = []
    y_datas = []
    for i in range(len(datasets)):
      x_datas.append(datasets[i].drop(label, axis=1).values)
      y_datas.append(datasets[i][label].values)
    return x_datas, y_datas


  # コピーを作成
  datasets = [data.copy() for data in datasets_origin]

  # 特徴量作成
  datasets = [createFeatureValue(data) for data in datasets]

  # 削減
  datasets = [reduce_mem_usage(data) for data in datasets]

  # 標準化・正規化
  if option['convert_data_range'] == True:
    datasets = convertDataRange(datasets)

  # データを整理
  # drop: indexをデータに挿入しない
  datasets = [datasets[i].reset_index(drop=True) for i in range(len(datasets))]

  if verbose >= 1:
    print(datasets[0].info())

  # numpyへの変換
  x_datas, y_datas = convertDataSets2Np(datasets, '人数')

  # 人数をone-hotに
  split_num = option['split_num']
  if option['split_y'] == True:
    y_datas = [splitY(y_datas[i], split_num, dimension=option['dimension']) for i in range(len(y_datas))]

  if convert == True:
    return x_datas, y_datas
  else:
    return data_pd

# # use case
# option = {
#     'split_num' : [0, 2, 10, 80, 120, 500000],
#     'drop_list_weather' : [],#+ ['平均気温', '最低気温', '平均風速'] + ['最高気温', '平均蒸気圧', '平均雲量', '降水量']
#     'drop_list_injury' : ['その他', '中等症', '死亡', '軽症', '重症'],
#     'drop_list_years' : ['不明', '乳幼児', '少年', '成人', '新生児', '高齢者'],
#     'drop_list_place' : ['その他', '仕事場①', '仕事場②', '住居', '公衆(屋内)', '公衆(屋外)', '教育機関', '道路'],
#     'drop_list_others' : ['県名', '日付'] + ['人口'],
#     'add_onehot_day' : False,
#     'add_previous_day' : False,
#     'add_population' : False,
#     'split_y' : True,
#     'dimension' : 2,
#     'convert_data_range' : True
# }
# datasets = [train_origin_pd, valid_origin_pd, test_origin_pd]
# X_datas, Y_datas = Preprocessing(datasets, verbose=1, convert=True, option=option)
