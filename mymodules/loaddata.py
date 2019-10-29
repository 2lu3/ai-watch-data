import subprocess
import pandas as pd
import glob
import re

def loadPdDic(root):
  # 熱中症患者のデータ
  heatstroke_path_list = glob.glob(root + 'heatstroke*.csv')
  heatstroke_pd_dic = {}
  for path in heatstroke_path_list:
    # -8 : -4  ~/heatstroke2018.csv
    heatstroke_pd_dic[path[-8:-4]] = pd.read_csv(path, encoding='utf-8')
  print('loaded heatstroke ', list(heatstroke_pd_dic.keys()))

  # 気象データ
  weather_pd_dic = {} 
  prefecture_path_list = glob.glob(root + 'weather-*.csv')
  for prefecture_path in prefecture_path_list:
    name = re.search(r'weather-\w+', prefecture_path).group()[8:]
    weather_pd_dic[name] = pd.read_csv(prefecture_path, encoding='utf-8')
  print('loaded weather', list(weather_pd_dic.keys()))
  
  # 人口関係
  pref_code_pd = pd.read_csv(root + 'code.csv', encoding='utf-8')
  pref_code_pd = pref_code_pd.loc[:,['県名', 'コード', '英語名']]
  population_pd = pd.read_csv(root + 'population.csv', encoding='utf-8')

  pref_pd = pd.merge(pref_code_pd, population_pd, on='県名')
  pref_pd = pref_pd.drop('県名', axis=1)
  pref_pd['コード'] = pref_pd['コード'].astype('int16')
  pref_pd['人口'] = pref_pd['人口'].apply(lambda x: x.replace(',',''))
  pref_pd['人口'] = pref_pd['人口'].astype('int64')
  print('loaded population and code', list(pref_pd.columns))

  return heatstroke_pd_dic, weather_pd_dic, pref_pd


def createDataSet(heatstroke_pd_dic, weather_pd_dic, prefecture_pd,
       merge_method, verbose=1):
  def SelectHeatStrokeDataByPrefectureCode(pref_code, year):
    query_message = "都道府県コード == " + str(pref_code)
    return heatstroke_pd_dic[year].query(query_message)
  def SelectWeatherDataByPrefectureNameAndMonth(prefecture_name, month = [5, 9]):
    def selectByMonth(month, data):
      def convertDate2MonthNumber(date):
        return pd.to_datetime(date).month
      data = data[(data['日付'].map(convertDate2MonthNumber) >= month[0]) & (data['日付'].map(convertDate2MonthNumber) <= month[1])]
      return data
    weather_pd = weather_pd_dic[prefecture_name]
    weather_pd = selectByMonth(month, weather_pd)
    return weather_pd
  def mergePdList(pd_list):
    if len(pd_list) == 1:
      return pd.concat([pd_list[0], None], sort=True)
    merged_pd = pd_list[0]
    for i in range(len(pd_list) - 1):
      merged_pd = pd.concat([merged_pd, pd_list[i+1]], sort=True)
    return merged_pd
  def mergeWeatherPdAndHeatstrokePd(pd1, pd2):
    return pd.merge(pd1, pd2, on='日付', how='outer', sort=True)
  def getMergeInfo(city_year_info):
    cities = city_year_info[0]
    if cities == ['all']:
      cities = list(prefecture_pd['英語名'].values)
    years = city_year_info[1]
    years = [years[i] if type(years[i]) is str else str(years[i]) for i in range(len(years))]
    if type(years) is not str:
        years = str(years)
    return cities, years, city_year_info[2]
  def getPrefectureInfo(city_name):
    prefecture_pd_line = prefecture_pd.query('英語名 == "' + city_name + '"')
    prefecture_code = prefecture_pd_line['コード'].values[0]
    prefecture_population = prefecture_pd_line['人口'].values[0]
    return prefecture_code, prefecture_population

  # listにすべてのデータを格納
  data_pd_list = []
  for city_year in merge_method: 
    cities, years, month = getMergeInfo(city_year)

    for city in cities:
      if city not in weather_pd_dic.keys():
        continue
      
      weather_pd = SelectWeatherDataByPrefectureNameAndMonth(city, month)

      prefecture_code, prefecture_population = getPrefectureInfo(city)
      weather_pd['人口'] = prefecture_population
      weather_pd['県名'] = city

      heatstroke_pd_list = [] 
      for year in years:
        heatstroke_pd = SelectHeatStrokeDataByPrefectureCode(prefecture_code, year)
        heatstroke_pd = heatstroke_pd.drop('都道府県コード', axis=1)
        heatstroke_pd = heatstroke_pd.rename(columns={'搬送人員（計）':'人数'})

        heatstroke_pd_list.append(heatstroke_pd)

      heatstroke_pd = mergePdList(heatstroke_pd_list)
      data_pd = mergeWeatherPdAndHeatstrokePd(heatstroke_pd, weather_pd)
      data_pd_list.append(data_pd)
  data_pd = mergePdList(data_pd_list)
  
  if verbose > 0:
    print(data_pd.info())

  return data_pd
