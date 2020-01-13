import glob
import re

import pandas as pd

from util import Util


def save_pickle_from_csv(root_dir="./../input/"):
    merge_method = [["tokyo"], [y for y in range(2008, 2019)], [5, 9]]
    dataset = BasicDataset(root_dir, merge_method)
    data = dataset.get_data()
    Util.dump_feature(data, "basic_data")


class BasicDataset:
    def __init__(self, root_path, merge_method, verbose=False):
        self.root_path = root_path
        self.merge_method = merge_method
        self.verbose = verbose

        self.heatstroke_pd_dic = self.__load_heatstroke_dic()
        self.weather_pd_dic = self.__load_weather_dic()
        self.pref_pd = self.__load_pref_pd()
        self.dataset = self.__fix_data_consistency(self.__load_basic_dataset())

    def get_data(self):
        return self.dataset.copy()

    def __fix_data_consistency(self, data):
        data = data.rename(columns={"搬送人員（計）": "target", "日付": "date"})
        data["date"] = pd.to_datetime(data["date"])
        return data

    # 熱中症患者のデータ
    def __load_heatstroke_dic(self):
        heatstroke_path_list = glob.glob(self.root_path + "heatstroke*.csv")
        heatstroke_pd_dic = {}
        for path in heatstroke_path_list:
            # -8 : -4  ~/heatstroke2018.csv
            heatstroke_pd_dic[path[-8:-4]] = pd.read_csv(path, encoding="utf-8")
        if self.verbose is True:
            print("loaded heatstroke ", list(heatstroke_pd_dic.keys()))
        return heatstroke_pd_dic

    # 気象データ
    def __load_weather_dic(self):
        weather_pd_dic = {}
        prefecture_path_list = glob.glob(self.root_path + "weather-*.csv")
        for prefecture_path in prefecture_path_list:
            name = re.search(r"weather-\w+", prefecture_path).group()[8:]
            weather_pd_dic[name] = pd.read_csv(prefecture_path, encoding="utf-8")
        if self.verbose is True:
            print("loaded weather", list(weather_pd_dic.keys()))
        return weather_pd_dic

    # 人口関係
    def __load_pref_pd(self):
        pref_code_pd = pd.read_csv(self.root_path + "code.csv", encoding="utf-8")
        pref_code_pd = pref_code_pd.loc[:, ["県名", "コード", "英語名"]]
        population_pd = pd.read_csv(self.root_path + "population.csv", encoding="utf-8")

        # 合成
        pref_pd = pd.merge(pref_code_pd, population_pd, on="県名")
        pref_pd = pref_pd.drop("県名", axis=1)

        # 3桁ごとの区切り文字の削除
        pref_pd["人口"] = pref_pd["人口"].apply(lambda x: x.replace(",", ""))
        pref_pd["コード"] = pref_pd["コード"].astype("int16")
        pref_pd["人口"] = pref_pd["人口"].astype("int64")

        if self.verbose is True:
            print("loaded population and code", list(pref_pd.columns))
        return pref_pd

    def __load_basic_dataset(self):
        def __select_heatstroke_data(self, pref_code, year):
            query = "都道府県コード == " + str(pref_code)
            return self.heatstroke_pd_dic[year].query(query)

        def __select_weather_data(self, prefecture_name, month):
            def __select_by_month(month, data):
                def __convert_date_to_month(date):
                    return pd.to_datetime(date).month

                data = data[
                    (data["日付"].map(__convert_date_to_month) >= month[0])
                    & (data["日付"].map(__convert_date_to_month) <= month[1])
                ]
                return data

            return __select_by_month(month, self.weather_pd_dic[prefecture_name])

        def __merge_pd_list(pd_list):
            if len(pd_list) == 1:
                pd_list.append(None)
            merged_pd = pd_list[0]
            for i in range(len(pd_list) - 1):
                merged_pd = pd.concat([merged_pd, pd_list[i + 1]], sort=True)
            return merged_pd

        def __merge_weather_heatstroke(pd1, pd2):
            return pd.merge(pd1, pd2, on="日付", how="outer", sort=True)

        def __get_merge_info(self, city_year_month_info):
            cities = city_year_month_info[0]
            if cities == ["all"]:
                cities = list(self.pref_pd["英語名"].values)
            for city in cities:
                if city not in self.weather_pd_dic.keys():
                    cities.remove(city)
            years = city_year_month_info[1]
            for i in range(len(years)):
                if type(years[i]) is not str:
                    years[i] = str(years[i])
            return cities, years, city_year_month_info[2]

        def __get_pref_info(self, city_name):
            prefecture_pd_line = self.pref_pd.query('英語名 == "' + city_name + '"')
            pref_code = prefecture_pd_line["コード"].values[0]
            prefecture_population = prefecture_pd_line["人口"].values[0]
            return pref_code, prefecture_population

        # listにすべてのデータを格納
        data_pd_list = []
        cities, years, month = __get_merge_info(self, self.merge_method)
        for city in cities:
            weather_pd = __select_weather_data(self, city, month)

            pref_code, prefecture_population = __get_pref_info(self, city)
            weather_pd["人口"] = prefecture_population
            weather_pd["県名"] = city

            heatstroke_pd_list = []
            for y in years:
                heatstroke_pd = __select_heatstroke_data(self, pref_code, y)
                heatstroke_pd = heatstroke_pd.drop("都道府県コード", axis=1)

                heatstroke_pd_list.append(heatstroke_pd)

            heatstroke_pd = __merge_pd_list(heatstroke_pd_list)
            data_pd = __merge_weather_heatstroke(heatstroke_pd, weather_pd)
            data_pd_list.append(data_pd)
        data_pd = __merge_pd_list(data_pd_list)

        if self.verbose > 0:
            print(data_pd.info())

        return data_pd


if __name__ == "__main__":
    merge_method = [["tokyo"], [y for y in range(2008, 2019)], [5, 9]]
    dataset = BasicDataset("./../input/", merge_method)
    data = dataset.get_data()
    # data = data.drop(['天気概況(夜)', '天気概況'
    Util.dump_feature(data, "basic_data")
