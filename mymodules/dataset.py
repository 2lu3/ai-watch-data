import loaddata

class Dataset():
    def __init__(self, option=None, root_dir='ai-watch-data/'):
        pddics = loaddata.loadPdDic(root_dir)
        self.heatstroke_dic, self.weather_dic, self.prefecture_pd = pddics
        if type(option) is None:
            option = [
                    [[['tokyo'], [i for i in range(2008, 2017)], [5, 9]]],
                    [[['tokyo'], [2017], [5,9]]],
                    [[['tokyo'], [2018], [5,9]]]
                    ]
        self.option = option
        train_data = loaddata.createDataSet(
                self.heatstroke_dic,
                self.weather_dic,
                self.prefecture_pd,
                self.option[0],
                verbose=0)
        valid_data = loaddata.createDataSet(
                self.heatstroke_dic,
                self.weather_dic,
                self.prefecture_pd,
                self.option[1],
                verbose=0)

        test_dat = loaddata.createDataSet(
                self.heatstroke_dic,
                self.weather_dic,
                self.prefecture_pd,
                self.option[2],
                verbose=0)


    def get_train(self):
        return train_data.copy()
    def get_valid(self):
        return valid_data.copy()
    def get_test(self):
        return test_data.copy()
