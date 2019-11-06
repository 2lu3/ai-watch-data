from loaddata import loadPdDic, createDataSet

class Dataset():
    def __init__(self, option=None, root='ai-watch-data/'):
        self.heat_dic, self.weather_dic, self.pref_pd = loadPdDic(root)

        if option is None:
            option = [
                    [[['tokyo'], [i for i in range(2008, 2017)], [5, 9]]],
                    [[['tokyo'], [2017], [5,9]]],
                    [[['tokyo'], [2018], [5,9]]]
                    ]
        self.option = option
        self.datasets = [None, None, None]

    def get_dataset(self, data_type_number, option = None):
        if option is not None:
            self.datasets[data_type_number] = None
        else:
            option = self.option[data_type_number]
        if self.datasets[data_type_number] is None:
            self.datasets[data_type_number] = createDataSet(
                    self.heat_dic,
                    self.weather_dic,
                    self.pref_pd,
                    option,
                    verbose=0)
        return self.datasets[data_type_number].copy()


    def get_train(self, option=None):
        return self.get_dataset(0, option)
    def get_valid(self, option=None):
        return self.get_dataset(1, option)
    def get_test(self, option=None):
        return self.get_dataset(2, option)
