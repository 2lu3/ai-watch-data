from feature import Feature

feature_names = [
    'target','最高気温', '平均気温', '最低気温', '平均湿度',
    '平均現地気圧', '平均蒸気圧', '平均雲量', '平均風速', '日照時間']

feature = Feature(feature_names)

feature.standarlization()
dataset = feature.get_dataset()

print(dataset)
print(dataset['target'].mean())
