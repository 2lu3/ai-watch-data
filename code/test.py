from runner import Runner
from model_lgbm import ModelLGBM

model = ModelLGBM('test', {})

runner = Runner('test', model, [], {})
runner.train_fold(0)





