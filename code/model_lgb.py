from model import Model
from util import Util
import lightgbm as lgb
import os
import shap

class ModelLGB(Model):

  def train(self, tr_X, tr_Y, va_X=None, va_Y=None):
    self.lgb = lgb
    # データセット
    validation = va_X is not None
    dtrain = lgb.Dataset(tr_X, label=tr_Y)
    if validation:
      dvalid = lgb.Dataset(va_X, label=va_Y, reference=dtrain)

    # ハイパーパラメーターの設定
    params = dict(self.prms)
    num_round = params.pop('num_round')
    verbose = params.pop('verbose')

    # 学習
    if validation:
      early_rounds = params.pop('early_stopping_rounds')
      watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
      self.model = lgb.train(
          params,
          dtrain,
          num_boost_round=num_round,
          valid_sets=dvalid,
          verbose_eval=verbose_eval,
          early_stopping_round=early_rounds
          )
      model_array.append(self.model)
    else:
      watchlist = [(dtrain, 'train')]
      self.model = lgb.train(
          params,
          dtrain,
          num_boost_round=num_round,
          evals=watchlist
          )
      model_array.append(self.model)

  def predict(self, te_X):
    return self.model.predict(te_X, num_iteration=self.model.best_iteration)

  def predict_and_shap(self, te_X):
    fold_instance = shap.TreeExplainer(self.model).shap_values(te_X[:shap_sampling])
    valid_prediction = self.model.predict(te_X, num_iteration=self.model.best_iteration)
    return valid_prediction, fold_importance

  def save_model(self, path):
    here_path = os.path.dirname(__file__)
    model_path = os.path.join(here_path + '../model/', path, f'/{self.run_hold_name}.lgbm')
    Util.dump(self.model, model_path)

  def load_model(self, path):
    model_path = os.path.join('../model/', path, f'{self.run_hold_name}.lgbm')
    self.model = Util.load(model_path)

  @classmethod
  def calc_feature_importance(self, di_name, run_name, features):
    """ feature importanceの計算
    """
    val_split = model_array[0].feature_importance(importance_type='splut')
    val_grain = model_array[0].feature_importance(importance_type='gain')
    val_split = pd.Series(val_split)
    val_gain = pd.Series(val_gain)

    for model in model_array[1:]:
      s = pd.Series(model.feature_importance(importance_type='split'))
      val_split = pd.concat([val_split, s], axis=1)
      s = pd.Series(model.feature_importance(importance_type='gain'))
      val_gain = pd.concat([val_gain, s], axis=1)

    # ------------
    # splitの計算
    # ------------
    # 各foldの平均を算出
    val_mean = val_split.mean(axis=1).values
    importance_df_medan = pd.DataFrame(val_mean, index=features, columns=['importance']).sort_values('importance')

    # 各foldの標準偏差を算出
    val_std = val_split.std(axis=1).values
    importance_df_std = pd.DataFrame(val_std, index=features, columns=['importance']).sort_values('importance')

    # マージ
    df = pd.merge(importance_df_mean, importance_df_std, left_index=True, right_index=True ,suffixes=['_mean', '_std'])

    df['coef_of_var'] = df['importance_std'] / df['importance_mean']
    df['coef_of_var'] = df['coef_of_var'].fillna(0)
    df = df.sort_values('importance_mean', ascending=True)

    # 出力
    fig, ax1 = plt.subplots(figsize = (10, 10))
    plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
    plt.tight_layout()

    # 棒グラフを出力
    ax1.set_title('feature importance split')
    ax1.set_xlabel('feature importance mean & std')
    ax1.barh(df.index, df['importance_mean'], label='importance_mean',  align="center", alpha=0.6)
    ax1.barh(df.index, df['importance_std'], label='importance_std',  align="center", alpha=0.6)

    # 折れ線グラフを出力
    ax2 = ax1.twiny()
    ax2.plot(df['coef_of_var'], df.index, linewidth=1, color="crimson", marker="o", markersize=8, label='coef_of_var')
    ax2.set_xlabel('Coefficient of variation')

    #凡例を表示（グラフ左上、ax2をax1のやや下に持っていく）
    ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=12)
    ax2.legend(bbox_to_anchor=(1, 0.93), loc='upper right', borderaxespad=0.5, fontsize=12)

    #グリッド表示(ax1のみ)
    ax1.grid(True)
    ax2.grid(False)

    plt.savefig(dir_name + run_name + '_fi_split.png', dpi=300, bbox_inches="tight")
    plt.close()


    # -----------
    # gainの計算
    # -----------
    # 各foldの平均を算出
    val_mean = val_gain.mean(axis=1)
    val_mean = val_mean.values
    importance_df_mean = pd.DataFrame(val_mean, index=features, columns=['importance']).sort_values('importance')

    # 各foldの標準偏差を算出
    val_std = val_gain.std(axis=1)
    val_std = val_std.values
    importance_df_std = pd.DataFrame(val_std, index=features, columns=['importance']).sort_values('importance')

    # マージ
    df = pd.merge(importance_df_mean, importance_df_std, left_index=True, right_index=True ,suffixes=['_mean', '_std'])

    # 変動係数を算出
    df['coef_of_var'] = df['importance_std'] / df['importance_mean']
    df['coef_of_var'] = df['coef_of_var'].fillna(0)
    df = df.sort_values('importance_mean', ascending=True)

    # 出力
    fig, ax1 = plt.subplots(figsize = (10, 10))
    plt.tick_params(labelsize=12) # 図のラベルのfontサイズ
    plt.tight_layout()

    # 棒グラフを出力
    ax1.set_title('feature importance gain')
    ax1.set_xlabel('feature importance mean & std')
    ax1.barh(df.index, df['importance_mean'], label='importance_mean',  align="center", alpha=0.6)
    ax1.barh(df.index, df['importance_std'], label='importance_std',  align="center", alpha=0.6)

    # 折れ線グラフを出力
    ax2 = ax1.twiny()
    ax2.plot(df['coef_of_var'], df.index, linewidth=1, color="crimson", marker="o", markersize=8, label='coef_of_var')
    ax2.set_xlabel('Coefficient of variation')

    # 凡例を表示（グラフ左上、ax2をax1のやや下に持っていく）
    ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=12)
    ax2.legend(bbox_to_anchor=(1, 0.93), loc='upper right', borderaxespad=0.5, fontsize=12)

    # グリッド表示(ax1のみ)
    ax1.grid(True)
    ax2.grid(False)

    plt.savefig(dir_name + run_name + '_fi_gain.png', dpi=300, bbox_inches="tight")
    plt.close()

