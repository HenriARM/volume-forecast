from hyperts import make_experiment
from hyperts.datasets import load_network_traffic
from hyperts.framework.search_space import StatsForecastSearchSpace, DLForecastSearchSpace

from sklearn.model_selection import train_test_split
import pandas as pd

filename = './train.csv'
data = pd.read_csv(filename, index_col=0)
# data = data.set_index('T')

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
print(train_data)
print(train_data.info())

model = make_experiment(train_data.copy(),
                        task='forecast',  # forecast, tsf, multivariate-forecast, regression, tsr
                        target='vol_mov',
                        mode='dl',
                        timestamp='T',
                        # covariates=['HourSin', 'WeekCos', 'CBWD']
                        # tf_gpu_usage_strategy=1,
                        reward_metric='rmse',
                        max_trials=30,
                        early_stopping_rounds=10).run()

X_test, y_test = model.split_X_y(test_data.copy())
y_pred = model.predict(X_test)
scores = model.evaluate(y_test, y_pred)
model.plot(forecast=y_pred, actual=test_data)


"""
DL
    enable_deepar: bool, default True.
    enable_hybirdrnn: bool, default True.
    enable_lstnet: bool, default True.
    
Autoregression
      
"""