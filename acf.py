import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

output = './plots'
features = ['vol_last_10', 'market_spread', 'mid_price',
            'vol_imbalance', 'market_depth', 'ask_cv', 'bid_cv']

filename = './datasets/data.pkl'
data = pd.read_pickle(filename)
for feature in features:
    sm.graphics.tsa.plot_acf(
        x=data[feature],
        lags=50,
        title=f'{feature} ACF'
    )
    plt.savefig(os.path.join(output, f'acf_{feature}.png'))
    plt.show()
