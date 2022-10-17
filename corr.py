import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7, 7))
filename = './datasets/data.pkl'
df = pd.read_pickle(filename)
df = df[['vol_last_10', 'market_spread', 'mid_price',
         'vol_imbalance', 'market_depth', 'ask_cv', 'bid_cv']][:100]
# calculate the correlation matrix
corr = df.corr()
# plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.savefig('./plots/corr.png')
plt.show()
