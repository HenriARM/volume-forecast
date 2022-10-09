import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
import datetime

plt.style.use('fivethirtyeight')

filename = 'real_vol.csv'
df = pd.read_csv(filename, index_col=0).dropna()
# TODO: instantly read it pd.datetime not as string
darray1 = df['T'].to_numpy()
dt_format = '%Y-%m-%d %H:%M:%S'
df['T'] = list(map(datetime.datetime.strptime, darray1, len(darray1) * [dt_format]))
df = df.set_index('T')
df[['vol_last_10']].plot(figsize=(8, 8))
print(df)
plt.savefig('finally.png')
plt.show()
