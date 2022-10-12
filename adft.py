"""
Augmented Dickey-Fuller Test

A time series is said to be “stationary” if it has no trend, exhibits constant variance over time,
and has a constant autocorrelation structure over time.
H0: The time series is non-stationary.
HA: The time series is stationary.
"""

from statsmodels.tsa.stattools import adfuller
import pandas as pd

filename = './datasets/raw.csv'
data = pd.read_csv(filename, index_col=0)
# data = [3, 4, 4, 5, 6, 7, 6, 6, 7, 8, 9, 12, 10]
print(adfuller(data['vol'][:500000]))

# (-0.9753836234744063, 0.7621363564361013, 0, 12,
# {'1%': -4.137829282407408, '5%': -3.1549724074074077, '10%': -2.7144769444444443}, 31.2466098872313)
# => Test statistic: -0.97538 and P-value: 0.7621

# (-23.13275406039588, 0.0, 64, 79508,
# {'1%': -3.4304322497250084, '5%': -2.8615763529866034, '10%': -2.5667893494406826}, -51670.81110656279)

# Since the p-value is not less than .05, we fail to reject the null hypothesis, time series is non-stationary.
