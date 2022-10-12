import numpy as np
import pandas as pd
import re
import csv
import talib as ta

JSON_LINES = 1000


def process(df, is_extra):
    #################### FEATURE ENGINEERING #################################
    # TODO: sanity checks - e.x. all rows are with BTC/USDT quotes and etc.
    # drop "e" and "s" columns with repetative data
    df = df.drop(['e', 's'], axis=1)
    # drop "E" column, usually the difference between "Event time" and "Trade time"
    # is only in few milliseconds https://stackoverflow.com/questions/50688471/binance-event-time-vs-trade-time
    df = df.drop(['E'], axis=1)
    # drop columns "a", "f", "l" we don't need
    df = df.drop(['a', 'f', 'l'], axis=1)
    # drop None elements
    df = df.dropna()
    # calculate Volume = Price x Quantity for each row
    df['vol'] = (df['p'] * df['q']).astype(np.float32)
    # trim miliseconds part from tmsp (last 3 digits), e.x. 1654819201 025
    df['T'] = df['T'].astype(str).apply(lambda x: x[:-3])
    # convert timestamp to human datetime
    df['T'] = df['T'].astype(int).apply(lambda x: pd.Timestamp.fromtimestamp(x).to_pydatetime())
    # merge trade volumes executed at same second ('m', 'M', 'p' and 'q' columns won't be used)
    df = df[['T', 'vol']].copy().groupby('T').sum().reset_index()

    # fill missing data with previous val
    # make sure  datetime is in chronological order and equidisant (constant time intervals)
    date_index = pd.date_range(start=df['T'].iloc[0], end=df['T'].iloc[-1], freq='S')
    df = df.set_index('T').reindex(date_index, method='pad').reset_index().rename(columns={'index': 'T'})

    # cumulative sum of last 10s Volume
    df['vol_last_10'] = df['vol'].rolling(min_periods=1, window=10).sum().reset_index()['vol'].astype(np.float32)
    # shift to get Volume for next 10s
    df['vol_next_10'] = df['vol_last_10'].shift(-10)
    # create Target - Volume movement: (new price - old price) / old price
    df['vol_mov'] = (df['vol_next_10'] - df['vol_last_10']) / df['vol_last_10']

    if is_extra is True:
        # # add seasonality with datetime's hour Fourier Transform
        # df['hour'] = df['T'].dt.hour
        # # f(x) = 2*pi*x/24, where 24 are hours in a day
        # df['hour_sin'] = df['hour'].apply(lambda x: np.sin(np.pi * x / 12))
        # df['hour_cos'] = df['hour'].apply(lambda x: np.cos(np.pi * x / 12))

        timeperiod = 10
        df['vol_last_10_MA'] = ta.SMA(df['vol_last_10'], timeperiod=timeperiod)
        df['vol_last_10_EMA'] = ta.EMA(df['vol_last_10'], timeperiod=timeperiod)
        df['vol_last_10_MACD'] = ta.MACD(df['vol_last_10'])[0]
        # drop first timeperiod rows (don't know EMA and SMA)
        df = df.drop(df.head(max(33, timeperiod - 1)).index)
        # ---- TODO: add more features here
    else:
        # drop first 9 rows since don't know correct 'vol_last_10' for them)
        df = df.drop(df.head(9).index)
    # drop last 10 rows since 'vol_next_10' is None because of shift
    df = df.drop(df.tail(10).index)
    assert len(df) == len(df.dropna())
    # save processed dataframe to csv
    df.to_csv('./data.csv', index=False)


def generate_dataset(filename, chunksize, is_extra):
    with open(filename, 'r') as read_obj:
        csv_reader = csv.reader(read_obj, delimiter=' ')
        rows = []
        i = 0
        for row in csv_reader:
            row = ','.join(row)
            # each line consist of e.x. "1654819200.0034804|" + json line,
            # to read it normally we need to remove first part
            row = re.sub('\d+\.\d+\|', '', row)
            rows.append(row)
            i += 1
            if JSON_LINES > 0 and i == JSON_LINES:
                break

        # join json objects
        json_obj = ','.join(rows)
        # TODO: chunksize and nrows not working for parallel processing of frame
        for chunk in pd.read_json(json_obj, lines=True, chunksize=chunksize, nrows=chunksize):
            print(len(chunk))
            process(chunk, is_extra)


if __name__ == '__main__':
    generate_dataset(filename='./datasets/BTCUSDT_aggTrade.csv', chunksize=10000, is_extra=True)
