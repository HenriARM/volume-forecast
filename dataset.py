import numpy as np
import pandas as pd
import re
import csv

TMPI = 10000


def process(chunk):
    #################### FEATURE ENGINEERING #################################
    # TODO: sanity checks - e.x. all rows are with BTC/USDT quotes and etc.
    # drop "e" and "s" columns with repetative data
    chunk = chunk.drop(['e', 's'], axis=1)
    # drop "E" column, usually the difference between "Event time" and "Trade time"
    # is only in few milliseconds https://stackoverflow.com/questions/50688471/binance-event-time-vs-trade-time
    chunk = chunk.drop(['E'], axis=1)
    # drop columns "a", "f", "l" we don't need
    chunk = chunk.drop(['a', 'f', 'l'], axis=1)
    # drop None elements
    chunk = chunk.dropna()
    # calculate Volume = Price x Quantity for each row
    chunk['vol'] = chunk['p'] * chunk['q']
    # trim miliseconds part from tmsp (last 3 digits), e.x. 1654819201 025
    chunk['T'] = chunk['T'].astype(str).apply(lambda x: x[:-3])
    # convert timestamp to human datetime
    chunk['T'] = chunk['T'].astype(int).apply(lambda x: pd.Timestamp.fromtimestamp(x).to_pydatetime())
    # merge Volumes of trades from same second
    train_df = chunk[['T', 'vol']].copy().groupby('T').sum().reset_index()

    # check that datetime is in chronological order and equidisant (constant time intervals)
    # chron_date = pd.date_range(start=train_df['T'].iloc[0], end=train_df['T'].iloc[-1], freq='1S')
    # # append zero rows
    # train_df = train_df.append([pd.Series(0, index=train_df.columns) for _ in range(len(chron_date) - len(train_df))],
    #                            ignore_index=True)
    # TODO: fill missing dates (pay attention that append changes type of 'T' from datetime to object )
    # train_df['col'] = chron_date

    # add seasonality with datetime's hour Fourier Transform
    train_df['hour'] = train_df['T'].dt.hour
    # f(x) = 2*pi*x/24, where 24 are hours in a day
    train_df['hour_sin'] = train_df['hour'].apply(lambda x: np.sin(np.pi * x / 12))
    train_df['hour_cos'] = train_df['hour'].apply(lambda x: np.cos(np.pi * x / 12))

    # cumulative sum of last 10s Volume
    train_df['vol_prev_10'] = train_df['vol'].rolling(min_periods=1, window=10).sum().reset_index()['vol']
    # shift to get Volume for next 10s
    train_df['vol_next_10'] = train_df['vol_prev_10'].shift(-10)
    # drop first 9 rows (don't know total Volume of first 9 seconds)
    train_df = train_df.drop(list(range(9)))
    # create Target - Volume movement: (new price - old price) / old price
    train_df['vol_mov'] = (train_df['vol_next_10'] - train_df['vol_prev_10']) / train_df['vol_prev_10']

    # print(chunk.iloc[0])
    # print(chunk.info())
    # print(chunk.columns)
    # print(train_df.head(50))

    # save processed dataframe to csv
    train_df[['vol_prev_10', 'vol_next_10', 'vol_mov']].to_csv('./train.csv')
    # train_df[['vol_prev_10', 'vol_mov']].to_csv('./train.csv')


def fix_json_lines(filename, chunksize):
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
            # TODO:tmp rows to use from dataset
            if i == TMPI:
                break

        # join json objects
        json_obj = ','.join(rows)
        # TODO: chunksize and nrows not working for parallel processing of frame
        for chunk in pd.read_json(json_obj, lines=True, chunksize=chunksize, nrows=chunksize):
            process(chunk)


if __name__ == '__main__':
    fix_json_lines(filename='./BTCUSDT_aggTrade.csv', chunksize=500)
