import numpy as np
import pandas as pd
import re
import csv
import talib as ta
import os
import argparse
import json

parser = argparse.ArgumentParser()
# parser.add_argument('-device', default='cuda', type=str)
parser.add_argument('-is_preprocess', default=False, type=lambda x: (str(x).lower() == 'true'))
args, other_args = parser.parse_known_args()

JSON_LINES = 10000


def csv_to_df(filename, time_index_column):
    with open(filename, 'r') as read_obj:
        csv_reader = csv.reader(read_obj, delimiter=' ')
        rows = []
        i = 0
        for row in csv_reader:
            row = ','.join(row)
            # each line consist of e.x. "1654819200.0034804|" + json line,
            # to read it normally we need to remove that part
            websocket_time = row.split('.')[0]
            row = re.sub('\d+\.\d+\|', '', row)
            # add websocket time as a param
            row = row.replace('}', f',"{time_index_column}":{websocket_time + "}"}')
            # # remove miliseconds from timestamp (magic)
            # row = row.split('"m":')[0][:-4] + ',"m":' + row.split('"m":')[-1]
            rows.append(row)
            i += 1
            if JSON_LINES > 0 and i == JSON_LINES:
                break

        # join json objects
        json_obj = ','.join(rows)
        return pd.read_json(json_obj, lines=True)


# fill missing data with previous val
def fill_missing_rows(df, tid_col):
    # make sure  datetime is in chronological order and equidisant (constant time intervals)
    date_index = pd.date_range(start=df[tid_col].iloc[0], end=df[tid_col].iloc[-1], freq='S')
    df = df.set_index(tid_col).reindex(date_index, method='pad').reset_index().rename(columns={'index': tid_col})
    return df


def process_agg_trade(df, time_index_column):
    # drop "e" and "s" columns with repetative data
    df = df.drop(['e', 's'], axis=1)
    # drop "E" column, usually the difference between "Event time" and "Trade time"
    # is only in few milliseconds https://stackoverflow.com/questions/50688471/binance-event-time-vs-trade-time
    df = df.drop(['E', 'T'], axis=1)
    # drop columns "a", "f", "l" we don't need
    df = df.drop(['a', 'f', 'l'], axis=1)
    # drop None elements
    df = df.dropna()
    # calculate Volume = Price x Quantity for each row
    df['vol'] = (df['p'] * df['q']).astype(np.float32)

    # # trim miliseconds part from tmsp (last 3 digits), e.x. 1654819201 025
    # df['T'] = df['T'].astype(str).apply(lambda x: x[:-3])
    # # convert timestamp to human datetime
    # df['T'] = df['T'].astype(int).apply(lambda x: pd.Timestamp.fromtimestamp(x).to_pydatetime())

    # merge trade volumes executed at same second ('m', 'M', 'p' and 'q' columns won't be used)
    df = df[[time_index_column, 'vol']].copy().groupby(time_index_column).sum().reset_index()
    df = fill_missing_rows(df, time_index_column)

    # cumulative sum of last 10s Volume
    df['vol_last_10'] = df['vol'].rolling(min_periods=1, window=10).sum().reset_index()['vol'].astype(np.float32)
    timeperiod = 10
    df['vol_last_10_MA'] = ta.SMA(df['vol_last_10'], timeperiod=timeperiod)
    df['vol_last_10_EMA'] = ta.EMA(df['vol_last_10'], timeperiod=timeperiod)
    df['vol_last_10_MACD'] = ta.MACD(df['vol_last_10'])[0]
    # drop first timeperiod rows (don't know EMA and SMA)
    # + first 9 rows we don't know correct 'vol_last_10'
    df = df.drop(df.head(max(33, timeperiod - 1)).index)

    # shift to get Volume for next 10s
    df['vol_next_10'] = df['vol_last_10'].shift(-10)
    # create Target - Volume movement: (new price - old price) / old price
    df['vol_mov'] = (df['vol_next_10'] - df['vol_last_10']) / df['vol_last_10']

    # drop last 10 rows since 'vol_next_10' is None because of shift
    df = df.drop(df.tail(10).index)
    assert len(df) == len(df.dropna())
    return df


def process_book_ticker(df, time_index_column):
    # drop None elements
    df = df.dropna()
    # average Book Ticker values over same second
    df = df.copy().groupby(time_index_column).mean().reset_index()
    df = fill_missing_rows(df, time_index_column)

    # feature engineering
    df['market_spread'] = abs(df['b'] - df['a'])
    df['mid_price'] = (df['b'] + df['a']) / 2
    df['vol_imbalance'] = (df['B'] - df['A']) / (df['B'] - df['A'])
    return df.drop(['u'], axis=1)


def process_market_depth(df, time_index_column):
    # drop None elements
    df = df.dropna()

    # feature engineering
    # if list was serialized as string => convert to list (issue of to_csv())
    # deserialize_list = lambda l: [s.strip() for s in l[1:-1].split(',')]
    # df['a'] = df['a'].map(lambda l: deserialize_list(l) if isinstance(df['a'][0], str) else l)
    df['a'] = df['a'].map(lambda l: np.asarray(l).astype(np.float32))
    df['b'] = df['b'].map(lambda l: np.asarray(l).astype(np.float32))
    depth_func = lambda x: np.sum(x[:, 0] * x[:, 1])
    df['ask_depth'] = df['a'].map(lambda l: depth_func(l) if len(l) > 0 else 0)
    df['bid_depth'] = df['b'].map(lambda l: depth_func(l) if len(l) > 0 else 0)
    df['market_depth'] = df['ask_depth'] + df['bid_depth']
    cv = lambda x: (np.std(x) / np.mean(x)) * 100
    # coefficient of variation
    df['ask_cv'] = df['a'].map(lambda l: cv(l[:, 0]) if len(l) > 0 else 0)
    df['bid_cv'] = df['b'].map(lambda l: cv(l[:, 0]) if len(l) > 0 else 0)

    # sum depth feature over same second
    depth_df = df[[time_index_column, 'market_depth']].copy().groupby(time_index_column).sum().reset_index()
    # average cv feature over same second
    cv_df = df[[time_index_column, 'ask_cv', 'bid_cv']].copy().groupby(time_index_column).mean().reset_index()
    # tmp: return only features
    df = depth_df.set_index(time_index_column).join(cv_df.set_index(time_index_column)).reset_index()
    df = fill_missing_rows(df, time_index_column)
    return df


def merge_datasets(agg_trade_df, book_ticker_df, market_depth_df, time_index_column):
    # join together
    df = agg_trade_df.set_index(time_index_column).join(book_ticker_df.set_index(time_index_column)).reset_index()
    df = df.set_index(time_index_column).join(market_depth_df.set_index(time_index_column)).reset_index()
    return df


if __name__ == '__main__':
    dirname = './datasets/'
    time_index_column = 'websocket_time'

    # First step: preprocess
    if args.is_preprocess is True:
        agg_trade_df = csv_to_df(os.path.join(dirname, 'BTCUSDT_aggTrade.csv'), time_index_column)
        book_ticker_df = csv_to_df(os.path.join(dirname, 'BTCUSDT_bookTicker.csv'), time_index_column)
        market_depth_df = csv_to_df(os.path.join(dirname, 'BTCUSDT_depth@100ms.csv'), time_index_column)

        agg_trade_df.to_pickle(os.path.join(dirname, 'agg_trade.pkl'))
        book_ticker_df.to_pickle(os.path.join(dirname, 'book_ticker.pkl'))
        market_depth_df.to_pickle(os.path.join(dirname, 'market_depth.pkl'))
        exit(0)

        # save results (untill process not fully parallelized to avoid mem kill)
        # saving as to_csv is a shit, its serializing lists to string and other objects, datetimes
        # which is not directly convertible with pd.read_csv()
        # agg_trade_df.to_csv(os.path.join(dirname, 'agg_trade.csv'), index=False)
    else:
        # agg_trade_df = pd.read_csv(os.path.join(dirname, 'agg_trade.csv'), parse_dates=[time_index_column])
        agg_trade_df = pd.read_pickle(os.path.join(dirname, 'agg_trade.pkl'))
        book_ticker_df = pd.read_pickle(os.path.join(dirname, 'book_ticker.pkl'))
        market_depth_df = pd.read_pickle(os.path.join(dirname, 'market_depth.pkl'))

    # Second step: feature engineering
    agg_trade_df = process_agg_trade(agg_trade_df, time_index_column)
    book_ticker_df = process_book_ticker(book_ticker_df, time_index_column)
    market_depth_df = process_market_depth(market_depth_df, time_index_column)
    df = merge_datasets(agg_trade_df, book_ticker_df, market_depth_df, time_index_column)
    # save processed dataframe to csv
    df.to_csv('./data.csv', index=False)

    # TODO: parallel processing of json obj not working
    # for chunk in pd.read_json(json_obj, lines=True, chunksize=chunksize, nrows=chunksize):
    # TODO: sanity checks - e.x. all rows are with BTC/USDT quotes and etc.

'''
    # add seasonality with datetime's hour Fourier Transform
    df['hour'] = df['T'].dt.hour
    # f(x) = 2*pi*x/24, where 24 are hours in a day
    df['hour_sin'] = df['hour'].apply(lambda x: np.sin(np.pi * x / 12))
    df['hour_cos'] = df['hour'].apply(lambda x: np.cos(np.pi * x / 12))
'''
