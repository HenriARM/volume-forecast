import numpy as np
import pandas as pd
import re
import csv
import talib as ta

JSON_LINES = 1000


# fill missing data with previous val
def fill_missing_rows(df, tid_col):
    # make sure  datetime is in chronological order and equidisant (constant time intervals)
    date_index = pd.date_range(start=df[tid_col].iloc[0], end=df[tid_col].iloc[-1], freq='S')
    df = df.set_index(tid_col).reindex(date_index, method='pad').reset_index().rename(columns={'index': tid_col})
    return df


def process(agg_trade_df, book_ticker_df, is_extra):
    # TODO: sanity checks - e.x. all rows are with BTC/USDT quotes and etc.
    time_index_column = 'websocket_time'

    # Agg Trade
    # drop "e" and "s" columns with repetative data
    agg_trade_df = agg_trade_df.drop(['e', 's'], axis=1)
    # drop "E" column, usually the difference between "Event time" and "Trade time"
    # is only in few milliseconds https://stackoverflow.com/questions/50688471/binance-event-time-vs-trade-time
    agg_trade_df = agg_trade_df.drop(['E', 'T'], axis=1)
    # drop columns "a", "f", "l" we don't need
    agg_trade_df = agg_trade_df.drop(['a', 'f', 'l'], axis=1)
    # drop None elements
    agg_trade_df = agg_trade_df.dropna()
    # calculate Volume = Price x Quantity for each row
    agg_trade_df['vol'] = (agg_trade_df['p'] * agg_trade_df['q']).astype(np.float32)

    # # trim miliseconds part from tmsp (last 3 digits), e.x. 1654819201 025
    # agg_trade_df['T'] = agg_trade_df['T'].astype(str).apply(lambda x: x[:-3])
    # # convert timestamp to human datetime
    # agg_trade_df['T'] = agg_trade_df['T'].astype(int).apply(lambda x: pd.Timestamp.fromtimestamp(x).to_pydatetime())

    # merge trade volumes executed at same second ('m', 'M', 'p' and 'q' columns won't be used)
    agg_trade_df = agg_trade_df[[time_index_column, 'vol']].copy().groupby(time_index_column).sum().reset_index()
    agg_trade_df = fill_missing_rows(agg_trade_df, time_index_column)

    # Book Ticker
    # drop None elements
    book_ticker_df = book_ticker_df.dropna()
    # average Book Ticker values over same second
    book_ticker_df = book_ticker_df.copy().groupby(time_index_column).mean().reset_index()
    book_ticker_df = fill_missing_rows(book_ticker_df, time_index_column)

    # join together
    df = agg_trade_df.set_index(time_index_column).join(book_ticker_df.set_index(time_index_column)).reset_index()
    df['market_spread'] = abs(df['b'] - df['a'])
    # TODO: finish

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


def csv_to_df(filename):
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
            row = row.replace('}', f',"websocket_time":{websocket_time + "}"}')
            # # remove miliseconds from timestamp (magic)
            # row = row.split('"m":')[0][:-4] + ',"m":' + row.split('"m":')[-1]
            rows.append(row)
            i += 1
            if JSON_LINES > 0 and i == JSON_LINES:
                break

        # join json objects
        json_obj = ','.join(rows)
        return pd.read_json(json_obj, lines=True)


def generate_dataset(agg_trade, book_ticker, market_depth, is_extra):
    agg_trade_df = csv_to_df(filename=agg_trade)
    book_ticker_df = csv_to_df(filename=book_ticker)
    # market_depth_df = csv_to_df(filename=market_depth)
    process(agg_trade_df, book_ticker_df, is_extra)
    # TODO: save csv_to_df results with averagin over rows and then merge_datasets() do after
    # TODO: separate process_agg_trade and process_book_ticker functions


if __name__ == '__main__':
    generate_dataset(
        agg_trade='./datasets/BTCUSDT_aggTrade.csv',
        book_ticker='./datasets/BTCUSDT_bookTicker.csv',
        market_depth='./datasets/BTCUSDT_depth@100ms.csv',
        is_extra=True)

    # TODO: parallel processing of json obj not working
    # for chunk in pd.read_json(json_obj, lines=True, chunksize=chunksize, nrows=chunksize):
