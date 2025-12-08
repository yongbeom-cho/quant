import FinanceDataReader as fdr
import time
import datetime
import os
import pandas as pd
import pykrx as krx
import argparse
import pyupbit
import traceback
import sys

#pd.options.display.max_columns = None
#pd.options.display.max_rows = None

def isNotDataframeOrEmpty(df):
    return not isinstance(df, pd.core.frame.DataFrame) or (isinstance(df, pd.core.frame.DataFrame) and df.empty)

def fetch_coin_ohlcv(code: str, interval: str, start_dt=None, end_dt=None):
    """
    Fetch candles for a date window or all history.
    - "all": start_dt is None, end_dt is today; walk backward until no data.
    - specific date: fetch [start_dt, end_dt).
    Cursor moves to the earliest timestamp returned each batch (no fixed step).
    """
    dfs = []
    to_cursor = end_dt if end_dt else datetime.datetime.now()
    max_iter = 100000  # safety guard
    last_earliest = None
    for _ in range(max_iter):
        df = pyupbit.get_ohlcv(code, interval=interval, count=200, to=to_cursor)
        if isNotDataframeOrEmpty(df):
            break

        dfs.append(df)

        earliest = df.index.min()
        if start_dt and earliest <= start_dt:
            break

        if last_earliest is not None and earliest >= last_earliest:
            break
        last_earliest = earliest

        to_cursor = earliest  # move cursor using the earliest timestamp from the batch
        time.sleep(0.11)  # throttle to avoid rate-limit

    if not dfs:
        return None

    merged = pd.concat(dfs)
    if start_dt and end_dt:
        merged = merged[(merged.index >= start_dt) & (merged.index < end_dt)]
    merged = merged.sort_index().drop_duplicates()
    return merged

def clean_and_save(df: pd.DataFrame, path: str):
    if 'value' in df.columns:
        df = df.drop('value', axis=1)
    df['date'] = df.index
    df['date'] = df['date'].apply(lambda d: d.tz_localize(None) if getattr(d, "tzinfo", None) else d)

    # date formatting depending on interval granularity
    if interval.startswith("minute"):
        fmt = "%Y%m%d%H%M"
    else:
        fmt = "%Y%m%d"
    df['date'] = df['date'].apply(lambda d: d.strftime(fmt))

    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
    df.to_excel(path, index=False)


#--market : [coin, korea, overseas, ...]
#--interval : [day, minute1, minute3, minute5, minute10, minute15, minute30, minute60, minute240, week, month]
#--date : {YYYYMMDD or "all"}
#--output_dir : {output_dir}
parser = argparse.ArgumentParser(description='get_daily_ohlcv_data')
parser.add_argument('--root_dir', type=str, default="/Users/yongbeom/project/coin_volume_trader")
parser.add_argument('--date', type=str, required=True, help="YYYYMMDD or all")
parser.add_argument('--market', type=str, default="coin")
parser.add_argument('--interval', type=str, default="minute1")
parser.add_argument('--output_dir', type=str, default="var/data/ohlcv_minute1")
args = parser.parse_args()

output_dir = os.path.join(args.root_dir, args.output_dir)
os.makedirs(output_dir, exist_ok=True)

ALLOWED_INTERVALS = ["day", "minute1", "minute3", "minute5", "minute10", "minute15", "minute30", "minute60", "minute240", "week", "month"]

if args.interval not in ALLOWED_INTERVALS:
    print(f"Unsupported interval: {args.interval}. Use one of {ALLOWED_INTERVALS}")
    sys.exit(1)

interval = args.interval

# get start and end date (for "all", start_dt=None, end_dt=today)
if args.date.lower() == "all":
    start_dt = None
    end_dt = datetime.datetime.now()
else:
    date = datetime.datetime.strptime(args.date, "%Y%m%d")
    start_dt = datetime.datetime(date.year, date.month, date.day)
    end_dt = start_dt + datetime.timedelta(days=1)


if args.market == 'coin':
    tickers = [t for t in pyupbit.get_tickers() if 'KRW-' in t]
    print("tickers len :", len(tickers))

    for ticker_i, ticker in enumerate(tickers):
        print(f"start {ticker_i+1}. {ticker}")
        write_path = os.path.join(output_dir, f"{ticker}.xlsx")
        try:
            df = fetch_coin_ohlcv(ticker, interval, start_dt, end_dt)
            if isNotDataframeOrEmpty(df):
                print(f"{ticker} - no data")
                continue
            clean_and_save(df, write_path)
            print(write_path, 'get data END', len(df))
        except Exception:
            print(f"error on {ticker}")
            print(traceback.format_exc())
            time.sleep(1.0)