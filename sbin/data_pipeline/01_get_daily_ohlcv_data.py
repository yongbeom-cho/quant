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
import sqlite3

pd.options.display.max_columns = None
pd.options.display.max_rows = None

def isNotDataframeOrEmpty(df):
    return not isinstance(df, pd.core.frame.DataFrame) or (isinstance(df, pd.core.frame.DataFrame) and df.empty)


def get_ohlcv(code: str, interval: str, count: int, to: str):
    """Get OHLCV data with retry if data count is less than requested count."""
    dfs = []
    merged = None
    max_retries = 3
    
    for retry in range(max_retries):
        df = pyupbit.get_ohlcv(code, interval=interval, count=count, to=to).drop_duplicates()
        time.sleep(0.11)
        if isNotDataframeOrEmpty(df):
            continue
        dfs.append(df)
        if len(dfs) == 1:
            merged = df
        elif len(dfs) > 1:
            merged = pd.concat(dfs).sort_index().drop_duplicates()

        if len(merged) == count:
            break
    
    if not dfs:
        return None
    
    # Merge all retry results
    merged = pd.concat(dfs)
    merged = merged.sort_index().drop_duplicates()
    
    return merged


def fetch_coin_ohlcv(code: str, interval: str, str_start_dt=None, str_end_dt=None):
    dfs = []
    
    str_end_dt = str_end_dt if str_end_dt else datetime.datetime.now().strftime("%Y-%m-%d")
    to_cursor = str_end_dt
    max_iter = 100000  # safety guard

    for _ in range(max_iter):
        print("%s\t%s\t%d" %(code, interval, _))
        if interval.startswith("minute"):
            count = 1440
        else:
            count = 200

        df = get_ohlcv(code, interval=interval, count=count, to=to_cursor)
        if isNotDataframeOrEmpty(df):
            break
        # print(df.head())
        dfs.append(df)

        str_earliest_date = df.index.min().strftime("%Y-%m-%d")

        if str_start_dt and str_earliest_date <= str_start_dt:
            break

        to_cursor = str_earliest_date
        time.sleep(0.11)  # throttle to avoid rate-limit

    if not dfs:
        return None

    merged = pd.concat(dfs)
    if str_start_dt and str_end_dt:
        start_dt = datetime.datetime.strptime(str_start_dt, "%Y-%m-%d") \
                        .replace(hour=9, minute=0, second=0, microsecond=0)
        end_dt = datetime.datetime.strptime(str_end_dt, "%Y-%m-%d") \
                        .replace(hour=9, minute=0, second=0, microsecond=0)
        merged = merged[(merged.index >= start_dt) & (merged.index < end_dt)]
    merged = merged.sort_index().drop_duplicates()
    
    return merged

def clean_and_save_to_db(df: pd.DataFrame, conn: sqlite3.Connection, table_name: str, ticker: str, interval: str):
    """Save OHLCV data to SQLite database table with upsert logic."""
    if 'value' in df.columns:
        df = df.drop('value', axis=1)
    
    df['ticker'] = ticker
    df['date'] = df.index
    df['date'] = df['date'].apply(lambda d: d.tz_localize(None) if getattr(d, "tzinfo", None) else d)

    # date formatting depending on interval granularity
    if interval.startswith("minute"):
        fmt = "%Y%m%d%H%M"
    else:
        fmt = "%Y%m%d"
    df['date'] = df['date'].apply(lambda d: d.strftime(fmt))

    df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
    
    # Use INSERT OR REPLACE for upsert
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute(f"""
            INSERT OR REPLACE INTO {table_name} (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (row['ticker'], row['date'], row['open'], row['high'], row['low'], row['close'], row['volume']))
    conn.commit()


def create_table_if_not_exists(conn: sqlite3.Connection, table_name: str):
    """Create OHLCV table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()


#--market : [coin, korea, overseas, ...]
#--interval : [day, minute1, minute3, minute5, minute10, minute15, minute30, minute60, minute240, week, month]
#--date : {YYYYMMDD or "all"}
#--output_dir : {output_dir for db file}
parser = argparse.ArgumentParser(description='get_daily_ohlcv_data')
parser.add_argument('--root_dir', type=str, default="/Users/yongbeom/project/coin_volume_trader")
parser.add_argument('--date', type=str, required=True, help="YYYYMMDD or all")
parser.add_argument('--market', type=str, default="coin")
parser.add_argument('--interval', type=str, default="minute1")
parser.add_argument('--output_dir', type=str, default="var/data")
args = parser.parse_args()

# Create output directory for db file
output_dir = os.path.join(args.root_dir, args.output_dir)
os.makedirs(output_dir, exist_ok=True)

# DB file path: {output_dir}/{market}_ohlcv_{interval}.db
db_path = os.path.join(output_dir, f"{args.market}_ohlcv_{args.interval}.db")
conn = sqlite3.connect(db_path)

ALLOWED_INTERVALS = ["day", "minute1", "minute3", "minute5", "minute10", "minute15", "minute30", "minute60", "minute240", "week", "month"]

if args.interval not in ALLOWED_INTERVALS:
    print(f"Unsupported interval: {args.interval}. Use one of {ALLOWED_INTERVALS}")
    sys.exit(1)

interval = args.interval

str_start_dt = None
str_end_dt = None
if args.date.lower() == "all":
    str_end_dt = datetime.datetime.now().strftime("%Y-%m-%d")
else:
    # args.date에 어제의 날짜가 들어옴.
    date = datetime.datetime.strptime(args.date, "%Y%m%d")
    str_start_dt = datetime.datetime(date.year, date.month, date.day).strftime("%Y-%m-%d")
    str_end_dt = (start_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")


if args.market == 'coin':
    tickers = [t for t in pyupbit.get_tickers() if 'KRW-' in t]
    # tickers = ["KRW-BTC"]
    print("tickers len :", len(tickers))
    
    # Table name: {market}_ohlcv_{interval}
    table_name = f"{args.market}_ohlcv_{args.interval}"
    create_table_if_not_exists(conn, table_name)
    print(f"Using table: {table_name} in DB: {db_path}")

    for ticker_i, ticker in enumerate(tickers):
        print(f"start {ticker_i+1}. {ticker}")
        try:
            df = fetch_coin_ohlcv(ticker, interval, str_start_dt, str_end_dt)
            if isNotDataframeOrEmpty(df):
                print(f"{ticker} - no data")
                continue
            
            clean_and_save_to_db(df, conn, table_name, ticker, interval)
            print(f"{ticker} - saved {len(df)} records")
        except Exception:
            print(f"error on {ticker}")
            print(traceback.format_exc())
            time.sleep(1.0)
    
    conn.close()
    print(f"All data saved to {db_path}")