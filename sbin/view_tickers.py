import sqlite3
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

def load_ohlcv(db_path, table_name, ticker):
    df = None
    max_retries = 3
    retry_delay = 10
    query = f"""
        SELECT *
        FROM {table_name}
        WHERE ticker = ?
        ORDER BY date ASC
    """

    for attempt in range(1, max_retries + 1):
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql(query, conn, params=(ticker,))
            conn.close()
            return df

        except Exception as e:
            print(f"[ERROR] DB Load 실패 (attempt {attempt}/{max_retries}): {e}")

            # 마지막 시도 실패면 예외 던짐
            if attempt == max_retries:
                raise

            # 대기 후 재시도
            time.sleep(retry_delay)

    # 논리적으로 여기까지 오면 안 오지만 안정성 위해
    raise RuntimeError("DB load retry failed unexpectedly.")
    return df

db_path = '/Users/yongbeom/cyb/project/2025/quant/var/data/coin_ohlcv_minute1.db'
db_path = '/Users/yongbeom/cyb/project/2025/quant/var/data/coin_ohlcv_month.db'
table_name = "coin_ohlcv_minute1"
table_name = "coin_ohlcv_month"
conn = sqlite3.connect(db_path)
cur = conn.cursor()
column = "ticker"

cur.execute(f"SELECT DISTINCT {column} FROM {table_name}")
tickers = [row[0] for row in cur.fetchall()]
print(tickers)
#for ticker in tickers:
#    df = load_ohlcv(db_path, table_name, ticker)
#    print(ticker, len(df))
#    # if ticker == 'KRW-WAVES':
#    #     print(df.head())

conn.close()
