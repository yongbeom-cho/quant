"""
Expanding 값 캐시 관리 모듈
매일 아침 06:15에 expanding 값을 미리 계산하여 저장하고, 
실시간 거래 시 저장된 값을 사용하도록 함
"""
import sqlite3
import pandas as pd
import numpy as np
import talib
import os
import time
from datetime import datetime


def get_db_path(root_dir, interval):
    """interval에 맞는 DB 경로 반환"""
    interval_map = {
        'day': 'day',
        'minute60': 'minute60',
        'minute240': 'minute240'
    }
    db_interval = interval_map.get(interval, interval)
    return os.path.join(root_dir, f'var/data/coin_ohlcv_{db_interval}.db')


def get_table_name(interval):
    """interval에 맞는 테이블 이름 반환"""
    interval_map = {
        'day': 'coin_ohlcv_day',
        'minute60': 'coin_ohlcv_minute60',
        'minute240': 'coin_ohlcv_minute240'
    }
    return interval_map.get(interval, f'coin_ohlcv_{interval}')


def load_ohlcv_from_db(db_path, table_name, ticker):
    """DB에서 OHLCV 데이터 로드"""
    max_retries = 3
    retry_delay = 1
    
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
            
            if df is not None and len(df) > 0:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            return df
        except Exception as e:
            print(f"[ERROR] DB Load 실패 (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                raise
            time.sleep(retry_delay)
    
    return None


def get_tickers_from_db(db_path, table_name):
    """DB에서 모든 ticker 목록 가져오기"""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT DISTINCT ticker FROM {table_name}")
        tickers = [row[0] for row in cur.fetchall()]
        conn.close()
        return tickers
    except Exception as e:
        print(f"[ERROR] Ticker 목록 가져오기 실패: {e}")
        return []


def calculate_expanding_values_day(df):
    """Day interval용 expanding 값 계산"""
    if df is None or len(df) == 0:
        return None, None, None
    
    # bb_range 계산
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = talib.BBANDS(
        df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    bb_range = df["bb_upper"] - df["bb_lower"]
    
    # bb_std (전체 데이터의 std 계산)
    bb_range_shifted = bb_range.shift(1)
    bb_std_last = bb_range_shifted.std() if len(bb_range_shifted) > 0 and not bb_range_shifted.isna().all() else None
    
    # feat0 계산 (feat46에 필요)
    v20 = talib.SMA(df['volume'], timeperiod=20)
    v2 = talib.SMA(df['volume'], timeperiod=2)
    feat0 = v2 / v20.shift(5)
    
    # feat0 mean/std (전체 데이터의 mean/std 계산)
    feat0_shifted = feat0.shift(1)
    feat0_mean_last = feat0_shifted.mean() if len(feat0_shifted) > 0 and not feat0_shifted.isna().all() else None
    feat0_std_last = feat0_shifted.std() if len(feat0_shifted) > 0 and not feat0_shifted.isna().all() else None
    
    return bb_std_last, feat0_mean_last, feat0_std_last


def calculate_expanding_values_minute60(df):
    """Minute60 interval용 expanding 값 계산"""
    if df is None or len(df) == 0:
        return None, None
    
    # feat0 계산 (feat46에 필요)
    v20 = talib.SMA(df['volume'], timeperiod=20)
    v2 = talib.SMA(df['volume'], timeperiod=2)
    feat0 = v2 / v20.shift(5)
    
    # feat0 mean/std (전체 데이터의 mean/std 계산)
    feat0_shifted = feat0.shift(1)
    feat0_mean_last = feat0_shifted.mean() if len(feat0_shifted) > 0 and not feat0_shifted.isna().all() else None
    feat0_std_last = feat0_shifted.std() if len(feat0_shifted) > 0 and not feat0_shifted.isna().all() else None
    
    return feat0_mean_last, feat0_std_last


def calculate_expanding_values_minute240(df):
    """Minute240 interval용 expanding 값 계산"""
    if df is None or len(df) == 0:
        return None, None
    
    # feat0 계산 (feat46에 필요)
    v20 = talib.SMA(df['volume'], timeperiod=20)
    v2 = talib.SMA(df['volume'], timeperiod=2)
    feat0 = v2 / v20.shift(5)
    
    # feat0 mean/std (전체 데이터의 mean/std 계산)
    feat0_shifted = feat0.shift(1)
    feat0_mean_last = feat0_shifted.mean() if len(feat0_shifted) > 0 and not feat0_shifted.isna().all() else None
    feat0_std_last = feat0_shifted.std() if len(feat0_shifted) > 0 and not feat0_shifted.isna().all() else None
    
    return feat0_mean_last, feat0_std_last


def create_cache_db(cache_db_path):
    """캐시 DB 생성"""
    conn = sqlite3.connect(cache_db_path)
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS expanding_cache (
            interval TEXT,
            ticker TEXT,
            bb_std REAL,
            feat0_expanding_mean REAL,
            feat0_expanding_std REAL,
            updated_at TEXT,
            PRIMARY KEY (interval, ticker)
        )
    """)
    
    conn.commit()
    conn.close()


def save_expanding_values(cache_db_path, interval, ticker, bb_std=None, feat0_mean=None, feat0_std=None):
    """expanding 값 저장"""
    create_cache_db(cache_db_path)
    
    conn = sqlite3.connect(cache_db_path)
    cur = conn.cursor()
    
    updated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    cur.execute("""
        INSERT OR REPLACE INTO expanding_cache 
        (interval, ticker, bb_std, feat0_expanding_mean, feat0_expanding_std, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (interval, ticker, bb_std, feat0_mean, feat0_std, updated_at))
    
    conn.commit()
    conn.close()


def load_expanding_values(cache_db_path, interval, ticker):
    """expanding 값 로드"""
    try:
        conn = sqlite3.connect(cache_db_path)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT bb_std, feat0_expanding_mean, feat0_expanding_std
            FROM expanding_cache
            WHERE interval = ? AND ticker = ?
        """, (interval, ticker))
        
        result = cur.fetchone()
        conn.close()
        
        if result:
            return {
                'bb_std': result[0],
                'feat0_expanding_mean': result[1],
                'feat0_expanding_std': result[2]
            }
        return None
    except Exception as e:
        print(f"[ERROR] Expanding 값 로드 실패: {e}")
        return None


def check_expanding_cache_exists(root_dir):
    """expanding cache 파일이 존재하고 데이터가 있는지 확인"""
    cache_db_path = os.path.join(root_dir, 'upbit_auto_trader', 'expanding_cache.db')
    
    if not os.path.exists(cache_db_path):
        return False
    
    try:
        conn = sqlite3.connect(cache_db_path)
        cur = conn.cursor()
        
        # 캐시에 데이터가 있는지 확인
        cur.execute("SELECT COUNT(*) FROM expanding_cache")
        count = cur.fetchone()[0]
        conn.close()
        
        return count > 0
    except Exception as e:
        print(f"[WARNING] Expanding cache 확인 실패: {e}")
        return False


def precompute_all_expanding_values(root_dir):
    """모든 interval-ticker에 대해 expanding 값 미리 계산"""
    cache_db_path = os.path.join(root_dir, 'upbit_auto_trader', 'expanding_cache.db')
    create_cache_db(cache_db_path)
    
    intervals = ['day', 'minute60', 'minute240']
    
    for interval in intervals:
        print(f"[INFO] Processing interval: {interval}")
        db_path = get_db_path(root_dir, interval)
        table_name = get_table_name(interval)
        
        if not os.path.exists(db_path):
            print(f"[WARNING] DB 파일이 없습니다: {db_path}")
            continue
        
        tickers = get_tickers_from_db(db_path, table_name)
        print(f"[INFO] Found {len(tickers)} tickers for {interval}")
        
        for idx, ticker in enumerate(tickers, 1):
            try:
                df = load_ohlcv_from_db(db_path, table_name, ticker)
                
                if df is None or len(df) == 0:
                    print(f"[WARNING] {ticker}: 데이터가 없습니다")
                    continue
                
                if interval == 'day':
                    bb_std, feat0_mean, feat0_std = calculate_expanding_values_day(df)
                    save_expanding_values(cache_db_path, interval, ticker, bb_std, feat0_mean, feat0_std)
                elif interval == 'minute60':
                    feat0_mean, feat0_std = calculate_expanding_values_minute60(df)
                    save_expanding_values(cache_db_path, interval, ticker, None, feat0_mean, feat0_std)
                elif interval == 'minute240':
                    feat0_mean, feat0_std = calculate_expanding_values_minute240(df)
                    save_expanding_values(cache_db_path, interval, ticker, None, feat0_mean, feat0_std)
                
                if idx % 10 == 0:
                    print(f"[INFO] {interval}: {idx}/{len(tickers)} tickers processed")
                    
            except Exception as e:
                print(f"[ERROR] {ticker} 처리 실패: {e}")
                continue
        
        print(f"[INFO] Completed interval: {interval}")
    
    print(f"[INFO] All expanding values precomputed and saved to {cache_db_path}")


if __name__ == "__main__":
    import sys
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/yongbeom/cyb/project/2025/quant"
    precompute_all_expanding_values(root_dir)

