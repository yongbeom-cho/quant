"""
데이터 로더 유틸리티

SQLite 데이터베이스에서 OHLCV 데이터를 로드하는 유틸리티 함수들입니다.
"""

import os
import sqlite3
from typing import List, Dict, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def get_tickers(db_path: str, table_name: str) -> List[str]:
    """
    DB에서 모든 종목 코드 조회
    
    Args:
        db_path: SQLite DB 파일 경로
        table_name: 테이블 이름 (예: 'coin_ohlcv_minute60')
        
    Returns:
        종목 코드 리스트
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT ticker FROM {table_name}")
    tickers = [row[0] for row in cur.fetchall()]
    conn.close()
    return tickers


def load_single_ticker(args: tuple) -> tuple:
    """
    단일 종목 OHLCV 데이터 로드 (ThreadPoolExecutor용)
    
    Args:
        args: (db_path, table_name, ticker, start_date, end_date)
        
    Returns:
        (ticker, DataFrame or None)
    """
    db_path, table_name, ticker, start_date, end_date = args
    
    try:
        conn = sqlite3.connect(db_path)
        
        query = f"""
            SELECT *
            FROM {table_name}
            WHERE ticker = ?
        """
        params = [ticker]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date ASC"
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        
        return ticker, df
    except Exception as e:
        print(f"[ERROR] Failed to load {ticker}: {e}")
        return ticker, None


def load_ohlcv_data(
    db_path: str,
    table_name: str,
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_workers: int = 10,
    min_bars: int = 50
) -> Dict[str, pd.DataFrame]:
    """
    여러 종목의 OHLCV 데이터 병렬 로드
    
    Args:
        db_path: SQLite DB 파일 경로
        table_name: 테이블 이름
        tickers: 로드할 종목 리스트 (None이면 전체)
        start_date: 시작 일자 (YYYY-MM-DD 형식)
        end_date: 종료 일자
        max_workers: 병렬 로드 워커 수
        min_bars: 최소 봉 개수 (이하면 제외)
        
    Returns:
        {ticker: DataFrame} 딕셔너리
    """
    # 종목 목록 로드
    if tickers is None:
        tickers = get_tickers(db_path, table_name)
    
    # 병렬 로드
    args_list = [(db_path, table_name, t, start_date, end_date) for t in tickers]
    
    result = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for ticker, df in executor.map(load_single_ticker, args_list):
            if df is not None and len(df) >= min_bars:
                result[ticker] = df
    
    return result


def get_db_path(root_dir: str, market: str, interval: str) -> str:
    """
    DB 경로 생성
    
    Args:
        root_dir: 프로젝트 루트 디렉토리
        market: 시장 (예: 'coin')
        interval: 간격 (예: 'minute60', 'day')
        
    Returns:
        DB 파일 경로
    """
    table_name = f'{market}_ohlcv_{interval}'
    return os.path.join(root_dir, f'var/data/{table_name}.db')


def get_table_name(market: str, interval: str) -> str:
    """
    테이블 이름 생성
    
    Args:
        market: 시장 (예: 'coin')
        interval: 간격 (예: 'minute60', 'day')
        
    Returns:
        테이블 이름
    """
    return f'{market}_ohlcv_{interval}'
