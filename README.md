data를 통한 backtesting 기반의 전략 발굴 및 자동매매 시스템

1. ohlcv data 추출 파이프라인
- coin 1m, 3m, 5m, 15m, 30m, 60m, 240m, 1d, 1w, 1m
- korea 1m, 3m, 5m, 15m, 30m, 60m, 240m, 1d, 1w, 1m
- overseas 1m, 3m, 5m, 15m, 30m, 60m, 240m, 1d, 1w, 1m

#매일 빈데이터 없이 데이터를 추출하여 xlsx에 저장 추후 mysql db table에 저장
sbin/data_pipeline/01_get_daily_ohlcv_data.py

위 script를 매일 돌리는 script
sbin/01_get_daily_ohlcv_data.sh

unit_backtest
sbin/02_strategy_unit_backtest.sh
- 추가해야할 기능
  1. 특정 coin만 unit test - 완료
  2. 특정 strategy만 unit test - 완료
  3. 특정 interval unit test - 완료
  4. buy_signal 있는 지점에서(왼쪽 180, 오른쪽 60개 캔들정도) chart view 기능
  5. sell_strategy
