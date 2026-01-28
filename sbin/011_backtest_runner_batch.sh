#!/bin/bash
# 백테스트 실행 스크립트 (체크포인트 모드)
# 워커 4개로 메모리 절약, 100개마다 중간 저장

. /mnt/c/Users/USER/Projects/quant/quant-venv/bin/activate

echo "=== Running All Buy/Sell Config Combinations ==="
python -u backtest/backtest_runner.py \
    --buy_config buy_strategy/config/buy_config.json --buy_config_idx all \
    --sell_config sell_strategy/config/sell_config.json --sell_config_idx all \
    --root_dir /mnt/c/Users/USER/Projects/quant \
    --market coin --interval day \
    --ticker KRW-BTC \
    --parallel --workers 4 \
    --checkpoint_interval 100 \
    --sort_by total_pnl --top_n 20 \
    --output /tmp/backtest_results.csv 2>&1

echo "=== Done ==="
