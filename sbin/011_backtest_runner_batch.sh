#!/bin/bash
# 백테스트 실행 스크립트 (체크포인트 모드)
# 워커 4개로 메모리 절약, 100개마다 중간 저장

source ./env.sh


echo "=== Running TS Day Buy 1, Sell 2 Config Combinations ==="
python -u backtest/backtest_runner.py \
    --buy_config buy_strategy/config/buy_config.json --buy_config_idx 1 \
    --sell_config sell_strategy/config/sell_config.json --sell_config_idx 2 \
    --root_dir ${root_dir} \
    --market coin --interval day \
    --parallel --workers 8 \
    --checkpoint_interval 100 \
    --sort_by total_pnl --top_n 20 \
    --is_timeseries_backtest \
    --output ${root_dir}/var/ts_backtest_results_day_1_2.csv 2>&1
echo "=== Done ==="

echo "=== Running Day Buy 1, Sell 2 Config Combinations ==="
python -u backtest/backtest_runner.py \
    --buy_config buy_strategy/config/buy_config.json --buy_config_idx 1 \
    --sell_config sell_strategy/config/sell_config.json --sell_config_idx 2 \
    --root_dir ${root_dir} \
    --market coin --interval day \
    --parallel --workers 8 \
    --checkpoint_interval 100 \
    --sort_by total_pnl --top_n 20 \
    --output ${root_dir}/var/unit_backtest_day_1_2.csv 2>&1
echo "=== Done ==="


echo "=== Running TS minute240 Buy 1, Sell 2 Config Combinations ==="
python -u backtest/backtest_runner.py \
    --buy_config buy_strategy/config/buy_config.json --buy_config_idx 1 \
    --sell_config sell_strategy/config/sell_config.json --sell_config_idx 2 \
    --root_dir ${root_dir} \
    --market coin --interval minute240 \
    --parallel --workers 8 \
    --checkpoint_interval 100 \
    --sort_by total_pnl --top_n 20 \
    --is_timeseries_backtest \
    --output ${root_dir}/var/ts_backtest_results_minute240_1_2.csv 2>&1
echo "=== Done ==="


echo "=== Running minute240 Buy 1, Sell 2 Config Combinations ==="
python -u backtest/backtest_runner.py \
    --buy_config buy_strategy/config/buy_config.json --buy_config_idx 1 \
    --sell_config sell_strategy/config/sell_config.json --sell_config_idx 2 \
    --root_dir ${root_dir} \
    --market coin --interval minute240 \
    --parallel --workers 8 \
    --checkpoint_interval 100 \
    --sort_by total_pnl --top_n 20 \
    --output ${root_dir}/var/unit_backtest_results_minute240_1_2.csv 2>&1
echo "=== Done ==="


echo "=== Running TS minute60 Buy 1, Sell 2 Config Combinations ==="
python -u backtest/backtest_runner.py \
    --buy_config buy_strategy/config/buy_config.json --buy_config_idx 1 \
    --sell_config sell_strategy/config/sell_config.json --sell_config_idx 2 \
    --root_dir ${root_dir} \
    --market coin --interval minute60 \
    --parallel --workers 8 \
    --checkpoint_interval 100 \
    --sort_by total_pnl --top_n 20 \
    --is_timeseries_backtest \
    --output ${root_dir}/var/ts_backtest_results_minute60_1_2.csv 2>&1
echo "=== Done ==="

echo "=== Running minute60 Buy 1, Sell 2 Config Combinations ==="
python -u backtest/backtest_runner.py \
    --buy_config buy_strategy/config/buy_config.json --buy_config_idx 1 \
    --sell_config sell_strategy/config/sell_config.json --sell_config_idx 2 \
    --root_dir ${root_dir} \
    --market coin --interval minute60 \
    --parallel --workers 8 \
    --checkpoint_interval 100 \
    --sort_by total_pnl --top_n 20 \
    --output ${root_dir}/var/unit_backtest_results_minute60_1_2.csv 2>&1
echo "=== Done ==="


