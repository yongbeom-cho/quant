#!/bin/bash
# 백테스트 실행 스크립트 (체크포인트 모드)
# 워커 4개로 메모리 절약, 100개마다 중간 저장

source ./env.sh

buy_strategy_name="pb_rebound"
buy_strategy_name="pb_inrange"
buy_config_idx=14
sell_config_idx=2
backtest_result_dir="${root_dir}/var/backtest_result"
parallel=6
mkdir -p ${backtest_result_dir}

# Run backtest for each interval
for interval in day minute240 minute60; do

    # TS (timeseries) backtest
    output_file="${backtest_result_dir}/ts_backtest_${interval}_${buy_config_idx}_${sell_config_idx}.csv"
    
    echo "=== Running TS ${interval} Buy ${buy_config_idx}, Sell ${sell_config_idx} Config Combinations ==="
    python -u backtest/backtest_runner.py \
        --buy_config buy_strategy/config/buy_config.json --buy_config_idx ${buy_config_idx} \
        --sell_config sell_strategy/config/sell_config.json --sell_config_idx ${sell_config_idx} \
        --root_dir ${root_dir} \
        --market coin --interval ${interval} \
        --parallel --workers ${parallel} \
        --checkpoint_interval ${parallel} \
        --sort_by total_pnl --top_n 20 \
        --is_timeseries_backtest \
        --max_position_cnts 100 \
        --output ${output_file} 2>&1
    echo "=== Done ==="
    echo ""
    
    # Unit backtest
    output_file="${backtest_result_dir}/unit_backtest_${interval}_${buy_config_idx}_${sell_config_idx}.csv"
    
    echo "=== Running ${interval} Buy ${buy_config_idx}, Sell ${sell_config_idx} Config Combinations ==="
    python -u backtest/backtest_runner.py \
        --buy_config buy_strategy/config/buy_config.json --buy_config_idx ${buy_config_idx} \
        --sell_config sell_strategy/config/sell_config.json --sell_config_idx ${sell_config_idx} \
        --root_dir ${root_dir} \
        --market coin --interval ${interval} \
        --parallel --workers ${parallel} \
        --checkpoint_interval ${parallel} \
        --sort_by total_pnl --top_n 20 \
        --max_position_cnts 3 \
        --output ${output_file} 2>&1
    echo "=== Done ==="
    echo ""
done

# Configuration: interval별 max_position_cnts 설정 함수
get_max_position_cnts() {
    case "$1" in
        day)
            echo "3,5,10"
            ;;
        minute240)
            echo "3,5"
            ;;
        minute60)
            echo "10,20"
            ;;
        *)
            echo "5,10,20"  # 기본값
            ;;
    esac
}

xgb_buy_config_idx=8  # XGB 모델을 사용하는 buy config index

XGB 모델 학습 및 TS 백테스트 실행
echo "=== Starting XGB Model Training and TS Backtest ==="
for interval in day minute240 minute60; do
    max_position_cnts=$(get_max_position_cnts ${interval})
    
    # 1. backtest 결과 파일 경로
    backtest_csv="${backtest_result_dir}/ts_backtest_${interval}_${buy_config_idx}_${sell_config_idx}.csv"
    
    # 2. XGB 모델 학습
    model_output_dir="var/${buy_strategy_name}_xgb_model"
    xgb_data_dir="${root_dir}/var/${buy_strategy_name}_xgb_data"
    
    echo "=== Training XGB models for ${interval} ==="
    python -u train_xgb/train_strategy_model.py \
        --root_dir ${root_dir} \
        --market coin \
        --interval ${interval} \
        --backtest_csv ${backtest_csv} \
        --strategy_name ${buy_strategy_name} \
        --output_data_dir ${xgb_data_dir} \
        --model_output_dir ${model_output_dir} 2>&1
    echo "=== XGB Training Done for ${interval} ==="
    echo ""
    
done

for interval in day minute240 minute60; do
    # 3. 학습된 모델을 사용하여 TS 백테스트 실행
    ts_backtest_output="${backtest_result_dir}/ts_backtest_${interval}_xgb_${buy_strategy_name}.csv"
    
    echo "=== Running TS ${interval} XGB Backtest ==="
    python -u backtest/backtest_runner.py \
        --buy_config buy_strategy/config/buy_config.json --buy_config_idx ${xgb_buy_config_idx} \
        --sell_config sell_strategy/config/sell_config.json --sell_config_idx ${sell_config_idx} \
        --root_dir ${root_dir} \
        --market coin --interval ${interval} \
        --parallel --workers ${parallel} \
        --checkpoint_interval ${parallel} \
        --sort_by total_pnl --top_n 20 \
        --is_timeseries_backtest \
        --max_position_cnts ${max_position_cnts} \
        --output ${ts_backtest_output} 2>&1
    echo "=== TS XGB Backtest Done for ${interval} ==="
    echo ""

done

echo "=== All XGB Training and TS Backtest Completed ==="