export PATH=/Users/yongbeom/opt/miniconda3/bin:/Users/yongbeom/opt/miniconda3/condabin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
source /Users/yongbeom/opt/miniconda3/etc/profile.d/conda.sh
cd /Users/yongbeom/cyb/project/2025/quant/sbin
conda activate
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1
conda activate stock
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

target_strategy=$1
target_interval=$2

market=coin
root_dir=$(cd .. && pwd)

if [ -z "$target_strategy" ]; then
    target_strategy=low_bb_du_2
fi

if [ -z "$target_interval" ]; then
    target_interval=minute60
fi


mkdir -p ${root_dir}/var/log/strategy_xgb_timeseries_backtest


python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-day-low_bb_du-label0-0.52220565-f7f3f8f35f30f5f4f14f34f22f18f28f11f20 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.day.label0.txt 2>&1 &
python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-day-low_bb_du-label1-0.5140001-f41f6f31f18f4f8f9f20f33f13f5f42f7f27 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.day.label1.txt 2>&1 &
python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-day-low_bb_du-label2-0.51423204-f18f37f3f12f33f21f14f35f8f9f2f6f43f36 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.day.label2.txt 2>&1 &

python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-minute240-low_bb_du-label0-0.92753154-f9f12f32f33f15f22f2f24f28 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.minute240.label0.txt 2>&1 &
python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-minute240-low_bb_du-label1-0.8006095-f19f43f14f32f29f3f40f41f10f6f16f37f12f15 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.minute240.label1.txt 2>&1 &
python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-minute240-low_bb_du-label2-0.78588766-f9f32f16f20f40f4f14f33f11f0f38f43f29f10f41f8f34f19f25f12f37f13f36f21f42f15f6f28f24f17f3f23f7f31 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.minute240.label2.txt 2>&1 &

python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-minute60-low_bb_du-label0-0.5630118-f33f32f19f18f22f16f35f7f42 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.minute60.label0.txt 2>&1 &
python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-minute60-low_bb_du-label1-0.5548528-f18f32f22f19f16f21f24f2f39f25f35f40f7f37f0f36f42f34f29f9f31f3f20f17 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.minute60.label1.txt 2>&1 &
python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-minute60-low_bb_du-label2-0.5172057-f8f21f6f9f42f31f22f1f34f14f10f4f11f13f40f38f37f23f5f20f3f29f17f41f30f24f26f18f28f27f12f33f25f0f36f7f19f39f16f32f43f15f2f35 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.minute60.label2.txt 2>&1 &
wait