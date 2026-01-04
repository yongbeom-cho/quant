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

python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-day-low_bb_du-label0-0.56676245-f3f8f31f14f22f34f40f20f36f7f24f38f5f23 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.day.label0.txt

# python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-day-low_bb_du-label0-0.56676245-f3f8f31f14f22f34f40f20f36f7f24f38f5f23 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.day.label0.txt 2>&1 &
# python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-day-low_bb_du-label1-0.5166048-f4f14f35f11f9f26f18f43f24f13f5f21f31f27 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.day.label1.txt 2>&1 &
# python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-day-low_bb_du-label2-0.5364216-f10f22f15f29f20f0f34f25f35f36f1f14f26f33 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.day.label2.txt 2>&1 &
# python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-minute240-low_bb_du-label0-0.5151893-f9f0f13f42f18f10f11f26f7f38f33f35f31f19 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.minute240.label0.txt 2>&1 &
# python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-minute240-low_bb_du-label1-0.5104595-f11f22f13f4f39f23f16f0f38f37f12f36f14f31 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.minute240.label1.txt 2>&1 &
# python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py --root_dir ${root_dir} --model_name xgb-coin-minute240-low_bb_du-label2-0.5040281-f23f12f5f28f35f24f42f1f30f36f39f3f29f26f31f4f32f21f16 > ${root_dir}/var/log/strategy_xgb_timeseries_backtest/low_bb_du.minute240.label2.txt 2>&1 &
# wait