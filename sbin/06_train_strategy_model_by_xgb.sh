export PATH=/Users/yongbeom/opt/miniconda3/bin:/Users/yongbeom/opt/miniconda3/condabin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
source /Users/yongbeom/opt/miniconda3/etc/profile.d/conda.sh
cd /Users/yongbeom/cyb/project/2025/quant/sbin
conda activate
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1
conda activate stock
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

target_strategy_feature=$1
target_interval=$2
model_output_dir="var/xgb_model"
market=coin
root_dir=$(cd .. && pwd)

if [ -z "$target_strategy" ]; then
    target_strategy=low_bb_du
fi

if [ -z "$target_interval" ]; then
    target_interval=minute60
fi

# bash 05_get_strategy_filtered_data.sh

python train_xgb/06_train_strategy_model_by_xgb.py --root_dir ${root_dir} --market ${market} --target_strategy_feature low_bb_du --interval minute60 --model_output_dir ${model_output_dir} > log_06.bt.minute60.txt 2>&1 &
python train_xgb/06_train_strategy_model_by_xgb.py --root_dir ${root_dir} --market ${market} --target_strategy_feature low_bb_du --interval minute240 --model_output_dir ${model_output_dir} > log_06.bt.minute240.txt 2>&1 &
python train_xgb/06_train_strategy_model_by_xgb.py --root_dir ${root_dir} --market ${market} --target_strategy_feature low_bb_du --interval day --model_output_dir ${model_output_dir} > log_06.bt.day.txt 2>&1 &
wait