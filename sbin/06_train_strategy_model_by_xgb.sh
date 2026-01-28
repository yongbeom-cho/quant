export PATH=/Users/yongbeom/opt/miniconda3/bin:/Users/yongbeom/opt/miniconda3/condabin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
source /Users/yongbeom/opt/miniconda3/etc/profile.d/conda.sh
cd /Users/yongbeom/cyb/project/2025/quant/sbin
conda activate
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1
conda activate stock
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

target_strategy=$1
xgb_data_dir=$2
model_output_dir=$3
log_dir=$4
market=coin
root_dir=$(cd .. && pwd)

if [ -z "$target_strategy" ]; then
    target_strategy=low_bb_du
fi
if [ -z "$xgb_data_dir" ]; then
    xgb_data_dir=${root_dir}/var/xgb_${target_strategy}_data
fi
if [ -z "$model_output_dir" ]; then
    model_output_dir=${root_dir}/var/xgb_${target_strategy}_model
fi
if [ -z "$log_dir" ]; then
    log_dir=${root_dir}/var/log/train_${target_strategy}_xgb
fi

# bash 05_get_strategy_filtered_data.sh ${target_strategy} ${xgb_data_dir}

mkdir -p ${model_output_dir}
mkdir -p ${log_dir}

python train_xgb/06_train_strategy_model_by_xgb.py --root_dir ${root_dir} --input_data_dir ${xgb_data_dir} --market ${market} --target_strategy_feature ${target_strategy} --interval minute60 --model_output_dir ${model_output_dir} > ${log_dir}/log_06.bt.minute60.txt 2>&1 &
python train_xgb/06_train_strategy_model_by_xgb.py --root_dir ${root_dir} --input_data_dir ${xgb_data_dir} --market ${market} --target_strategy_feature ${target_strategy} --interval minute240 --model_output_dir ${model_output_dir} > ${log_dir}/log_06.bt.minute240.txt 2>&1 &
python train_xgb/06_train_strategy_model_by_xgb.py --root_dir ${root_dir} --input_data_dir ${xgb_data_dir} --market ${market} --target_strategy_feature ${target_strategy} --interval day --model_output_dir ${model_output_dir} > ${log_dir}/log_06.bt.day.txt 2>&1 &
wait