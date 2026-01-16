export PATH=/Users/yongbeom/opt/miniconda3/bin:/Users/yongbeom/opt/miniconda3/condabin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
source /Users/yongbeom/opt/miniconda3/etc/profile.d/conda.sh
cd /Users/yongbeom/cyb/project/2025/quant/sbin
conda activate
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1
conda activate stock
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

target_strategy_feature=$1
output_dir=$2
market=coin
root_dir=$(cd .. && pwd)

if [ -z "$target_strategy" ]; then
    target_strategy=low_bb_du
fi

if [ -z "$output_dir" ]; then
    output_dir=${root_dir}/var/xgb_data
fi

mkdir -p ${output_dir}

python train_xgb/05_get_strategy_filtered_data.py --root_dir ${root_dir} --market ${market} --target_strategy_feature ${target_strategy_feature} --output_dir ${output_dir} --interval minute60 &
python train_xgb/05_get_strategy_filtered_data.py --root_dir ${root_dir} --market ${market} --target_strategy_feature ${target_strategy_feature} --output_dir ${output_dir} --interval day &
python train_xgb/05_get_strategy_filtered_data.py --root_dir ${root_dir} --market ${market} --target_strategy_feature ${target_strategy_feature} --output_dir ${output_dir} --interval minute240 &
wait