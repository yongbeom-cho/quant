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

if [ -z "$target_strategy_feature" ]; then
    target_strategy_feature=low_bb_du
fi

if [ -z "$output_dir" ]; then
    output_dir=${root_dir}/var/xgb_${target_strategy_feature}_data
fi

# test_log_dir=${root_dir}/var/log/test_strategy_feature
# mkdir ${test_log_dir}
# parallel=7
# for part in {0..6}; do
#     python train_xgb/test_strategy_feature.py --root_dir ${root_dir} --market ${market} --interval day --parallel ${parallel} --part ${part} > ${test_log_dir}/day_${part}.log &
# done
# wait

# for part in {0..6}; do
#     python train_xgb/test_strategy_feature.py --root_dir ${root_dir} --market ${market} --interval minute240 --parallel ${parallel} --part ${part} > ${test_log_dir}/minute240_${part}.log &
# done
# wait

# for part in {0..6}; do
#     python train_xgb/test_strategy_feature.py --root_dir ${root_dir} --market ${market} --interval minute60 --parallel ${parallel} --part ${part} > ${test_log_dir}/minute60_${part}.log &
# done
# wait

mkdir -p ${output_dir}

# target_strategy_feature=pb_du
# output_dir=${root_dir}/var/xgb_${target_strategy_feature}_data
# python train_xgb/05_get_strategy_filtered_data.py --root_dir ${root_dir} --market ${market} --target_strategy_feature ${target_strategy_feature} --output_dir ${output_dir} --interval day

python train_xgb/05_get_strategy_filtered_data.py --root_dir ${root_dir} --market ${market} --target_strategy_feature ${target_strategy_feature} --output_dir ${output_dir} --interval minute60 &
python train_xgb/05_get_strategy_filtered_data.py --root_dir ${root_dir} --market ${market} --target_strategy_feature ${target_strategy_feature} --output_dir ${output_dir} --interval day &
python train_xgb/05_get_strategy_filtered_data.py --root_dir ${root_dir} --market ${market} --target_strategy_feature ${target_strategy_feature} --output_dir ${output_dir} --interval minute240 &
wait