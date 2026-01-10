export PATH=/Users/yongbeom/opt/miniconda3/bin:/Users/yongbeom/opt/miniconda3/condabin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
source /Users/yongbeom/opt/miniconda3/etc/profile.d/conda.sh
cd /Users/yongbeom/cyb/project/2025/quant/sbin
conda activate
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1
conda activate stock
conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1


market=coin
strategy_name=low_bb_du
root_dir=$(cd .. && pwd)

xgb_data_dir=${root_dir}/var/xgb_data.v3.0
train_log_dir=${root_dir}/var/log/train_xgb.v3.0
log_dir=${root_dir}/var/log/strategy_xgb_timeseries_backtest.v3.0
model_dir=${root_dir}/var/xgb_model.v3.0

mkdir -p ${log_dir}

# bash 06_train_strategy_model_by_xgb.sh ${strategy_name} ${xgb_data_dir} ${model_dir} ${train_log_dir}

MAX_JOBS=8
job_cnt=0

for model in ${model_dir}/xgb-${market}-day-${strategy_name}* \
             ${model_dir}/xgb-${market}-minute240-${strategy_name}* \
             ${model_dir}/xgb-${market}-minute60-${strategy_name}*; do

    model_name="$(basename "$model")"
    tmp="${model_name#*-*-}"
    log_fname="${tmp%-*}"

    python strategy_timeseries_backtest/07_strategy_timeseries_backtest.py \
        --root_dir ${root_dir} \
        --model_name ${model_name} \
        --model_dir ${model_dir} \
        > ${log_dir}/${log_fname}.txt 2>&1 &

    ((job_cnt++))

    if (( job_cnt >= MAX_JOBS )); then
        wait
        job_cnt=0
    fi
done
wait


best_model_name_dir=${root_dir}/var/best_model_name
mkdir -p ${best_model_name_dir}

python strategy_timeseries_backtest/08_get_best_model_by_ts_backtest.py --log_dir ${log_dir} --interval day --strategy_name ${strategy_name} > ${best_model_name_dir}/${strategy_name}.day.txt
python strategy_timeseries_backtest/08_get_best_model_by_ts_backtest.py --log_dir ${log_dir} --interval minute240 --strategy_name ${strategy_name} > ${best_model_name_dir}/${strategy_name}.minute240.txt
python strategy_timeseries_backtest/08_get_best_model_by_ts_backtest.py --log_dir ${log_dir} --interval minute60 --strategy_name ${strategy_name} > ${best_model_name_dir}/${strategy_name}.minute60.txt
