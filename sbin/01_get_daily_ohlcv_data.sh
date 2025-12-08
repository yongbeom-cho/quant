#!/bin/bash

root_dir=$1
date_arg=$2 # "all" or YYYYMMDD

if [ -z "$root_dir" ]; then
    root_dir=$(cd .. && pwd)
fi

if [ -z "$date_arg" ]; then
    date_arg=$(date -v-1d +%Y%m%d)
fi

echo "$root_dir $date_arg"
##--interval : [day, minute1, minute3, minute5, minute10, minute15, minute30, minute60, minute240, week, month]
market=coin
output_dir="${root_dir}/var/data/ohlcv_${interval}"

for interval in day minute1 minute3 minute5 minute10 minute15 minute30 minute60 minute240 week month; do
    python ${root_dir}/sbin/data_pipeline/01_get_daily_ohlcv_data.py --date ${date} --market ${market} --interval ${interval} --output_dir ${root_dir}var/data/ohlcv_${x} 
done