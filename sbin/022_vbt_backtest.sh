#!/bin/bash

# 가상환경 활성화
source /mnt/c/Users/USER/Projects/quant/quant-venv/bin/activate

# 사용법: ./022_vbt_backtest.sh [--root_dir 경로] [--market 마켓] [--interval 인터벌] [--processes 개수]
# ex: 022_vbt_backtest.sh --root_dir /mnt/c/Users/USER/Projects/quant --market coin --interval minute60 --processes 16

# 인자 파싱
root_dir=""
market="coin"
interval="day"
processes=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --root_dir)
      root_dir="$2"
      shift 2
      ;;
    --market)
      market="$2"
      shift 2
      ;;
    --interval)
      interval="$2"
      shift 2
      ;;
    --processes)
      processes="$2"
      shift 2
      ;;
    *)
      if [ -z "$root_dir" ] && [[ "$1" != --* ]]; then root_dir="$1"; fi
      shift
      ;;
  esac
done

# 루트 디렉토리 기본값 설정
if [ -z "$root_dir" ]; then
    root_dir=$(cd .. && pwd)
fi

echo "VBT 고도화 백테스트를 시작합니다... (Market: $market, Interval: $interval, Processes: ${processes:-default})"

# 백테스트 실행 및 로그 저장
cmd="python ${root_dir}/sbin/strategy_unit_backtest/022_vbt_backtest.py --root_dir ${root_dir} --market ${market} --interval ${interval}"
if [ ! -z "$processes" ]; then
    cmd="$cmd --processes $processes"
fi

$cmd > "${root_dir}/sbin/log_022_vbt_test_${market}_${interval}.txt" 2>&1

echo "백테스트가 완료되었습니다. 로그 파일: ${root_dir}/sbin/log_022_vbt_test_${market}_${interval}.txt"
