01_get_all_ohlcv_data.sh
02_strategy_unit_backtest.sh -> 51% 이상인 condition 찾아내기.
03_strategy_view_signals.sh -> 해당 condition이 ticker별로 잘 맞는지 확인하기 (어떤 특성에서 잘 안맞는지 확인해서 condition을 변경하기)

02 - 03 cycling(반복 test) -> 02, 03이 편해야하는데, condition idea는 trading view로 확인하고(03), 02로 테스트 해봐도 됨.
TODO: 03에서 필요한 기능들, 보조지표 추가할수 있는 기능 이런거 필요할듯. (trading view 처럼)

추출 condition 만족하는 signal의 수가 너무 작으면 그대로 04_strategy_timeseries_backtest 테스트
그렇지 않고 꽤 많은경우에는 05, 06, 07, 08 진행
