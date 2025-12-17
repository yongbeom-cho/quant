#python strategy_unit_backtest/02_strategy_unit_backtest.py --interval day --target_strategy low_bb_du > log_02_low_bb_du.exact_gc.day.txt 2>&1 &
#python strategy_unit_backtest/02_strategy_unit_backtest.py --interval day --target_strategy low_bb_du_2 > log_02_low_bb_du_2.exact_gc.day.txt 2>&1 &
#wait

python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute240 --target_strategy low_bb_du > log_02_low_bb_du.exact_gc.minute240.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute60 --target_strategy low_bb_du > log_02_low_bb_du.exact_gc.minute60.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute240 --target_strategy low_bb_du_2 > log_02_low_bb_du_2.exact_gc.minute240.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute60 --target_strategy low_bb_du_2 > log_02_low_bb_du_2.exact_gc.minute60.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute240 --target_strategy low_bb_du_3 > log_02_low_bb_du_3.exact_gc.minute240.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute60 --target_strategy low_bb_du_3 > log_02_low_bb_du_3.exact_gc.minute60.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute240 --target_strategy low_bb_du_4 > log_02_low_bb_du_4.exact_gc.minute240.txt 2>&1 &
python strategy_unit_backtest/02_strategy_unit_backtest.py --interval minute60 --target_strategy low_bb_du_4 > log_02_low_bb_du_4.exact_gc.minute60.txt 2>&1 &
wait
