"""
Backtest Result Analyzer

backtest 결과 CSV를 분석하여 최적의 buy/sell 전략을 선택합니다.
"""
import pandas as pd
import ast
import itertools
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter


def parse_buy_params(buy_params_str):
    """buy_params 문자열을 딕셔너리로 변환"""
    try:
        if isinstance(buy_params_str, str):
            return ast.literal_eval(buy_params_str)
        return buy_params_str
    except:
        return {}


def select_best_strategies_from_backtest(
    csv_path: str,
    interval: str,
    top_n: int = 3
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    backtest 결과 CSV에서 최적 전략 선택
    
    로직:
    1. PnL > 0인 것만 필터링
    2. 조합 생성 후 전체 top 10 (PnL * mdd) 안에 있는지 확인
    3. buy_params의 각 파라미터별로 PnL 평균 계산
    4. 각 파라미터별로 top n개 값 선택
    5. 없으면 top n 증가 후 반복
    6. 최종적으로 PnL * mdd가 가장 높은 것을 선택
    
    Args:
        csv_path: backtest 결과 CSV 파일 경로
        interval: 시간 간격 (day, minute60, minute240)
        top_n: 각 파라미터별로 선택할 top N개 (기본 3)
    
    Returns:
        (best_buy_strategy, best_sell_strategies)
        - best_buy_strategy: {'strategy_name': str, 'params': dict}
        - best_sell_strategies: [{'strategy_name': str, 'params': dict}, ...]
    """
    df = pd.read_csv(csv_path)
    
    # 1. PnL > 0이고 mdd < 50% 인 것만 필터링
    df_positive = df[(df['total_pnl'] > 0) & (df['max_drawdown_pct'] < 50)].copy()
    
    if len(df_positive) == 0:
        # 조건에 맞는 것이 없으면 전체에서 선택
        df_positive = df.copy()
    
    # 2. 전체 Top 10 (PnL * mdd) 먼저 계산
    df_positive['score'] = df_positive['total_pnl'] * (1.0 - df_positive['max_drawdown_pct'] / 100.0)
    top10_df = df_positive.nlargest(10, 'score')
    
    # Print top10 rows
    print("=" * 80)
    print("Top 10 Rows (by score = total_pnl * (1 - max_drawdown_pct/100)):")
    print("=" * 80)
    for idx, row in top10_df.iterrows():
        print(f"\nRow Index: {idx}")
        print(f"  total_pnl: {row['total_pnl']}")
        print(f"  max_drawdown_pct: {row['max_drawdown_pct']}")
        print(f"  score: {row['score']}")
        print(f"  buy_strategy_name: {row['buy_strategy_name']}")
        print(f"  buy_params: {row['buy_params']}")
        print(f"  sell_strategy_name: {row['sell_strategy_name']}")
        print(f"  sell_params: {row['sell_params']}")
    print("=" * 80)
    print()
    
    top10_score = top10_df['score'].min()
    top10_indices = set(top10_df.index)
    
    # Top 10의 buy_params를 미리 파싱해서 저장 (비교용)
    top10_params_set = set()
    top10_params_to_row = {}  # {frozenset(params): (idx, row, score)}
    for idx, row in top10_df.iterrows():
        buy_params = parse_buy_params(row['buy_params'])
        # strategy_name, max_investment_ratio 제외한 실제 파라미터만
        clean_params = {
            k: v for k, v in buy_params.items() 
            if k not in ['strategy_name', 'max_investment_ratio']
        }
        params_key = frozenset(clean_params.items())
        top10_params_set.add(params_key)
        score = row['score']
        top10_params_to_row[params_key] = (idx, row, score)
    
    # 3. buy_params 파싱 및 파라미터별 PnL 평균 계산
    param_avg_pnl = defaultdict(lambda: defaultdict(list))  # {param_name: {param_value: [pnl_values]}}
    
    for _, row in df_positive.iterrows():
        buy_params = parse_buy_params(row['buy_params'])
        pnl = row['total_pnl']
        
        # strategy_name, max_investment_ratio 등은 제외하고 실제 파라미터만
        for param_name, param_value in buy_params.items():
            if param_name not in ['strategy_name', 'max_investment_ratio']:
                param_avg_pnl[param_name][param_value].append(pnl)
    
    # 4. 각 파라미터별로 평균 PnL 계산 및 top n개 선택 (반복)
    current_top_n = top_n
    max_iterations = 5  # 최대 반복 횟수
    best_combination = None
    best_score = -float('inf')
    valid_combinations = []  # 반복문 밖에서 초기화
    
    for iteration in range(max_iterations):
        # 각 파라미터별 top n개 값 선택
        param_top_values = {}
        for param_name, value_pnl_dict in param_avg_pnl.items():
            # 평균 PnL 계산
            avg_pnl_dict = {
                value: sum(pnl_list) / len(pnl_list) 
                for value, pnl_list in value_pnl_dict.items()
            }
            # top n개 선택
            sorted_values = sorted(avg_pnl_dict.items(), key=lambda x: x[1], reverse=True)
            param_top_values[param_name] = [val for val, _ in sorted_values[:current_top_n]]
        
        # 조합 생성
        param_names = list(param_top_values.keys())
        param_value_lists = [param_top_values[name] for name in param_names]
        combinations = list(itertools.product(*param_value_lists))
        
        # 6. 각 조합에 해당하는 행 찾기 및 top10과 비교
        combination_scores = []
        for combo in combinations:
            # 조합에 해당하는 파라미터 딕셔너리 생성
            combo_params = dict(zip(param_names, combo))
            combo_params_key = frozenset(combo_params.items())
            
            # top10에 있는지 바로 확인
            if combo_params_key in top10_params_set:
                # top10에 있으면 해당 row와 score 사용
                idx, row, score = top10_params_to_row[combo_params_key]
                pnl = row['total_pnl']
                mdd_ratio = 1.0 - (row['max_drawdown_pct'] / 100.0)
                combination_scores.append({
                    'params': combo_params,
                    'row': row,
                    'row_idx': idx,
                    'score': score,
                    'pnl': pnl,
                    'mdd_ratio': mdd_ratio,
                    'is_in_top10': True
                })
        
        # 7. 조합 중에서 top 10 안에 드는 것이 있는지 확인
        valid_combinations = [c for c in combination_scores if c['is_in_top10']]
        
        if valid_combinations:
            # 유효한 조합 중에서 최고 점수 선택
            best_combo = max(valid_combinations, key=lambda x: x['score'])
            best_combination = best_combo
            best_score = best_combo['score']
            break
        else:
            current_top_n += 1
    
    # 8. 최종 결과 생성
    if best_combination is None:
        # 조합을 찾지 못한 경우 None 반환
        return None, None
    
    # best_combination에서 buy strategy 추출
    best_row = best_combination['row']
    buy_strategy_name = best_row['buy_strategy_name']
    best_params = best_combination['params']
    
    # strategy_name 추가 (원본에서 가져오기)
    original_params = parse_buy_params(best_row['buy_params'])
    if 'strategy_name' in original_params:
        best_params['strategy_name'] = original_params['strategy_name']
    if 'max_investment_ratio' in original_params:
        best_params['max_investment_ratio'] = original_params['max_investment_ratio']
    
    best_buy_strategy = {
        'strategy_name': buy_strategy_name,
        'params': best_params
    }
    
    # 9. sell strategies 추출 (같은 buy_params를 가진 것들 중에서)
    # valid_combinations에서 같은 buy_params를 가진 모든 row 추출
    sell_strategies = []
    seen_sell_params = set()
    
    # valid_combinations의 모든 조합이 같은 buy_params를 공유하므로 여기서 추출
    matching_rows = [combo['row'] for combo in valid_combinations]
    
    # PnL * mdd 기준으로 정렬
    matching_df = pd.DataFrame(matching_rows)
    if len(matching_df) > 0:
        matching_df['score'] = matching_df['total_pnl'] * (1.0 - matching_df['max_drawdown_pct'] / 100.0)
        matching_df = matching_df.sort_values('score', ascending=False)
        
        # 최대 3개까지만 반환
        for _, row in matching_df.head(3).iterrows():
            sell_strategy_name = row['sell_strategy_name']
            sell_params = parse_buy_params(row['sell_params'])
            params_key = str(sorted(sell_params.items()))
            
            if params_key not in seen_sell_params:
                seen_sell_params.add(params_key)
                sell_strategies.append({
                    'strategy_name': sell_strategy_name,
                    'params': sell_params
                })
                # 최대 3개까지만
                if len(sell_strategies) >= 3:
                    break
    
    return best_buy_strategy, sell_strategies


def get_strategy_config_from_backtest(
    csv_path: str,
    interval: str,
    top_n: int = 3
) -> Dict[str, Any]:
    """
    backtest 결과에서 전략 설정 구성
    
    Returns:
        {
            'buy_strategy': {'strategy_name': str, 'params': dict} or None,
            'sell_strategies': [{'strategy_name': str, 'params': dict}, ...] or None,
            'label_count': int
        }
    """
    best_buy, sell_strategies = select_best_strategies_from_backtest(csv_path, interval, top_n)
    
    if best_buy is None or sell_strategies is None:
        return {
            'buy_strategy': None,
            'sell_strategies': None,
            'label_count': 0
        }
    
    return {
        'buy_strategy': best_buy,
        'sell_strategies': sell_strategies,
        'label_count': len(sell_strategies)
    }
