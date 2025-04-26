# Here, we will implement the backtesting engine logic, wrapping the code in reusable functions.

import pandas as pd
import numpy as np

def generate_signals(z_scores, entry_band=1.5, exit_band=0.05):
    """
    Generate long/short/exit signals based on z-scores and thresholds.

    Args:
        z_scores (pd.Series): Z-score time series.
        entry_band (float): Entry threshold.
        exit_band (float): Exit threshold.

    Returns:
        pd.DataFrame: Signal DataFrame.
    """
    signals = pd.DataFrame(index=z_scores.index)
    signals["Z-Score"] = z_scores
    signals["Long"] = (z_scores < -entry_band).astype(int)
    signals["Short"] = (z_scores > entry_band).astype(int)
    signals["Exit"] = (z_scores.abs() < exit_band).astype(int)

    signals["curr_position"] = 0
    in_position = 0

    for i in range(1, len(signals)):
        if in_position == 0:
            if signals.iloc[i]["Long"]:
                in_position = 1
            elif signals.iloc[i]["Short"]:
                in_position = -1
        elif in_position != 0 and signals.iloc[i]["Exit"]:
            in_position = 0
        signals.iat[i, signals.columns.get_loc("curr_position")] = in_position

    return signals

def simulate_backtest(returns_df, hedge_ratios, signals, parameters):
    """
    Simulate the backtest using entry/exit signals, returns, and hedge ratios.

    Args:
        returns_df (pd.DataFrame): DataFrame containing returns of the cointegrated pair of assets. (from data_utils.get_returns())
        hedge_ratios (pd.Series): Time-varying hedge ratio. (from pairs_analysis.calculate_hedge_ratio())
        signals (pd.DataFrame): Signal DataFrame with curr_position. (either generate_signals() or generate_dynamic_signals())
        parameters (dict): Dictionary with keys: (from Streamlit UI)
            - 'max_hold_days'
            - 'take_profit'
            - 'stop_loss'
            - 'cooldown_days'
            - 'slippage' (bps)
            - 'transaction_cost' (bps)

    Returns:
        pd.DataFrame: Backtest results with strategy returns and cumulative returns.
    """
    # Infer tickers from returns_df columns
    tickers = list(returns_df.columns)
    if len(tickers) != 2:
        raise ValueError("Expected exactly two return columns for a pair.")

    asset1, asset2 = tickers

    merged = returns_df.copy()
    merged['hedge_ratio'] = hedge_ratios
    merged['curr_position'] = signals['curr_position']
    merged.dropna(inplace=True)

    merged[f'{asset1}_leg'] = 0.0
    merged[f'{asset2}_leg'] = 0.0
    merged['strategy_ret'] = 0.0
    merged['cumulative_returns'] = 1.0
    merged['executed_position'] = 0

    max_hold = parameters['max_hold_days']
    tp = parameters['take_profit']
    sl = parameters['stop_loss']
    cooldown_days = parameters['cooldown_days']
    slippage_bps = parameters.get('slippage', 0) / 10000
    tx_cost_bps = parameters.get('transaction_cost', 0) / 10000

    in_trade = False
    entry_cumret = 1.0
    days_held = 0
    position = 0
    cooldown_counter = 0

    for i in range(len(merged)):
        row = merged.iloc[i]
        date = merged.index[i]
        pos_signal = row['curr_position']

        if not in_trade and pos_signal != 0 and cooldown_counter == 0:
            in_trade = True
            entry_cumret = 1.0
            days_held = 0
            position = pos_signal
            tx_cost = 2 * tx_cost_bps
        else:
            tx_cost = 0.0

        if in_trade:
            leg1 = position * row[asset1]
            leg2 = -position * row['hedge_ratio'] * row[asset2]
            strat_ret = leg1 + leg2 - slippage_bps - tx_cost
            entry_cumret *= (1 + strat_ret)
            days_held += 1
        else:
            leg1 = leg2 = strat_ret = 0.0

        merged.at[date, f'{asset1}_leg'] = leg1
        merged.at[date, f'{asset2}_leg'] = leg2
        merged.at[date, 'strategy_ret'] = strat_ret
        merged.at[date, 'executed_position'] = position if in_trade else 0
        if i > 0:
            merged.at[date, 'cumulative_returns'] = merged.iloc[i - 1]['cumulative_returns'] * (1 + strat_ret)

        if in_trade and (
            pos_signal == 0 or
            days_held >= max_hold or
            entry_cumret - 1 >= tp or
            entry_cumret - 1 <= sl
        ):
            in_trade = False
            position = 0
            cooldown_counter = cooldown_days + 1

        if cooldown_counter > 0:
            cooldown_counter -= 1

    return merged

def simulate_portfolio_backtest(
    pairs_dict,
    parameters,
    top_n_pairs=3,
    capital=1.0,
    transaction_cost=0.001,
    slippage=0.001
):
    """
    Simulates a portfolio-level backtest over multiple cointegrated pairs.
    Capital is dynamically allocated to the top N most divergent (by z-score) pairs each day.

    Args:
        pairs_dict (dict): Dictionary containing {pair_name: {z_scores, returns_df, hedge_ratios}}.
        parameters (dict): Strategy parameters: max_hold_days, stop_loss, take_profit, cooldown_days.
        top_n_pairs (int): Max number of concurrent positions.
        capital (float): Starting portfolio capital.
        transaction_cost (float): Proportional cost per leg per trade.
        slippage (float): Slippage applied to both entry and exit returns.

    Returns:
        pd.DataFrame: Portfolio-level DataFrame with strategy performance.
    """
    # Initialize per-pair tracking structures
    pair_states = {}
    pair_results = {}

    # Infer common index (assumes all series are aligned on dates)
    common_dates = next(iter(pairs_dict.values()))['z_scores'].index

    # Initialize portfolio DataFrame
    portfolio_df = pd.DataFrame(index=common_dates)
    portfolio_df['portfolio_ret'] = 0.0
    portfolio_df['capital'] = capital
    portfolio_df['n_active_positions'] = 0

    for pair_name, data in pairs_dict.items():
        pair_states[pair_name] = {
            'in_trade': False,
            'entry_index': None,
            'entry_cumret': 1.0,
            'days_held': 0,
            'position': 0,
            'cooldown_counter': 0,
            'executed_position': np.zeros(len(common_dates))
        }

        pair_results[pair_name] = pd.DataFrame(index=common_dates)
        pair_results[pair_name]['z_score'] = data['z_scores']
        pair_results[pair_name]['strategy_ret'] = 0.0

    # Daily portfolio loop
    for t, date in enumerate(common_dates):
        day_returns = []
        abs_zscore_pairs = []

        # Step 1: Score & rank all candidate trades
        for pair_name, data in pairs_dict.items():
            state = pair_states[pair_name]
            z = data['z_scores'].iloc[t]

            if not np.isnan(z) and state['cooldown_counter'] == 0:
                abs_zscore_pairs.append((pair_name, abs(z)))

        # Step 2: Sort by absolute z-score magnitude
        sorted_pairs = sorted(abs_zscore_pairs, key=lambda x: x[1], reverse=True)
        selected_pairs = [p[0] for p in sorted_pairs[:top_n_pairs]]

        # Step 3: Execute logic per pair
        for pair_name in pairs_dict.keys():
            state = pair_states[pair_name]
            ret_df = pairs_dict[pair_name]['returns_df']
            hedge = pairs_dict[pair_name]['hedge_ratios']

            tickers = list(ret_df.columns)
            if len(tickers) != 2:
                raise ValueError("Expected exactly two return columns for a pair.")

            asset1, asset2 = tickers

            if t == 0 or date not in ret_df.index:
                continue

            row = ret_df.loc[date]
            z = pairs_dict[pair_name]['z_scores'].loc[date]

            # Skip pairs that aren't active today
            if pair_name not in selected_pairs:
                continue

            # ENTRY: if not in trade
            if not state['in_trade'] and z > parameters['entry_threshold']:
                state['in_trade'] = True
                state['entry_index'] = t
                state['entry_cumret'] = 1.0
                state['days_held'] = 0
                state['position'] = -1
                state['cooldown_counter'] = 0
            elif not state['in_trade'] and z < -parameters['entry_threshold']:
                state['in_trade'] = True
                state['entry_index'] = t
                state['entry_cumret'] = 1.0
                state['days_held'] = 0
                state['position'] = 1
                state['cooldown_counter'] = 0

            # EXIT: strategy_ret based on current holdings
            leg1 = 0.0
            leg2 = 0.0
            strat_ret = 0.0

            if state['in_trade']:
                pos = state['position']
                leg1 = pos * row[asset1]
                leg2 = -pos * hedge.loc[date] * row[asset2]
                strat_ret = leg1 + leg2

                # Apply slippage and cost on both legs
                strat_ret -= (transaction_cost + slippage) * 2

                state['entry_cumret'] *= (1 + strat_ret)
                state['days_held'] += 1

                # Exit condition
                if (
                    abs(z) < parameters['exit_threshold'] or
                    state['days_held'] >= parameters['max_hold_days'] or
                    state['entry_cumret'] - 1 >= parameters['take_profit'] or
                    state['entry_cumret'] - 1 <= parameters['stop_loss']
                ):
                    state['in_trade'] = False
                    state['position'] = 0
                    state['entry_index'] = None
                    state['days_held'] = 0
                    state['cooldown_counter'] = parameters['cooldown_days']

            # Log result
            pair_results[pair_name].at[date, 'strategy_ret'] = strat_ret
            state['executed_position'][t] = state['position']

            if state['cooldown_counter'] > 0 and not state['in_trade']:
                state['cooldown_counter'] -= 1

            day_returns.append(strat_ret)

        # Step 4: Aggregate portfolio return
        if day_returns:
            avg_ret = np.mean(day_returns)
        else:
            avg_ret = 0.0

        portfolio_df.at[date, 'portfolio_ret'] = avg_ret
        portfolio_df.at[date, 'capital'] *= (1 + avg_ret)
        portfolio_df.at[date, 'n_active_positions'] = len(day_returns)

    portfolio_df['cumulative_returns'] = portfolio_df['capital'] / capital

    return portfolio_df, pair_results

def compute_performance_metrics(df):
    """
    Compute strategy-level performance metrics.

    Args:
        df (pd.DataFrame): Backtest result DataFrame.

    Returns:
        dict: Dictionary containing performance statistics.
    """
    strategy_returns = df['strategy_ret']
    cumulative_returns = df['cumulative_returns']

    total_return = cumulative_returns.iloc[-1] - 1
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    cagr = (cumulative_returns.iloc[-1]) ** (1 / years) - 1
    vol = strategy_returns.std() * np.sqrt(252)
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1
    max_dd = drawdown.min()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }

def summarize_trades(executed_positions: pd.Series, cumulative_returns: pd.Series) -> dict:
    """
    Summarize trade-level statistics.

    Args:
        executed_positions (pd.Series): Series indicating actual held position over time.
        cumulative_returns (pd.Series): Series of cumulative return values.

    Returns:
        dict: Dictionary with trade count, win rate, avg hold period.
    """
    trade_count = 0
    win_trades = 0
    holding_periods = []

    in_trade = False
    entry_value = 1.0
    days_held = 0

    for i in range(len(executed_positions)):
        pos = executed_positions.iloc[i]
        curr_ret = cumulative_returns.iloc[i]

        if not in_trade and pos != 0:
            in_trade = True
            entry_value = curr_ret
            days_held = 1
        elif in_trade and pos != 0:
            days_held += 1
        elif in_trade and pos == 0:
            in_trade = False
            exit_value = curr_ret
            trade_count += 1
            if exit_value > entry_value:
                win_trades += 1
            holding_periods.append(days_held)

    return {
        "Number of Trades": trade_count,
        "Win Rate": win_trades / trade_count if trade_count > 0 else np.nan,
        "Average Holding Period (days)": np.mean(holding_periods) if holding_periods else np.nan
    }