# This file will be used to run all statistical tests and analytics on the chosen stock pairs.
# It will be useful to show the results of this file to the user before backtesting.

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def test_cointegration(ts1, ts2):
    """
    Perform Engle-Granger cointegration test on two price series.
    Returns the p-value and test statistic.
    """
    score, pval, _ = coint(ts1, ts2)
    return {"p_value": pval, "test_statistic": score}

def johansen_test(log_df, det_order=0, k_ar_diff=1):
    """
    Perform Johansen cointegration test on a log-transformed DataFrame.
    Returns trace statistics and critical values (90%, 95%, 99%).
    """
    johan = coint_johansen(log_df, det_order, k_ar_diff)
    return {
        "trace_stat": johan.lr1.tolist(),
        "crit_vals": johan.cvt.tolist()
    }

def calculate_hedge_ratio(ts1, ts2, method="rolling" or "ols", window=60):
    """
    Calculate the hedge ratio (beta) between two price series.
    method = 'rolling' for rolling hedge ratio (no intercept),
    method = 'ols' for static OLS regression with intercept.
    """
    if method == "ols":
        X = sm.add_constant(ts2)
        model = sm.OLS(ts1, X).fit()
        return model.params[1]
    elif method == "rolling":
        ratio = ts1.rolling(window).corr(ts2) * (
            ts1.rolling(window).std() / ts2.rolling(window).std()
        )
        return ratio
    else:
        raise ValueError("method must be 'ols' or 'rolling'")
    
def analyze_multiple_pairs(pair_data_dict, test_method='eg'):
    """
    Run bidirectional cointegration tests across multiple stock pairs and return summary stats.

    Args:
        pair_data_dict (dict): Dictionary with (ticker1, ticker2) as keys and cleaned price DataFrames as values.
        test_method (str): 'eg' for Engle-Granger, 'johansen' for Johansen test (only applicable for >2 series).

    Returns:
        pd.DataFrame: DataFrame summarizing p-values, cointegration status, and hedge ratios for both directions.
    """
    results = []

    for (ticker1, ticker2), df in pair_data_dict.items():
        ts1, ts2 = df[ticker1], df[ticker2]

        # Engle-Granger coint test: ts1 ~ ts2
        pval_1 = coint(ts1, ts2)[1]
        is_coint_1 = pval_1 < 0.05
        hedge_ratio_1 = calculate_hedge_ratio(ts1, ts2, method='ols')

        # Engle-Granger coint test: ts2 ~ ts1 (reverse)
        pval_2 = coint(ts2, ts1)[1]
        is_coint_2 = pval_2 < 0.05
        hedge_ratio_2 = calculate_hedge_ratio(ts2, ts1, method='ols')

        results.append({
            "Pair": f"{ticker1}/{ticker2}",
            "P-Value ({}~{})".format(ticker1, ticker2): f"{pval_1:.2e}",
            "Cointegrated ({}~{})".format(ticker1, ticker2): is_coint_1,
            "Hedge Ratio ({}~{})".format(ticker1, ticker2): f"{hedge_ratio_1:.4e}",
            "P-Value ({}~{})".format(ticker2, ticker1): f"{pval_2:.2e}",
            "Cointegrated ({}~{})".format(ticker2, ticker1): is_coint_2,
            "Hedge Ratio ({}~{})".format(ticker2, ticker1): f"{hedge_ratio_2:.4e}"
        })

    return pd.DataFrame(results)

def compute_spread(ts1, ts2, hedge_ratio):
    """
    Compute spread from ts1 and ts2 using a given hedge ratio (can be scalar or Series).
    """
    return ts1 - hedge_ratio * ts2

def compute_zscore(spread, lookback=60):
    """
    Compute z-score of spread over a rolling window (60 days by default).
    """
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    zscore = (spread - mean) / std
    return zscore.dropna()

def compute_dynamic_zscore(spread, lookback=60, volatility_scale=True):
    """
    Compute a dynamic z-score of the spread, optionally scaled by rolling volatility.

    Args:
        spread (pd.Series): Time series spread between asset pairs.
        lookback (int): Lookback window for rolling stats.
        volatility_scale (bool): Whether to scale z-score by volatility bands.

    Returns:
        pd.Series: Time series of z-scores.
    """
    rolling_mean = spread.rolling(window=lookback).mean()
    rolling_std = spread.rolling(window=lookback).std()

    if volatility_scale:
        dynamic_threshold = rolling_std.rolling(window=lookback).mean()
        zscore = (spread - rolling_mean) / dynamic_threshold
    else:
        zscore = (spread - rolling_mean) / rolling_std

    return zscore.dropna()