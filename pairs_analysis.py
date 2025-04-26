# This file will be used to run all statistical tests and analytics on the chosen stock pairs.
# It will be useful to show the results of this file to the user before backtesting.

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

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
    results = {}

    for (ticker1, ticker2), df in pair_data_dict.items():
        ts1, ts2 = df[ticker1], df[ticker2]
        pval_1 = coint(ts1, ts2)[1]
        pval_2 = coint(ts2, ts1)[1]
        is_coint_1 = pval_1 < 0.05
        is_coint_2 = pval_2 < 0.05
        hedge_ratio_1 = calculate_hedge_ratio(ts1, ts2, method='ols')
        hedge_ratio_2 = calculate_hedge_ratio(ts2, ts1, method='ols')

        results[(ticker1, ticker2)] = pd.DataFrame({
            "Test Direction": [f"{ticker1} ~ {ticker2}", f"{ticker2} ~ {ticker1}"],
            "P-Value": [f"{pval_1:.2e}", f"{pval_2:.2e}"],
            "Cointegrated?": [is_coint_1, is_coint_2],
            "Hedge Ratio (OLS)": [f"{hedge_ratio_1:.4e}", f"{hedge_ratio_2:.4e}"]
        })

    return results

def compute_spread(ts1, ts2, hedge_ratio):
    """
    Compute spread from ts1 and ts2 using a given hedge ratio (can be scalar or Series).
    """
    return ts1 - hedge_ratio * ts2

def compute_zscore(spread, lookback=60, volatility_scale=True):
    """
    Compute z-score of the spread, optionally scaled by volatility bands.

    Args:
        spread (pd.Series): Spread time series.
        lookback (int): Rolling window size.
        volatility_scale (bool): If True, scale by rolling std of std.

    Returns:
        pd.Series: Z-score series.
    """
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()

    if volatility_scale:
        dynamic_band = std.rolling(lookback).mean()
        zscore = (spread - mean) / dynamic_band
    else:
        zscore = (spread - mean) / std

    return zscore.dropna()