# This file will be used to run all statistical tests and analytics on the chosen stock pairs.
# It will be useful to show the results of this file to the user before backtesting.

def test_cointegration(ts1, ts2):
    pass

def johansen_test(log_df):
    pass

def calculate_hedge_ratio(ts1, ts2, method="rolling" or "ols"):
    pass

def compute_spread(ts1, ts2, hedge_ratio):
    pass

def compute_zscore(spread, lookback):
    pass