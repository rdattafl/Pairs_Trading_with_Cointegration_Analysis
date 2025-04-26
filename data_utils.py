# This file encapsulates all data retrieval and cleaning logic for the "Pairs Trading with Cointegration Analysis" interactive web app.
# It contains modular functions to download price data using the yfinance API, clean and align data, and format it into proper multi-index or
# single-index DataFrames.

import yfinance as yf
import pandas as pd
import numpy as np

def download_multiple_pairs(ticker_pairs, start_date, end_date):
    """
    Download adjusted close price data for multiple stock pairs.

    Args:
        ticker_pairs (list of tuple): List of (ticker1, ticker2) pairs.
        start_date (str): Start date for historical data (e.g., '2018-01-01').
        end_date (str): End date for historical data (e.g., '2024-12-31').

    Returns:
        dict: Dictionary where each key is a (ticker1, ticker2) tuple and value is a cleaned Adj Close DataFrame.
    """
    all_tickers = sorted(set([ticker for pair in ticker_pairs for ticker in pair]))
    raw_data = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True)

    cleaned_pairs = {}
    for ticker1, ticker2 in ticker_pairs:
        try:
            df = pd.concat([
                raw_data['Close'][ticker1],
                raw_data['Close'][ticker2]
            ], axis=1)
            df.columns = [ticker1, ticker2]
            df.dropna(inplace=True)
            cleaned_pairs[(ticker1, ticker2)] = df
        except Exception as e:
            print(f"Error loading pair ({ticker1}, {ticker2}): {e}")

    return cleaned_pairs

def get_returns(data):
    """
    Calculate log or simple returns from price data.

    Args:
        data (pd.DataFrame): DataFrame of adjusted close prices.

    Returns:
        pd.DataFrame: Daily percentage returns.
    """
    return data.pct_change().dropna()