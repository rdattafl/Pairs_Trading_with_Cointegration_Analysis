# This file encapsulates all data retrieval and cleaning logic for the "Pairs Trading with Cointegration Analysis" interactive web app.
# It contains modular functions to download price data using the yfinance API, clean and align data, and format it into proper multi-index or
# single-index DataFrames.

import yfinance as yf
import pandas as pd
import numpy as np

def download_pair_data(ticker_1, ticker_2, start_date, end_date):
    """
    Download adjusted close price data for two tickers from Yahoo Finance.

    Args:
        ticker_1 (str): First ticker symbol.
        ticker_2 (str): Second ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: MultiIndex DataFrame with adjusted close prices.
    """
    tickers = [ticker_1, ticker_2]
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    return data

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
    raw_data = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=False, group_by='ticker')

    cleaned_pairs = {}
    for ticker1, ticker2 in ticker_pairs:
        try:
            df = pd.concat([
                raw_data['Adj Close'][ticker1],
                raw_data['Adj Close'][ticker2]
            ], axis=1)
            df.columns = [ticker1, ticker2]
            df.dropna(inplace=True)
            cleaned_pairs[(ticker1, ticker2)] = df
        except Exception as e:
            print(f"Error loading pair ({ticker1}, {ticker2}): {e}")

    return cleaned_pairs

def clean_data(data):
    """
    Clean the raw Yahoo Finance data by removing NaNs.

    Args:
        data (pd.DataFrame): MultiIndex DataFrame from yfinance.

    Returns:
        pd.DataFrame: Cleaned DataFrame with only 'Adj Close' values.
    """
    # Ensure data has exactly two tickers
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Expected multi-index DataFrame from yfinance with 'Adj Close' columns.")

    adj_close = data['Adj Close']

    if adj_close.shape[1] != 2:
        raise ValueError("Expected exactly two tickers under 'Adj Close'.")

    adj_close = adj_close.dropna()
    return adj_close

def get_returns(data):
    """
    Calculate log or simple returns from price data.

    Args:
        data (pd.DataFrame): DataFrame of adjusted close prices.

    Returns:
        pd.DataFrame: Daily percentage returns.
    """
    return data.pct_change().dropna()