# This file contains the code for the interactive web app that allows users to explore and simulate a pairs trading strategy 
# based on cointegration between stock pairs. Built with Streamlit, the app enables selection of stock pairs, visual inspection 
# of spread and z-score behavior, and statistical testing for cointegration. Users can adjust strategy parameters such as 
# entry/exit z-score thresholds, maximum holding period, take-profit/stop-loss levels, transaction costs, and cooldown period.
# The app then runs a full backtest, visualizes cumulative returns vs. individual assets, and reports detailed performance 
# metrics (CAGR, Sharpe ratio, max drawdown, win rate, average holding time). Educational tooltips and plots help build 
# intuitive understanding of market-neutral stat arb strategies.

"""
Concrete additions our app could support:

Feature	Description
🔁 Multiple Pairs	Loop through and rank stock pairs by p-value from Engle-Granger test
⚙️ Parameter Tuning	Let user run grid search on entry/exit thresholds
🧮 Trade Log Table	Show per-trade return, duration, win/loss
💸 Transaction Costs	Optional toggles/sliders to apply slippage or fixed per-trade cost
💡 Dynamic z-score	Use volatility-adjusted or exponentially-weighted z-score
📊 Portfolio View	Combine multiple cointegrated pairs for aggregate market-neutral strategy
📥 Export CSV	Allow download of trade logs or cumulative return series
🧠 Educational Tooltips	Use Streamlit expander/tooltips to explain concepts interactively (e.g., “What is Cointegration?”)
"""

# Section 1 - App configuration and imports

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from data_utils import *
from pairs_analysis import *
from strategy_simulation import *

# === 2. Page Config ===
st.set_page_config(page_title="Pairs Trading Simulator", layout="wide")

# === 3. Sidebar Inputs ===
st.sidebar.header("⚙️ Strategy Configuration")

start_date = st.sidebar.date_input("Start Date", value=date(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date(2023, 12, 31))

available_pairs = [
    ("KO", "PEP"), ("MSFT", "AAPL"), ("JPM", "BAC"),
    ("V", "MA"), ("XOM", "CVX")
]

selected_pairs = st.sidebar.multiselect(
    "Select Stock Pairs (max 5):",
    options=available_pairs,
    default=[("KO", "PEP")],
    help="Choose up to 5 stock pairs to analyze."
)

hedge_method = st.sidebar.radio(
    "Hedge Ratio Estimation",
    ["ols", "rolling"],
    help="OLS fits a static beta; rolling uses a dynamic windowed estimate."
)

zscore_type = st.sidebar.radio(
    "Z-score Type",
    ["Static", "Dynamic"],
    help="Dynamic z-scores use scaled volatility bands."
)

entry_threshold = st.sidebar.slider("Entry Threshold", 0.5, 3.0, 1.5, 0.1)
exit_threshold = st.sidebar.slider("Exit Threshold", 0.0, 1.0, 0.05, 0.05)

st.sidebar.markdown("### Trade Logic & Risk Settings")
max_hold_days = st.sidebar.number_input("Max Hold Days", 1, 60, 20)
take_profit = st.sidebar.number_input("Take Profit (%)", 0.0, 100.0, 10.0) / 100
stop_loss = st.sidebar.number_input("Stop Loss (%)", 0.0, 100.0, 5.0) / 100
cooldown_days = st.sidebar.number_input("Cooldown Days", 0, 10, 5)

slippage_bps = st.sidebar.number_input("Slippage (bps)", 0, 100, 10)
tx_cost_bps = st.sidebar.number_input("Transaction Cost (bps)", 0, 100, 5)

run_portfolio = st.sidebar.checkbox("Run Portfolio Backtest", value=False)
top_n = st.sidebar.slider("Top N Pairs to Trade", 1, 5, 3)

# === 4. Data Download ===
# Call download_pair_data or download_multiple_pairs
# Clean data using clean_data(), get_returns()
st.subheader("📥 Load and Clean Historical Price Data")

if not selected_pairs:
    st.warning("Please select at least one stock pair from the sidebar.")
    st.stop()

with st.spinner("Downloading price data..."):
    raw_pair_data = download_multiple_pairs(
        selected_pairs,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )

    cleaned_returns_dict = {}
    for pair_key, price_df in raw_pair_data.items():
        try:
            returns_df = get_returns(price_df)
            cleaned_returns_dict[pair_key] = returns_df
        except Exception as e:
            st.error(f"Failed to process returns for pair {pair_key}: {e}")

if not cleaned_returns_dict:
    st.error("❌ No valid pairs could be loaded. Check data or date ranges.")
    st.stop()
else:
    st.success(f"✅ Loaded {len(cleaned_returns_dict)} valid pair(s).")


# === 5. Cointegration Analysis ===
# Call analyze_multiple_pairs(), display table and filtering


# === 6. Strategy Logic Per Pair ===
# Compute hedge ratio, spread, z-score, signals
# Visualize spread and z-score


# === 7. Backtesting (Per Pair or Portfolio) ===
# simulate_backtest() or simulate_portfolio_backtest()
# Show metrics and plots


# === 8. Glossary / Educational Panel ===
# Use st.sidebar.expander or st.expander blocks


# === 9. Export Options ===
# st.download_button for DataFrame CSV export




