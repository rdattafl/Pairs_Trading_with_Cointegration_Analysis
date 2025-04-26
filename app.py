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

download_data = st.sidebar.button("📥 Download Ticker Data for Selected Pairs")

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

tabs = st.tabs([
    "📥 Data Download",
    "🔍 Cointegration Analysis",
    "⚙️ Strategy Logic",
    "📈 Backtesting",
    "📚 Glossary & Education"
])

if 'cleaned_returns_dict' not in st.session_state:
    st.session_state['cleaned_returns_dict'] = {}

# === 4. Data Download ===
# Call download_pair_data or download_multiple_pairs
# Clean data using clean_data(), get_returns()
with tabs[0]:
    st.header("📥 Load and Clean Historical Price Data")

    cleaned_returns_dict = st.session_state['cleaned_returns_dict']

    if not selected_pairs:
        st.warning("Please select at least one stock pair from the sidebar.")
    else:
        if download_data:
            with st.spinner("Downloading and processing price data..."):
                raw_pair_data = download_multiple_pairs(
                    selected_pairs,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )

            # st.write("✅ Raw downloaded data keys:", list(raw_pair_data.keys()))  # Force print to Streamlit

            temp_returns_dict = {}
            for pair_key, price_df in raw_pair_data.items():
                # st.write(f"🔍 Checking raw price_df for pair {pair_key}")
                # st.dataframe(price_df.head())  # Show first few rows if exists

                try:
                    if price_df is not None and not price_df.empty:
                        returns_df = get_returns(price_df)
                        # st.write(f"📈 Returns sample for {pair_key}:")
                        # st.dataframe(returns_df.head())

                        if not returns_df.empty:
                            temp_returns_dict[pair_key] = returns_df
                        else:
                            st.warning(f"⚠️ No returns computed for pair {pair_key}.")
                    else:
                        st.warning(f"⚠️ No price data available for {pair_key}.")
                except Exception as e:
                    st.error(f"🚨 Failed to process returns for {pair_key}: {e}")

            if not temp_returns_dict:
                st.error("❌ No valid pairs could be loaded. Check data or date ranges.")
                st.stop()
            else:
                st.success(f"✅ Successfully loaded {len(temp_returns_dict)} pair(s)!")
                st.session_state['cleaned_returns_dict'] = temp_returns_dict
                cleaned_returns_dict = temp_returns_dict
                
        # Show the returns dataframes in expandable sections
        for pair_key, returns_df in cleaned_returns_dict.items():
            with st.expander(f"View Returns for {pair_key[0]} / {pair_key[1]}"):
                st.dataframe(returns_df)


# === 5. Cointegration Analysis ===
# Call analyze_multiple_pairs(), display table and filtering
with tabs[1]:
    st.header("🔍 Cointegration Testing")

    if not st.session_state.get('cleaned_returns_dict'):
        st.warning("Please first download stock pair data from the sidebar.")
    else:
        st.subheader("Run Cointegration Tests Across Pairs")

        with st.spinner("Running Engle-Granger tests..."):
            coint_summary_df = analyze_multiple_pairs(cleaned_returns_dict)

        st.write("Coint pair downloaded keys:", list(coint_summary_df.keys()))

        st.success("✅ Cointegration analysis complete!")

        st.dataframe(
            coint_summary_df.style.background_gradient(cmap="YlGnBu"),
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📊 Filter Cointegrated Pairs Only")

        only_coint_pairs = st.checkbox("Show only pairs where cointegration detected (p < 0.05)", value=True)

        if only_coint_pairs:
            # Filtering: show if either direction has cointegration True
            filtered_df = coint_summary_df[
                (coint_summary_df.filter(like="Cointegrated").any(axis=1))
            ]
        else:
            filtered_df = coint_summary_df

        st.dataframe(
            filtered_df.style.background_gradient(cmap="PuBu"),
            use_container_width=True
        )


# === 6. Strategy Logic Per Pair ===
# Compute hedge ratio, spread, z-score, signals
# Visualize spread and z-score
with tabs[2]:
    st.header("⚙️ Strategy Logic and Signals")


# === 7. Backtesting (Per Pair or Portfolio) ===
# simulate_backtest() or simulate_portfolio_backtest()
# Show metrics and plots
with tabs[3]:
    st.header("📈 Backtesting and Portfolio Analysis")


# === 8. Glossary / Educational Panel ===
# Use st.sidebar.expander or st.expander blocks
with tabs[4]:
    st.header("📚 Glossary and Educational Insights")


# === 9. Export Options ===
# st.download_button for DataFrame CSV export




