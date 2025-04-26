# This file contains the code for the interactive web app that allows users to explore and simulate a pairs trading strategy 
# based on cointegration between stock pairs. Built with Streamlit, the app enables selection of stock pairs, visual inspection 
# of spread and z-score behavior, and statistical testing for cointegration. Users can adjust strategy parameters such as 
# entry/exit z-score thresholds, maximum holding period, take-profit/stop-loss levels, transaction costs, and cooldown period.
# The app then runs a full backtest, visualizes cumulative returns vs. individual assets, and reports detailed performance 
# metrics (CAGR, Sharpe ratio, max drawdown, win rate, average holding time). Educational tooltips and plots help build 
# intuitive understanding of market-neutral stat arb strategies.

# === 1. App Config and Imports ===

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
    "Select Stock Pairs (max 3):",
    options=available_pairs,
    default=[("KO", "PEP")],
    max_selections=4,
    help="Choose up to 3 stock pairs to analyze."
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
        # st.subheader("Run Cointegration Tests Across Pairs")

        with st.spinner("Running Engle-Granger tests..."):
            coint_results_dict = analyze_multiple_pairs(cleaned_returns_dict)

        # st.write("Coint pair downloaded keys:", list(coint_summary_df.keys()))

        st.success("✅ Cointegration analysis complete!")

        for (ticker1, ticker2), df in coint_results_dict.items():
            with st.expander(f"📈 {ticker1} / {ticker2} Cointegration Results"):
                st.dataframe(df.style.background_gradient(cmap="YlGnBu"), use_container_width=True)


# === 6. Strategy Logic Per Pair ===
# Compute hedge ratio, spread, z-score, signals
# Visualize spread and z-score
with tabs[2]:
    st.header("⚙️ Strategy Logic and Signals")

    # 1. Check if data is loaded
    cleaned_returns_dict = st.session_state.get('cleaned_returns_dict', {})

    if not cleaned_returns_dict:
        st.warning("Please first download and load some stock pair data.")
    else:
        # 2. Let user select which pair to visualize strategy logic for
        pair_selection = st.selectbox(
            "Select a Stock Pair to Analyze:",
            options=list(cleaned_returns_dict.keys()),
            format_func=lambda x: f"{x[0]} / {x[1]}"
        )

        if pair_selection:
            returns_df = cleaned_returns_dict[pair_selection]

            ticker1, ticker2 = returns_df.columns

            st.subheader(f"Selected Pair: {ticker1} / {ticker2}")

            # 3. Compute rolling hedge ratio
            hedge_ratio = calculate_hedge_ratio(
                returns_df[ticker1], 
                returns_df[ticker2],
                method="rolling", 
                window=60
            )

            # 4. Compute spread
            spread = compute_spread(
                returns_df[ticker1], 
                returns_df[ticker2], 
                hedge_ratio
            )

            st.success("✅ Spread computed using rolling hedge ratio!")

            # 5. Compute z-score
            zscore_vol_adj = compute_zscore(
                spread, 
                lookback=60, 
                volatility_scale=True
            )

            zscore_standard = compute_zscore(
                spread, 
                lookback=60, 
                volatility_scale=False
            )

            st.success("✅ Z-score (both standard and volatility-adjusted) computed!")

            # 6. Display a quick overview
            st.markdown("### Preview of Spread and Z-Score")
            st.dataframe(
                pd.DataFrame({
                    "Spread": spread,
                    "Z-Score (Standard)": zscore_standard,
                    "Z-Score (Volatility Adjusted)": zscore_vol_adj
                }).dropna().head(10)
            )



# === 7. Backtesting (Per Pair or Portfolio) ===
# simulate_backtest() or simulate_portfolio_backtest()
# Show metrics and plots
with tabs[3]:
    st.header("📈 Backtesting and Portfolio Analysis")


# === 8. Glossary / Educational Panel ===
# Use st.sidebar.expander or st.expander blocks
with tabs[4]:
    st.header("📚 Glossary and Educational Insights")

    st.markdown(
        """
        Learn about key concepts used in this Pairs Trading app.
        Expand the sections below for detailed explanations.
        """
    )

    with st.expander("Pairs Trading Overview"):
        st.markdown(
            """
            **Pairs Trading** is a market-neutral strategy that involves finding two historically correlated assets. 
            When their relationship deviates beyond a typical range (measured by statistical indicators like z-scores),
            a trade is executed to bet on the convergence of their prices. 
            One asset is typically shorted while the other is bought.
            """
        )

    with st.expander("Cointegration and Its Importance"):
        st.markdown(
            """
            **Cointegration** refers to a statistical relationship between two or more time series
            where their combination results in a stable, mean-reverting spread over time, 
            even if the individual series themselves are non-stationary.
            
            In pairs trading, identifying cointegrated pairs is critical because it ensures 
            that the spread between the assets behaves predictably, enabling mean-reversion strategies.
            """
        )

    with st.expander("Hedge Ratio"):
        st.markdown(
            """
            The **hedge ratio** represents the relative weight between two assets to create a stationary spread.
            It's typically calculated via **Ordinary Least Squares (OLS)** regression or using a rolling-window method.
            
            In this app, we allow you to choose either:
            - **Static OLS**: A single hedge ratio over the entire dataset.
            - **Rolling Hedge**: A dynamically updating hedge ratio based on recent data windows.
            """
        )

    with st.expander("Z-Score and Dynamic Thresholds"):
        st.markdown(
            """
            The **z-score** measures how far the current spread deviates from its historical mean,
            in units of standard deviations.
            
            - A **static z-score** assumes constant volatility over time.
            - A **dynamic z-score** adjusts for rolling volatility, making entry/exit signals more adaptive 
              to changing market conditions.
            
            Trading signals are generated when the z-score crosses predefined thresholds.
            """
        )

    with st.expander("Entry, Exit, and Risk Management"):
        st.markdown(
            """
            Once a trade is entered based on z-score thresholds:
            
            - **Exit** occurs either when the spread reverts back (z-score near zero), or based on a **take-profit** or **stop-loss** rule.
            - **Max Hold Days** ensures that no trade stays open indefinitely if conditions don't resolve.
            - **Cooldown Days** prevent immediate re-entry after closing a position to avoid over-trading.

            These risk controls are critical for managing drawdowns and maintaining strategy discipline.
            """
        )

    with st.expander("Slippage and Transaction Costs"):
        st.markdown(
            """
            **Slippage** accounts for imperfect execution when entering or exiting trades — prices move slightly 
            against you before the trade completes.
            
            **Transaction Costs** represent broker fees, bid/ask spreads, and other trading costs.
            
            This app allows you to configure both slippage (bps) and transaction costs (bps) to simulate more realistic trading performance.
            """
        )


# === 9. Export Options ===
# st.download_button for DataFrame CSV export




