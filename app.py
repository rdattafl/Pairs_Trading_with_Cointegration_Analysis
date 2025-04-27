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
stop_loss = -st.sidebar.number_input("Stop Loss (%)", 0.0, 100.0, 5.0) / 100
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
    "📚 Glossary & Education",
    "💾 Export Options"
])

if 'cleaned_prices_dict' not in st.session_state:
    st.session_state['cleaned_prices_dict'] = {}


# === 4. Data Download ===
# Call download_pair_data or download_multiple_pairs
# Clean data using clean_data(), get_returns()
with tabs[0]:
    st.header("📥 Load and Clean Historical Price Data")

    cleaned_prices_dict = st.session_state['cleaned_prices_dict']

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

            temp_prices_dict = {}
            for pair_key, price_df in raw_pair_data.items():
                try:
                    if price_df is not None and not price_df.empty:
                        temp_prices_dict[pair_key] = price_df
                    else:
                        st.warning(f"⚠️ No price data available for {pair_key}.")
                except Exception as e:
                    st.error(f"🚨 Failed to process returns for {pair_key}: {e}")

            if not temp_prices_dict:
                st.error("❌ No valid pairs could be loaded. Check data or date ranges.")
            else:
                st.success(f"✅ Successfully loaded {len(temp_prices_dict)} pair(s)!")
                st.session_state['cleaned_prices_dict'] = temp_prices_dict
                cleaned_prices_dict = temp_prices_dict
                
        # Show the returns dataframes in expandable sections
        for pair_key, returns_df in cleaned_prices_dict.items():
            with st.expander(f"View Returns for {pair_key[0]} / {pair_key[1]}"):
                st.dataframe(returns_df)


# === 5. Cointegration Analysis ===
# Call analyze_multiple_pairs(), display table and filtering
with tabs[1]:
    st.header("🔍 Cointegration Testing")

    if not st.session_state.get('cleaned_prices_dict'):
        st.warning("Please first download stock pair data from the sidebar.")
    else:
        # st.subheader("Run Cointegration Tests Across Pairs")

        with st.spinner("Running Engle-Granger tests..."):
            coint_results_dict = analyze_multiple_pairs(cleaned_prices_dict)

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

    cleaned_prices_dict = st.session_state.get('cleaned_prices_dict', {})

    if not cleaned_prices_dict:
        st.warning("Please first download and load some stock pair data.")
    else:
        pair_selection = st.selectbox(
            "Select a Stock Pair to Analyze:",
            options=list(cleaned_prices_dict.keys()),
            format_func=lambda x: f"{x[0]} / {x[1]}"
        )

        if pair_selection:
            prices_df = cleaned_prices_dict[pair_selection]

            ticker1, ticker2 = prices_df.columns

            st.subheader(f"Selected Pair: {ticker1} / {ticker2}")

            hedge_ratio_1on2 = calculate_hedge_ratio(
                prices_df[ticker1], prices_df[ticker2],
                method="rolling", window=60
            )
            hedge_ratio_2on1 = calculate_hedge_ratio(
                prices_df[ticker2], prices_df[ticker1],
                method="rolling", window=60
            )

            spread_1on2 = compute_spread(
                prices_df[ticker1], prices_df[ticker2], hedge_ratio_1on2
            )
            spread_2on1 = compute_spread(
                prices_df[ticker2], prices_df[ticker1], hedge_ratio_2on1
            )

            st.success("✅ Rolling hedge ratios and spreads computed for both directions!")

            # === Expanders for visualization ===
            with st.expander("📈 Rolling Hedge Ratios"):
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(hedge_ratio_1on2, label=f"{ticker1} ~ {ticker2}")
                ax.plot(hedge_ratio_2on1, label=f"{ticker2} ~ {ticker1}")
                ax.set_title("Rolling Hedge Ratios (60-day Window)")
                ax.set_xlabel("Date")
                ax.set_ylabel("Hedge Ratio")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            with st.expander("📈 Spread Time Series"):
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(spread_1on2, label=f"Spread: {ticker1} - HR * {ticker2}")
                ax.plot(spread_2on1, label=f"Spread: {ticker2} - HR * {ticker1}")
                ax.set_title("Computed Spreads (Rolling Hedge Ratios)")
                ax.set_xlabel("Date")
                ax.set_ylabel("Spread Value")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            # === Save for further steps ===
            st.session_state['current_spreads'] = {
                'spread_1on2': spread_1on2,
                'spread_2on1': spread_2on1
            }
            st.session_state['current_hedge_ratios'] = {
                'hedge_ratio_1on2': hedge_ratio_1on2,
                'hedge_ratio_2on1': hedge_ratio_2on1
            }

            zscore_vol_adj_1 = compute_zscore(
                spread_1on2, 
                lookback=60, 
                volatility_scale=False
            )

            zscore_vol_adj_2 = compute_zscore(
                spread_2on1, 
                lookback=60, 
                volatility_scale=False
            )

            # st.line_chart(zscore_vol_adj_1)
            # st.line_chart(zscore_vol_adj_2)

            st.success("✅ Z-scores (volatility-adjusted) computed for both hedge directions!")

            signals_1 = generate_signals(
                zscore_vol_adj_1,
                entry_band=entry_threshold,
                exit_band=exit_threshold
            )

            signals_2 = generate_signals(
                zscore_vol_adj_2,
                entry_band=entry_threshold,
                exit_band=exit_threshold
            )

            st.success("✅ Trading signals generated for both hedge directions!")

            strategy_logic_df = pd.DataFrame(index=spread_1on2.dropna().index)

            strategy_logic_df[f"Spread ({ticker1} ~ {ticker2})"] = spread_1on2
            strategy_logic_df[f"Z-Score ({ticker1} ~ {ticker2})"] = zscore_vol_adj_1
            strategy_logic_df[f"Position ({ticker1} ~ {ticker2})"] = signals_1['curr_position']

            strategy_logic_df[f"Spread ({ticker2} ~ {ticker1})"] = spread_2on1
            strategy_logic_df[f"Z-Score ({ticker2} ~ {ticker1})"] = zscore_vol_adj_2
            strategy_logic_df[f"Position ({ticker2} ~ {ticker1})"] = signals_2['curr_position']

            strategy_logic_df.dropna(inplace=True)

            st.session_state['strategy_logic_df'] = strategy_logic_df
            st.session_state['selected_pair_name'] = f"{ticker1}_{ticker2}"

            st.markdown("### 📈 Strategy Signals and Z-Scores Overview")
            st.dataframe(
                strategy_logic_df.style.background_gradient(cmap="coolwarm"),
                use_container_width=True
            )


# === 7. Backtesting (Per Pair or Portfolio) ===
# simulate_backtest() or simulate_portfolio_backtest()
# Show metrics and plots
with tabs[3]:
    st.header("📈 Backtesting and Portfolio Analysis")

    # 1. Check if strategy logic is ready
    if 'strategy_logic_df' not in st.session_state:
        st.warning("⚠️ No strategy logic computed yet. Please generate signals in the previous tab.")
    else:
        st.subheader("🔹 Single Pair Backtest Simulation")

        strategy_logic_df = st.session_state['strategy_logic_df']
        # cleaned_ret_dict = st.session_state.get('cleaned_returns_dict', {})
        cleaned_prices_dict = st.session_state.get('cleaned_prices_dict', {})

        # 2. Reconstruct necessary inputs
        # Infer selected pair
        selected_pair = st.session_state.get('selected_pair_name', None)
        if not selected_pair:
            st.error("❌ Could not infer selected pair for backtesting.")

        # Parse the selected pair back into tuple
        ticker1, ticker2 = selected_pair.split("_")
        pair_key = (ticker1, ticker2)

        # if pair_key not in cleaned_returns_dict:
        #     st.error(f"❌ Data for selected pair {pair_key} not found.")

        prices_df = cleaned_prices_dict[pair_key]
        returns_df = get_returns(prices_df)

        # Extract hedge ratio and signals
        hedge_ratios = st.session_state['current_hedge_ratios']['hedge_ratio_1on2']
        signals = st.session_state['strategy_logic_df'][f"Position ({ticker1} ~ {ticker2})"].to_frame()
        signals.rename(columns={signals.columns[0]: 'curr_position'}, inplace=True)

        # 3. Prepare parameters
        backtest_params = {
            'max_hold_days': max_hold_days,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'cooldown_days': cooldown_days,
            'slippage': slippage_bps,
            'transaction_cost': tx_cost_bps
        }

        # 4. Run the backtest
        with st.spinner("Running backtest simulation..."):
            st.write(f"backtest_params: ", backtest_params)
            backtest_results = simulate_backtest(
                returns_df=returns_df,
                hedge_ratios=hedge_ratios,
                signals=signals,
                parameters=backtest_params
            )

        st.success("✅ Backtest simulation complete!")

        # 5. Visualize the cumulative returns
        st.subheader("📈 Strategy Cumulative Returns")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(backtest_results.index, backtest_results['cumulative_returns'], label='Cumulative Returns')
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.set_title("Backtested Strategy Performance")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # 6. Save backtest results to session state
        st.session_state['backtest_results'] = backtest_results

        with st.expander("🔍 View Detailed Backtest Data"):
            st.dataframe(backtest_results, use_container_width=True)    


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
with tabs[5]:
    st.header("💾 Export Results")

    st.markdown("You can export key outputs for further offline analysis.")

    if 'strategy_logic_df' in st.session_state and 'selected_pair_name' in st.session_state:
        filename = f"strategy_logic_signals_{st.session_state['selected_pair_name']}.csv"

        st.download_button(
            label=f"📥 Download Strategy Logic ({st.session_state['selected_pair_name']})",
            data=st.session_state['strategy_logic_df'].to_csv(index=True),
            file_name=filename,
            mime="text/csv",
        )
    else:
        st.info("ℹ️ Strategy logic not computed yet. Run a simulation first.")



