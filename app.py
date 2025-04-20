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

import streamlit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

