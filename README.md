# Pairs Trading with Cointegration Analysis

This repo contains the code for an interactive web application that allows users to explore, simulate, and visualize a **market-neutral statistical arbitrage strategy** based on cointegration analysis between stock pairs.

The application is built using **Streamlit**, providing an intuitive and educational interface for understanding and evaluating pairs trading strategies. Users can dynamically select stock pairs, adjust trading logic parameters, and view detailed backtesting results, including strategy performance metrics and visualizations.

---

## 📚 Table of Contents

- [Core Objectives](#core-objectives)
- [Key Features](#key-features)
- [How It Works (Quick Flow)](#how-it-works-quick-flow)
- [Modular Structure](#modular-structure)
- [Coming Enhancements](#coming-enhancements)

---

## Core Objectives

- Help users **identify cointegrated stock pairs** using rigorous statistical tests (Engle-Granger, Johansen).
- Construct **mean-reverting spread portfolios** based on a dynamically estimated hedge ratio (OLS or rolling).
- Let users interactively configure and simulate a **pairs trading backtest**, including:
  - Z-score thresholds (entry/exit)
  - Maximum holding period
  - Take-profit and stop-loss exits
  - Cooldown periods between trades
  - Transaction cost / slippage estimates
- Provide **educational context** via tooltips and expandable explanations for all core strategy components.
- Allow users to **compare cumulative returns** of the strategy against the individual assets.
- Display detailed **performance metrics**: Total return, CAGR, volatility, Sharpe ratio, drawdowns, win rate, and average holding time.
- Future-proof: Designed for generalization to multi-pair strategies, portfolio backtests, and extended analytics.

---

## Key Features

- **Stock Pair Selection**: Choose from predefined or manually entered tickers for backtesting.
- **Cointegration Testing**: Evaluate long-term relationships using Engle-Granger & Johansen tests.
- **Spread & Z-Score Construction**:
  - Calculate spread via OLS or rolling beta.
  - Z-score computed over customizable rolling window.
- **Backtest Engine**:
  - Realistic simulation of trade entries/exits based on signals
  - Configurable stop-loss / take-profit bands
  - Max holding period and cooldown logic
  - Optional **transaction costs or slippage** built-in
- **Cumulative Return Visuals**:
  - Plot strategy vs. individual stocks (e.g., KO, PEP)
  - Log-scale toggle and zoomable charts
- **Performance Metrics Dashboard**:
  - Total Return, CAGR, Annual Volatility, Sharpe Ratio
  - Max Drawdown
  - Number of Trades, Win Rate, Avg. Holding Period
- **Interactive Education**:
  - Tooltips for key terms (e.g., cointegration, hedge ratio)
  - Strategy rationale breakdowns in collapsible sections

---

## How It Works (Quick Flow)

1. **Select Stock Pair**  
   Input two stock tickers (e.g., KO and PEP).
2. **Load Data**  
   Historical price data is fetched (default: 2018–2024) via `yfinance`.
3. **Test Cointegration**  
   Run Engle-Granger and Johansen tests, display p-values and interpretation.
4. **Compute Spread & Z-Score**  
   Construct spread using OLS or rolling hedge ratio. Normalize via z-score.
5. **Configure Strategy Parameters**  
   Customize thresholds, holding period, stop-loss, take-profit, cooldown, and transaction costs.
6. **Simulate Backtest**  
   Execute strategy logic, manage position states, and track cumulative returns.
7. **Visualize Performance**  
   Compare strategy returns vs. individual asset returns.
8. **Review Metrics**  
   Analyze key performance indicators and trade-level stats.

---

## Modular Structure

```text
notebook/
├── app.py                 # Streamlit web app
├── data_utils.py          # Data loading, cleaning, and return generation
├── pairs_analysis.py      # Cointegration tests, hedge ratio, spread, z-score logic
├── strategy_simulation.py # Backtest engine, trade tracking, performance metrics
├── README.md              # Project documentation
├── requirements.txt       # Required Python packages
└── assets/                # Optional folder for logos or saved configs
```

---

## Coming Enhancements

- **Multi-pair portfolio backtesting** with ranking logic to simulate strategies across multiple cointegrated pairs.
- **Transaction cost and slippage modeling** for more realistic performance estimation.
- **Dynamic (volatility-adjusted) z-score thresholds** to adapt entry/exit logic to market regimes.
- **Embedded educational tooltips**, a glossary sidebar, and “Explain this” buttons to improve learning and transparency throughout the app.