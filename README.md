# Reinforcement Learning Trading Bot

This is a reinforcement learning trading bot built using Python, Gymnasium, and Stable-Baselines3. 
The bot uses the Soft Actor Critic (SAC) algorithm to learn profitable trading strategies while handling multiple stocks.

It includes features like:
- Support for multiple stocks
- Decision making using sentiment and price data using FinBERT and historical stock prices
- Stop-loss and take-profit logic
- Transaction cost simulation
- Customizable training and evaluation
- Backtesting and visualization tools

# API Key Setup

This bot uses the Finnhub module to retreive news headlines for each stock, you will need to use your own FinnHub API key for it to work.

You may use your API key in Line 26 of the code.
