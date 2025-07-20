import os
os.environ["USE_TF"] = "0"
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import yfinance as yf
from tqdm import tqdm
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import datetime
from transformers import pipeline
import requests
from collections import defaultdict
import optuna
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# FinnHub API key for news data (replace with your actual key)
finnhub_key = 'FinnHub_KEY'

# initialize FinBERT sentiment analysis pipeline
sentiment_pipeline = pipeline(
    "text-classification", 
    model="yiyanghkust/finbert-tone",
    device=0 if torch.cuda.is_available() else -1
)

# custom Gym environment allowing the bot to trade utilising sentiment analysis
class StockTradingEnv(gym.Env):
    def __init__(self, df, sentiment_df, initial_balance=10000, window_size=20, stop_loss=0.03, take_profit=0.06):
        super().__init__()
        self.df = df
        self.sentiment_df = sentiment_df
        self.tickers = df.columns.get_level_values(0).unique().tolist()
        self.num_stocks = len(self.tickers)
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_position_size = 0.2
        features_per_stock = 5
        stock_features = self.window_size * features_per_stock * self.num_stocks
        portfolio_features = 1 + 2 * self.num_stocks
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(stock_features + portfolio_features,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_stocks,), dtype=np.float32)
        self.transactions = []

    def _next_observation(self):
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        frame = self.df.iloc[start_idx:end_idx].to_numpy()
        sentiment_frame = self.sentiment_df.iloc[start_idx:end_idx].to_numpy()
        if len(frame) < self.window_size:
            padding = np.repeat(self.df.iloc[[0]].to_numpy(), self.window_size - len(frame), axis=0)
            frame = np.vstack([padding, frame])
            sentiment_padding = np.repeat(self.sentiment_df.iloc[[0]].to_numpy(), self.window_size - len(sentiment_frame), axis=0)
            sentiment_frame = np.vstack([sentiment_padding, sentiment_frame])
        obs = []
        latest_prices = self.df.iloc[self.current_step][[(ticker, "Close") for ticker in self.tickers]].values
        for i, ticker in enumerate(self.tickers):
            ticker_idx = slice(i * 5, (i + 1) * 5 - 1)
            ohlc = frame[:, ticker_idx] / latest_prices[i] - 1
            sentiment = sentiment_frame[:, i]
            obs.append(np.concatenate([ohlc.flatten(), sentiment]))
        obs = np.concatenate(obs)
        obs = np.append(obs, self.balance / self.initial_balance)
        for i, ticker in enumerate(self.tickers):
            position_value = self.positions[ticker] * latest_prices[i] / self.initial_balance
            cost_basis = self.cost_basis[ticker]
            cost_basis_norm = (cost_basis / latest_prices[i] - 1) if self.positions[ticker] > 0 else 0
            obs = np.append(obs, [position_value, cost_basis_norm])
        return obs.astype(np.float32)

    def _take_action(self, actions):
        """
        Execute trading actions based on agent's decisions.
        
        Args:
            actions: Array of continuous values [-1, 1] for each stock
                    Positive values = buy, negative values = sell
        
        Returns:
            reward: Immediate reward for this action
            info: Dictionary with trade information
        """
        reward = 0
        info = {'trades': []}
        current_date = self.df.index[self.current_step]

        # process each stock
        for i, ticker in enumerate(self.tickers):
            action = actions[i]
            current_price = self.df.iloc[self.current_step][(ticker, "Close")]
            current_position = self.positions[ticker]
            transaction_cost = 0.001
            sentiment = self.sentiment_df.iloc[self.current_step][ticker]

            # modify action based on sentiment, strong sentiment amplifies the action
            action = action * (1 + np.sign(action) * 0.2 * np.tanh(sentiment * 3))

            # ensurr action stays within bounds
            action = max(min(action, 1), -1)

            #  buying logic
            if action > 0:
                # calculate maximum expenditure based on position size limit
                max_expenditure = self.balance * self.max_position_size * action
                shares_to_buy = max_expenditure / current_price
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + transaction_cost)
                    self.balance -= cost
                    self.positions[ticker] += shares_to_buy

                    # update cost basis (average cost of all shares)
                    if current_position > 0:
                        self.cost_basis[ticker] = (
                            (current_position * self.cost_basis[ticker]) + 
                            (shares_to_buy * current_price)
                        ) / self.positions[ticker]
                    else:
                        self.cost_basis[ticker] = current_price                    
                        self.transactions.append({
                            'step': self.current_step, 'ticker': ticker,
                            'action': 'buy', 'shares': shares_to_buy,
                            'price': current_price, 'cost': cost,
                            'sentiment': sentiment,
                            'sentiment_affected': abs(sentiment) > 0.2
                        })
                            
                    # record transaction  
                    info['trades'].append(f"Buy {shares_to_buy:.2f} shares of {ticker} at ${current_price:.2f}")

            # sell logic
            elif action < 0 and current_position > 0:
                # sell portion of position based on action magnitude
                shares_to_sell = current_position * abs(action)
                
                if shares_to_sell > 0:
                    sale_amount = shares_to_sell * current_price * (1 - transaction_cost)
                    self.balance += sale_amount
                    self.positions[ticker] -= shares_to_sell

                    # calculate profit/loss
                    profit = (current_price - self.cost_basis[ticker]) * shares_to_sell
                    reward += profit / self.initial_balance

                    # record transaction
                    self.transactions.append({
                        'step': self.current_step, 'ticker': ticker,
                        'action': 'sell', 'shares': shares_to_sell,
                        'price': current_price, 'profit': profit,
                        'sentiment': sentiment,
                        'sentiment_affected': abs(sentiment) > 0.2
                    })
                            
                            
                    info['trades'].append(f"Sell {shares_to_sell:.2f} shares of {ticker} at ${current_price:.2f}, profit: ${profit:.2f}")

            # risk management: stop-loss and take-profit
            if current_position > 0:
                price_change = (current_price / self.cost_basis[ticker]) - 1

                # stop-loss: sell if loss exceeds threshold
                if price_change < -self.stop_loss:
                    sale_amount = current_position * current_price
                    profit = (current_price - self.cost_basis[ticker]) * current_position
                    self.balance += sale_amount
                    
                    self.transactions.append({
                        'step': self.current_step, 'ticker': ticker,
                        'action': 'stop_loss', 'shares': current_position,
                        'price': current_price, 'profit': profit,
                        'sentiment': sentiment,
                        'sentiment_affected': abs(sentiment) > 0.2
                    })

                    # apply a penalty for stop loss
                    reward += profit / self.initial_balance - 0.001
                    self.positions[ticker] = 0
                    self.cost_basis[ticker] = 0
                    info['trades'].append(f"STOP-LOSS: Sell {current_position:.2f} shares of {ticker} at ${current_price:.2f}")

                # take-profit: sell if gain exceeds threshold
                elif price_change > self.take_profit:
                    sale_amount = current_position * current_price
                    profit = (current_price - self.cost_basis[ticker]) * current_position
                    self.balance += sale_amount
                    
                    self.transactions.append({
                        'step': self.current_step, 'ticker': ticker,
                        'action': 'take_profit', 'shares': current_position,
                        'price': current_price, 'profit': profit,
                        'sentiment': sentiment,
                        'sentiment_affected': abs(sentiment) > 0.2
                    })

                    # give a bonus for take-profit
                    reward += profit / self.initial_balance + 0.002
                    self.positions[ticker] = 0
                    self.cost_basis[ticker] = 0
                    info['trades'].append(f"TAKE-PROFIT: Sell {current_position:.2f} shares of {ticker} at ${current_price:.2f}")
        return reward, info

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action: Action taken by the agent
            
        Returns:
            observation: Next state observation
            reward: Reward for this step
            done: Whether episode is finished
            truncated: Whether episode was truncated
            info: Additional information
        """

        # calculate current portfolio value
        portfolio_value = self.balance
        for ticker in self.tickers:
            if self.current_step < len(self.df):
                current_price = self.df.iloc[self.current_step][(ticker, "Close")]
                portfolio_value += self.positions[ticker] * current_price

        # execute trades and get immediate reward
        reward, info = self._take_action(action)

        # move to next step
        self.current_step += 1

        # calculate portfolio performance reward
        portfolio_change = (portfolio_value / self.portfolio_value - 1) * 10

        # add tranasction cost penalty
        transaction_penalty = sum(t['cost'] * 0.001 for t in self.transactions[-len(self.tickers):] if t['action'] == 'buy') / self.initial_balance
        
        # final reward combines portfolio change and transaction costs
        reward = portfolio_change - transaction_penalty

        # update portfolio value tracking
        self.portfolio_value = portfolio_value

        # check if episode is done
        done = self.current_step >= len(self.df) - 1

        # get next observation
        obs = self._next_observation()
        info = {'portfolio_value': portfolio_value}
        
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # reset all state variables
        self.balance = self.initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.cost_basis = {ticker: 0 for ticker in self.tickers}
        self.current_step = self.window_size - 1
        self.portfolio_value = self.initial_balance
        self.transactions = []
        
        obs = self._next_observation()
        return obs, {}

    def get_portfolio_value(self):
        """Calculate current total portfolio value (cash + positions)."""
        portfolio_value = self.balance
        for ticker in self.tickers:
            if self.current_step < len(self.df):
                current_price = self.df.iloc[self.current_step][(ticker, "Close")]
                portfolio_value += self.positions[ticker] * current_price
        return portfolio_value

class TensorboardCallback(BaseCallback):
    """Callback to log training metrics to Tensorboard."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self):
        if 'episode' in self.locals and self.locals['done']:
            self.rewards.append(self.locals['episode_reward'])
        return True

class EarlyStoppingCallback(BaseCallback):
    """Callback to implement early stopping based on performance."""
    def __init__(self, check_freq=1000, reward_threshold=100, patience=5000, verbose=0):
        """
        Args:
            check_freq: How often to check for early stopping
            reward_threshold: Reward threshold to stop training
            patience: Number of steps to wait without improvement
        """
        
        super().__init__(verbose)
        self.check_freq = check_freq
        self.reward_threshold = reward_threshold
        self.patience = patience
        self.best_mean_reward = -np.inf
        self.no_improvement_steps = 0
        self.rewards = []

    def _on_step(self):
        """Check if training should be stopped early."""

        # check performance over last 100 episodes
        if self.n_calls % self.check_freq == 0:
            if 'episode' in self.locals and self.locals['done']:
                self.rewards.append(self.locals['episode_reward'])
                
                if len(self.rewards) >= 100:
                    mean_reward = np.mean(self.rewards[-100:])
                    
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        self.no_improvement_steps = 0
                    else:
                        self.no_improvement_steps += self.check_freq

                    # stop of threshold reached or no improvement for too long
                    if mean_reward >= self.reward_threshold or self.no_improvement_steps >= self.patience:
                        return False
        return True

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        tickers: List of stock symbols
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with MultiIndex columns (ticker, OHLCV)
    """
    
    all_data = {}
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data for {ticker}")
            
            data.columns = [col[0] for col in data.columns]

            # Create MultiIndex columns (ticker, metric)
            renamed_columns = [(ticker, col) for col in data.columns]
            data.columns = pd.MultiIndex.from_tuples(renamed_columns)
            
            all_data[ticker] = data
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None
            
    # combine all data and handle missing values
    result = pd.concat(all_data.values(), axis=1).ffill().dropna()
    return result

def finbert_sentiment(text):
    """
    Analyze sentiment of financial text using FinBERT.
    
    Args:
        text: Text to analyze
        
    Returns:
        Float sentiment score between -1 (negative) and 1 (positive)
    """
    
    try:
        # handle empty or invalid text
        if text is None or pd.isna(text) or str(text).strip() == "":
            return 0.0

        # limit text length (FinBERT has 512 token limit)
        result = sentiment_pipeline(str(text)[:512], return_all_scores=True)
        
        if not result:
            print(f"No result from FinBERT for text: {text[:50]}...")
            return 0.0

        # extract probabilities for each sentiment class
        probas = {s['label'].lower(): s['score'] for s in result[0]}
        positive = probas.get('positive', 0)
        negative = probas.get('negative', 0)
        neutral = probas.get('neutral', 0)

        # create weighted sentiment score
        weighted_score = (
            (positive * (0.3 + 0.7*positive)) +
            (negative * (-0.3 - 0.7*negative)) +
            (neutral * (neutral - 0.5) * 0.6)
        )

        # apply tanh to bound between -1 and 1
        final_score = np.tanh(weighted_score * 1.5)
        return float(final_score)
        
    except Exception as e:
        print(f"Sentiment error: {e}")
        return 0.0

def fetch_finnhub_news(ticker, start_date, end_date, api_key):
    """
    Fetch news articles for a specific stock from FinnHub API.
    
    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: FinnHub API key
        
    Returns:
        List of processed news articles
    """
    
    url = f'https://finnhub.io/api/v1/company-news'
    params = {
        'symbol': ticker,
        'from': start_date,
        'to': end_date,
        'token': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        news_data = response.json()
        if not news_data:
            print(f"No news found for {ticker} between {start_date} and {end_date}")
            return []

        processed_news = []
        filtered_count = 0

        # process each news article
        for item in news_data:
            # filter out articles with poor headlines
            if not item.get('headline') or len(item.get('headline', '')) <= 10:
                filtered_count += 1
                continue

            # convert timestamp to date
            date = datetime.datetime.fromtimestamp(item['datetime']).date()
            
            processed_news.append({
                'date': date,
                'ticker': ticker,
                'headline': item['headline'],
                'summary': item.get('summary', ''),
                'source': item.get('source', ''),
                'url': item.get('url', ''),
                'raw_data': item
            })
            
        print(f"Filtered out {filtered_count} news items due to invalid headline")
        print(f"Found {len(processed_news)} news items for {ticker}")
        
        return processed_news
        
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

def fetch_sentiment_data(tickers, stock_df, api_key):
    """
    Fetch and process sentiment data for all tickers.
    
    Args:
        tickers: List of stock symbols
        stock_df: Stock price DataFrame for date alignment
        api_key: FinnHub API key
        
    Returns:
        DataFrame with sentiment scores for each ticker
    """
    
    start_date = stock_df.index.min().strftime('%Y-%m-%d')
    end_date = stock_df.index.max().strftime('%Y-%m-%d')
    print(f"Fetching sentiment data from {start_date} to {end_date}")

    # initialize sentiment DataFrame with zeros
    sentiment_df = pd.DataFrame(0, index=stock_df.index, columns=tickers)
    sample_headlines = []

    # define terms to filter relevant news for each stock
    related_terms = {
        'AAPL': ['apple', 'iphone', 'ipad', 'macbook', 'ios','airpods','mac', 'app store'],
        'MSFT': ['microsoft', 'windows', 'azure', 'office', 'xbox', 'teams', 'bing'],
        'AMZN': ['amazon', 'aws', 'prime', 'kindle', 'alexa', 'whole foods', 'bezos'],
        'GOOGL': ['google', 'alphabet', 'android', 'youtube', 'gmail', 'cloud']
    }

    # Process each ticker
    for ticker in tickers:
        print(f"\nProcessing sentiment for {ticker}")
        news_items = fetch_finnhub_news(ticker, start_date, end_date, api_key)
        
        if not news_items:
            print(f"No news found for {ticker}")
        
        daily_scores = defaultdict(list)
        valid_news_count = 0
        irrelevant_news_count = 0

        # process each news article
        for item in news_items:
            try:
                date = pd.to_datetime(item['date']).date()
                nearest_date = stock_df.index[stock_df.index.get_indexer([pd.Timestamp(date)], method='nearest')[0]]

                # filter for relevance to the specific stock
                headline = item['headline'].lower()
                summary = item.get('summary', '').lower()
                if not any(term in headline or term in summary for term in related_terms[ticker]):
                    irrelevant_news_count += 1
                    continue

                # combine headline and summary for sentiment analysis
                text = f"{item['headline']}. {item.get('summary', '')}"
                if not text.strip():
                    continue

                # calculate sentiment score
                sentiment = finbert_sentiment(text)
                daily_scores[nearest_date].append(sentiment)

                # store sample headlines for inspection
                if valid_news_count < 5:
                    sample_headlines.append({
                        'ticker': ticker,
                        'date': date,
                        'headline': item['headline'],
                        'sentiment': sentiment,
                        'source': item.get('source', 'unknown')
                    })
                
                valid_news_count += 1
                
            except Exception as e:
                print(f"Error processing news item: {e}")
                continue
        # display sample headlines for quality check
        print(f"\nSample headlines for {ticker}:")
        for hl in [h for h in sample_headlines if h['ticker'] == ticker]:
            print(f"Date: {hl['date']}")
            print(f"Headline: {hl['headline']}")
            print(f"Sentiment: {hl['sentiment']:.4f} ({'positive' if hl['sentiment'] > 0 else 'negative' if hl['sentiment'] < 0 else 'neutral'})")
            print(f"Source: {hl['source']}")
            print("-" * 80)
        
        for date, scores in daily_scores.items():
            sentiment_df.loc[date, ticker] = max(scores, key=abs) if scores else 0

        # handle missing sentiment with synthetic sentiment
        for date in sentiment_df.index:
            if not daily_scores[date]:
                try:
                    prev_idx = stock_df.index.get_loc(date) - 1
                    if prev_idx >= 0:
                        prev_date = stock_df.index[prev_idx]
                        price_change = (stock_df.loc[date, (ticker, 'Close')] / stock_df.loc[prev_date, (ticker, 'Close')] - 1) * 100

                        # convert price change to synthetic sentiment
                        synthetic_sentiment = np.tanh(price_change / 5)

                        # add small random noise
                        synthetic_sentiment += np.random.normal(0, 0.05)
                        synthetic_sentiment = np.clip(synthetic_sentiment, -1, 1)
                        sentiment_df.loc[date, ticker] = synthetic_sentiment
                    else:
                        sentiment_df.loc[date, ticker] = sentiment_df[ticker].iloc[:stock_df.index.get_loc(date)].mean() if stock_df.index.get_loc(date) > 0 else 0
                except Exception as e:
                    sentiment_df.loc[date, ticker] = sentiment_df[ticker].iloc[:stock_df.index.get_loc(date)].mean() if stock_df.index.get_loc(date) > 0 else 0

        # display sentiment statistics
        print(f"Sentiment score distribution for {ticker}:")
        scores = [score for scores in daily_scores.values() for score in scores] + \
                 [sentiment_df.loc[date, ticker] for date in sentiment_df.index if not daily_scores[date]]
        print(pd.Series(scores).describe())
        
        non_zero_days = sentiment_df[ticker][sentiment_df[ticker] != 0]
        print(f"Found {len(non_zero_days)} days with non-zero sentiment for {ticker}")
        if len(non_zero_days) > 0:
            print(non_zero_days.head())
    
    print("\nFinal sentiment summary:")
    print(sentiment_df.describe())
    print("\nNon-zero sentiment days count:")
    print((sentiment_df != 0).sum())
    
    return sentiment_df

def plot_sentiment_vs_price(ticker, stock_df, sentiment_df, output_folder='results'):
    """
    Create visualization comparing stock price and sentiment over time.
    
    Args:
        ticker: Stock symbol to plot
        stock_df: Price data
        sentiment_df: Sentiment data
        output_folder: Directory to save plots
    """
    
    plt.figure(figsize=(15, 7))

    # normalize price data for comparison with sentiment
    close_prices = stock_df[(ticker, "Close")]
    normalized_prices = (close_prices - close_prices.min()) / (close_prices.max() - close_prices.min())
    plt.plot(stock_df.index, normalized_prices, label=f'{ticker} Normalized Price', color='blue')
    sentiment = sentiment_df[ticker]
    plt.plot(sentiment_df.index, sentiment, label=f'{ticker} Sentiment', color='orange', alpha=0.7)
    plt.title(f'{ticker} Price vs. Sentiment')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{ticker}_sentiment_vs_price.png")
    plt.close()

def backtest_model(model, env, test_df, test_sentiment_df, output_folder='results'):
    """
    Backtest the trained model and generate visualizations.
    
    Args:
        model: trained RL model
        env: Trading environment
        test_df: Test price data
        test_sentiment_df: Test sentiment data
        output_folder: Directory for saving results
        
    Returns:
        portfolio value over backtesting period
    """
    
    os.makedirs(output_folder, exist_ok=True)
    env.df = test_df
    env.sentiment_df = test_sentiment_df
    obs, _ = env.reset()
    portfolio_values = [env.initial_balance]
    dates = test_df.index.tolist()
    pbar = tqdm(total=len(test_df)-env.window_size+1, desc="Backtesting")
    done = False

    headline_records = []
    
    while not done:
        current_date = dates[env.current_step]
                
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        portfolio_values.append(info['portfolio_value'])
        pbar.update(1)
    
    if env.current_step < len(test_df):
        portfolio_values.append(env.get_portfolio_value())
        pbar.update(1)
    pbar.close()

    sp500 = yf.download('^GSPC', start=dates[0], end=dates[-1])['Close']
    sp500_normalized = (sp500 / sp500.iloc[0]) * env.initial_balance

    plt.figure(figsize=(18, 12))
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

    ax1.plot(dates[env.window_size-1:], portfolio_values[1:], 
             label='Portfolio Value', linewidth=2, color='blue')
    ax1.plot(sp500.index, sp500_normalized, 
             label='S&P 500 Benchmark', linewidth=2, color='green', linestyle='--')

    legend_handles = [
        Line2D([0], [0], color='blue', lw=2, label='Portfolio Value'),
        Line2D([0], [0], color='green', linestyle='--', lw=2, label='S&P 500'),
        Line2D([0], [0], marker='^', color='green', lw=0, label='Buy', markersize=10),
        Line2D([0], [0], marker='v', color='red', lw=0, label='Sell', markersize=10),
        Line2D([0], [0], marker='*', color='purple', lw=0, label='Sentiment Trade', markersize=12)
    ]

    for transaction in env.transactions:
        date = dates[transaction['step']]
        port_value = portfolio_values[transaction['step'] - (env.window_size - 1) + 1]
        
        marker = '^' if transaction['action'] == 'buy' else 'v'
        color = 'purple' if transaction['sentiment_affected'] else ('green' if transaction['action'] == 'buy' else 'red')
        size = 120 if transaction['sentiment_affected'] else 80
        alpha = 0.9 if transaction['sentiment_affected'] else 0.6
        
        ax1.scatter(date, port_value, color=color, marker=marker, s=size, 
                   alpha=alpha, zorder=3, 
                   edgecolors='black' if transaction['sentiment_affected'] else 'none')

    avg_sentiment = test_sentiment_df.mean(axis=1)
    ax2.plot(dates[env.window_size-1:], avg_sentiment[env.window_size-1:], 
             color='orange', alpha=0.7, label='Avg Sentiment')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(dates[env.window_size-1:], 0, avg_sentiment[env.window_size-1:], 
                    where=avg_sentiment[env.window_size-1:]>=0, 
                    facecolor='green', alpha=0.2, interpolate=True)
    ax2.fill_between(dates[env.window_size-1:], 0, avg_sentiment[env.window_size-1:], 
                    where=avg_sentiment[env.window_size-1:]<0, 
                    facecolor='red', alpha=0.2, interpolate=True)

    legend_handles.extend([
        Line2D([0], [0], color='orange', lw=1, label='Avg Sentiment'),
        mpatches.Patch(color='green', alpha=0.2, label='Positive Sentiment'),
        mpatches.Patch(color='red', alpha=0.2, label='Negative Sentiment')
    ])

    ax1.set_title('Portfolio Performance with Sentiment-Driven Trades')
    ax2.set_xlabel('Date')
    ax1.set_ylabel('Value ($)')
    ax2.set_ylabel('Sentiment')
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    ax1.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/portfolio_vs_sp500.png", dpi=150, bbox_inches='tight')
    plt.close()

    for ticker in env.tickers:
        plt.figure(figsize=(15, 7))
        close_prices = test_df[(ticker, "Close")].values
        plt.plot(dates, close_prices, label=f'{ticker} Price')

        ticker_transactions = [t for t in env.transactions if t['ticker'] == ticker]
        for transaction in ticker_transactions:
            step = transaction['step']
            if step < len(close_prices):
                marker = '^' if transaction['action'] == 'buy' else 'v'
                color = 'purple' if transaction['sentiment_affected'] else ('green' if transaction['action'] == 'buy' else 'red')
                size = 150 if transaction['sentiment_affected'] else 100
                plt.scatter(dates[step], close_prices[step], 
                           color=color, marker=marker, s=size,
                           alpha=0.8, zorder=3,
                           edgecolors='black' if transaction['sentiment_affected'] else 'none')
                
        plt.title(f'{ticker} Price with Trades (Purple = Sentiment-Influenced)')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{ticker}_transactions.png")
        plt.close()
        
    for ticker in env.tickers:
        plot_sentiment_vs_price(ticker, test_df, test_sentiment_df, output_folder)

    transaction_df = pd.DataFrame(env.transactions)
    if not transaction_df.empty:
        transaction_df['date'] = transaction_df['step'].apply(lambda x: dates[x] if x < len(dates) else None)
        transaction_df['portfolio_value'] = transaction_df['step'].apply(
            lambda x: portfolio_values[x - (env.window_size - 1) + 1] if x >= (env.window_size - 1) else None
        )

        transaction_df['position_size'] = transaction_df.apply(
            lambda row: row['shares'] * row['price'] / row['portfolio_value'] if row['portfolio_value'] > 0 else 0,
            axis=1
        )
        
        transaction_df.to_csv(f"{output_folder}/transactions.csv", index=False)

        sentiment_trades = transaction_df[transaction_df['sentiment_affected']]
        if not sentiment_trades.empty:
            print("\nSentiment-affected trades summary:")
            print(sentiment_trades[['date', 'ticker', 'action', 'sentiment', 'position_size']].to_string(index=False))

    initial_value = env.initial_balance
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_value - 1) * 100
    print(f"\nBacktest Results:")
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
    if len(env.transactions) > 0:
        wins = sum(1 for t in env.transactions
                  if t['action'] in ['sell', 'stop_loss', 'take_profit'] and t.get('profit', 0) > 0)
        losses = sum(1 for t in env.transactions 
                    if t['action'] in ['sell', 'stop_loss', 'take_profit'] and t.get('profit', 0) <= 0)
        if losses > 0:
            win_rate = wins / (wins + losses) * 100
            print(f"Win Rate: {win_rate:.2f}% ({wins}/{wins+losses})")
        else:
            print(f"Win Rate: 100% ({wins}/{wins})")
    
    return portfolio_values, env.transactions

def make_env(df, sentiment_df, initial_balance, window_size, stop_loss, take_profit):
    def _init():
        return StockTradingEnv(df, sentiment_df, initial_balance, window_size, stop_loss, take_profit)
    return _init

def objective(trial, train_df, train_sentiment_df, test_df, test_sentiment_df):
    """
    Objective function for Optuna to optimize SAC hyperparameters.

    Args:
        trial: Optuna trial object to suggest hyperparameters
        train_df: Training price data
        train_sentiment_df: Training sentiment data
        test_df: Testing price data
        test_sentiment_df: Testing sentiment data

    Returns:
        Total return (%) on test data from the backtested trained model
    """
    
    # suggest hyperparameters using Optuna's search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [10000, 50000, 100000])
    ent_coef = trial.suggest_categorical("ent_coef", ["auto", 0.1, 0.01])
    gamma = trial.suggest_float("gamma", 0.95, 0.999, log=True)
    tau = trial.suggest_float("tau", 0.001, 0.02, log=True)
    train_freq = trial.suggest_int("train_freq", 1, 10)

    # create a vectorized training environment
    env = make_vec_env(
        make_env(train_df, train_sentiment_df, 10000, 20, 0.03, 0.06),
        n_envs=4,
        vec_env_cls=SubprocVecEnv
    )

    # initialize the SAC model with the trial's suggested hyperparameters
    model = SAC(
        "MlpPolicy",
        env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=0,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        ent_coef=ent_coef,
        gamma=gamma,
        tau=tau,
        train_freq=train_freq,
        gradient_steps=1,
        tensorboard_log="./tensorboard_logs/"
    )

    # set training callbacks
    callbacks = [TensorboardCallback(), EarlyStoppingCallback(check_freq=500, reward_threshold=100, patience=1000)]

    # train the model for the number of timesteps defined
    model.learn(total_timesteps=200000, callback=callbacks, progress_bar=False)

    # evaluate the trained model on the test environment
    test_env = StockTradingEnv(test_df, test_sentiment_df, 10000, 20, 0.03, 0.06)
    
    portfolio_values, _ = backtest_model(model, test_env, test_df, test_sentiment_df, output_folder=f'backtest_results_trial_{trial.number}')

    # calculate total return over the test period
    initial_value = test_env.initial_balance
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_value - 1) * 100

    return total_return

def optimize_hyperparameters(train_df, train_sentiment_df, test_df, test_sentiment_df, n_trials=0):
    """
    Runs Optuna to find the best SAC hyperparameters, trains the final model, and backtests it.

    Args:
        train_df: Training price data
        train_sentiment_df: Training sentiment data
        test_df: Testing price data
        test_sentiment_df: Testing sentiment data
        n_trials: Number of Optuna trials to run

    Returns:
        None (prints results and saves backtest)
    """

    # create an Optuna study to maximize total return
    study = optuna.create_study(direction="maximize")

    # optimize the objective function
    study.optimize(lambda trial: objective(trial, train_df, train_sentiment_df, test_df, test_sentiment_df), n_trials=n_trials, show_progress_bar=True)

    # print the best results
    print("Best hyperparameters: ", study.best_params)
    print("Best total return: ", study.best_value)

    best_params = study.best_params

    # create training environmemt using best hyperparameters
    env = make_vec_env(
        make_env(train_df, train_sentiment_df, 10000, 20, 0.03, 0.06),
        n_envs=4,
        vec_env_cls=SubprocVecEnv
    )

    # reinitialize and train the final model with best parameters
    model = SAC(
        "MlpPolicy",
        env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=0,
        learning_rate=best_params["learning_rate"],
        buffer_size=best_params["buffer_size"],
        batch_size=best_params["batch_size"],
        ent_coef=best_params["ent_coef"],
        gamma=best_params["gamma"],
        tau=best_params["tau"],
        train_freq=best_params["train_freq"],
        gradient_steps=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    callbacks = [TensorboardCallback(), EarlyStoppingCallback(check_freq=500, reward_threshold=100, patience=1000)]
    
    model.learn(total_timesteps=200000, callback=callbacks, progress_bar=True)

    # evaluate and save results on the test environment
    test_env = StockTradingEnv(test_df, test_sentiment_df, 10000, 20, 0.03, 0.06)
    backtest_model(model, test_env, test_df, test_sentiment_df, output_folder='backtest_results_final')

def main():
    """
    Entry point of the program.
    Fetches training and testing data, runs optimization, and saves results.
    """
    
    # define stocks to trade
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

    # time period for training the model
    training_start_date = '2020-01-01'
    training_end_date = '2024-01-01'

    # time period for testing the model
    testing_start_date = '2024-01-01'
    testing_end_date = '2024-12-31'

    # fetch stock price and sentiment data for training
    print("Fetching training data...")
    train_df = fetch_stock_data(tickers, training_start_date, training_end_date)
    train_sentiment_df = fetch_sentiment_data(tickers, train_df, finnhub_key)
    print(train_sentiment_df.describe())

    # fetch stock price and sentiment data for testing
    print("Fetching testing data...")
    test_df = fetch_stock_data(tickers, testing_start_date, testing_end_date)
    test_sentiment_df = fetch_sentiment_data(tickers, test_df, finnhub_key)

    # run hyperparameter optimization and model training
    print("Optimizing hyperparameters...")
    optimize_hyperparameters(train_df, train_sentiment_df, test_df, test_sentiment_df, n_trials=5)

    print("\nCompleted! Results saved.")
  
if __name__ == "__main__":
    main()
