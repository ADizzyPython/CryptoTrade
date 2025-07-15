import numpy as np
import pandas as pd
import re
import random
from datetime import datetime, timedelta
import json
from argparse import Namespace
import os
from nifty50_data_utils import NIFTY50DataManager
from nifty50_config import NIFTY50Config

class NIFTY50TradingEnv:
    def __init__(self, args):
        self.args = args
        
        # Initialize configuration
        self.config = getattr(args, 'config', NIFTY50Config())
        if isinstance(self.config, str):
            # If config is a file path, load it
            self.config = NIFTY50Config(self.config)
        
        self.data_manager = NIFTY50DataManager()
        
        # Set up data paths and parameters
        self.dataset = args.dataset.lower()  # 'nifty50'
        self.starting_date = args.starting_date
        self.ending_date = args.ending_date
        self.news_dir = getattr(args, 'news_dir', self.config.data.news_dir)
        
        # Load and process NIFTY50 data
        self.load_nifty50_data()
        
        # Initialize trading parameters from config
        self.starting_net_worth = self.config.trading.starting_capital
        self.starting_cash_ratio = (1 - self.config.trading.max_position_size + self.config.trading.min_cash_reserve) / 2
        self.total_transaction_cost = self.config.get_total_transaction_cost()
        self.total_steps = len(self.data)
        
    def load_nifty50_data(self):
        """Load and process NIFTY50 data"""
        # Update data if needed
        start_str = self.starting_date.strftime('%Y-%m-%d')
        end_str = self.ending_date.strftime('%Y-%m-%d')
        
        # Try to load existing data, otherwise fetch new data
        filename = f"nifty50_daily_{start_str}_{end_str}.csv"
        existing_data = self.data_manager.data_fetcher.load_existing_data(filename)
        
        if existing_data.empty:
            print(f"Fetching new NIFTY50 data for {start_str} to {end_str}")
            self.data_manager.update_nifty50_data(start_str, end_str)
            existing_data = self.data_manager.data_fetcher.load_existing_data(filename)
        
        # If still empty, try to use sample data
        if existing_data.empty:
            print("No real data available, trying sample data...")
            existing_data = self.data_manager.data_fetcher.load_existing_data("sample_data.csv")
            
        if existing_data.empty:
            raise ValueError(f"No NIFTY50 data available for {start_str} to {end_str}")
        
        # Process data
        df = existing_data.copy()
        df['date'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('date')
        
        # Add technical indicators if not present
        if 'sma_5' not in df.columns:
            df = self.data_manager.data_fetcher.calculate_technical_indicators(df)
        
        # Filter by date range
        self.data = df[(df['date'] >= self.starting_date) & (df['date'] <= self.ending_date)]
        
        # Set up transaction statistics (placeholder since NIFTY50 doesn't have crypto-style transaction stats)
        self.create_market_statistics()
        
    def create_market_statistics(self):
        """Create market statistics for NIFTY50 (volume, volatility, etc.)"""
        market_stats = []
        
        for _, row in self.data.iterrows():
            stats = {
                'date': row['date'],
                'volume': row['volume'],
                'volatility': row.get('volatility', 0),
                'market_cap': row.get('market_cap', 0),
                'high_low_spread': (row['high'] - row['low']) / row['open'],
                'daily_return': (row['close'] - row['open']) / row['open']
            }
            market_stats.append(stats)
        
        self.market_stats = pd.DataFrame(market_stats)
        
    def get_close_state(self, today, next_day, first_day=False):
        """Get the closing state for the current trading day"""
        next_open_price = next_day['open']
        close_net_worth = self.cash + self.nifty_units * next_open_price
        close_roi = close_net_worth / self.starting_net_worth - 1  # return on investment
        today_roi = close_net_worth / self.last_net_worth - 1
        self.last_net_worth = close_net_worth
        
        date = today['timestamp']
        parsed_time = pd.to_datetime(date)
        if first_day:
            parsed_time = parsed_time - timedelta(days=1)
        year, month, day = parsed_time.year, parsed_time.month, parsed_time.day
        
        # Technical indicators for next day
        ma5 = next_day.get('sma_5', 0)
        ma10 = next_day.get('sma_10', 0)
        ma15 = next_day.get('sma_15', 0)
        ma20 = next_day.get('sma_20', 0)
        
        # SMA crossover signals
        slma_signal = 'hold'
        if ma15 > ma20:
            slma_signal = 'sell'
        elif ma15 < ma20:
            slma_signal = 'buy'
        
        # Bollinger Bands
        sma = next_day.get('bb_middle', next_open_price)
        upper_band = next_day.get('bb_upper', next_open_price)
        lower_band = next_day.get('bb_lower', next_open_price)
        
        boll_signal = 'hold'
        if next_open_price < lower_band:
            boll_signal = 'buy'
        elif next_open_price > upper_band:
            boll_signal = 'sell'
        
        # MACD signals
        macd = next_day.get('macd', 0)
        macd_signal_line = next_day.get('macd_signal', 0)
        macd_signal = 'hold'
        if macd < macd_signal_line:
            macd_signal = 'buy'
        elif macd > macd_signal_line:
            macd_signal = 'sell'
        
        # RSI signals
        rsi = next_day.get('rsi', 50)
        rsi_signal = 'hold'
        if rsi < 30:
            rsi_signal = 'buy'  # Oversold
        elif rsi > 70:
            rsi_signal = 'sell'  # Overbought
        
        # Market statistics for today
        market_stat = self.market_stats[self.market_stats['date'] == parsed_time]
        if market_stat.empty:
            market_data = {
                'volume': 'N/A',
                'volatility': 'N/A',
                'market_cap': 'N/A',
                'high_low_spread': 'N/A',
                'daily_return': 'N/A'
            }
        else:
            market_data = {
                'volume': market_stat['volume'].values[0],
                'volatility': market_stat['volatility'].values[0],
                'market_cap': market_stat['market_cap'].values[0],
                'high_low_spread': market_stat['high_low_spread'].values[0],
                'daily_return': market_stat['daily_return'].values[0]
            }
        
        # Load news for today
        news_path = f"{self.news_dir}/{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.json"
        if not os.path.exists(news_path):
            news = 'N/A'
        else:
            try:
                with open(news_path, 'r', encoding='utf-8') as f:
                    loaded_news_data = json.load(f)
                    loaded_news = loaded_news_data.get('organic', [])
                
                seen_titles = set()  # remove duplicates
                news = []
                for loaded_item in loaded_news:
                    if loaded_item.get('title') not in seen_titles:
                        item = {
                            'title': loaded_item.get('title', ''),
                            'description': loaded_item.get('description', ''),
                            'date': loaded_item.get('date', ''),
                            'link': loaded_item.get('link', '')
                        }
                        # Clip content length
                        K = 3000
                        if len(item['description']) > K:
                            item['description'] = item['description'][:K] + '...'
                        news.append(item)
                        seen_titles.add(item['title'])
                        
                        # Limit to top 10 news items
                        if len(news) >= 10:
                            break
            except Exception as e:
                print(f"Error loading news for {date}: {e}")
                news = 'N/A'
        
        close_state = {
            'cash': self.cash,
            'nifty_units': self.nifty_units,
            'open': next_open_price,
            'net_worth': close_net_worth,
            'roi': close_roi,
            'today_roi': today_roi,
            'technical': {
                'sma_crossover_signal': slma_signal,
                'macd_signal': macd_signal,
                'bollinger_bands_signal': boll_signal,
                'rsi_signal': rsi_signal,
                'rsi_value': rsi,
                'macd_value': macd,
                'sma_5': ma5,
                'sma_20': ma20
            },
            'market_stats': market_data,
            'news': news,
            'date': date,
        }
        return close_state
    
    def reset(self):
        """Reset the trading environment"""
        self.current_step = 0
        today = self.data.iloc[self.current_step]
        next_day = self.data.iloc[self.current_step + 1] if self.current_step + 1 < len(self.data) else today
        
        # Initialize portfolio
        self.cash = self.starting_net_worth * self.starting_cash_ratio
        nifty_investment = self.starting_net_worth * (1 - self.starting_cash_ratio)
        self.nifty_units = nifty_investment / today['open']
        
        self.last_net_worth = self.starting_net_worth
        self.action_history = []
        self.portfolio_history = []
        
        # Get initial state
        close_state = self.get_close_state(today, next_day, first_day=True)
        return close_state
    
    def step(self, action):
        """Execute a trading action"""
        if self.current_step >= self.total_steps - 1:
            return None, 0, True, {}
        
        today = self.data.iloc[self.current_step]
        next_day = self.data.iloc[self.current_step + 1]
        
        # Execute trade
        reward = self.execute_trade(action, today)
        
        # Move to next step
        self.current_step += 1
        
        # Get new state
        done = self.current_step >= self.total_steps - 1
        next_state = self.get_close_state(today, next_day) if not done else None
        
        # Store action history
        self.action_history.append({
            'date': today['timestamp'],
            'action': action,
            'price': today['open'],
            'cash': self.cash,
            'nifty_units': self.nifty_units,
            'net_worth': self.cash + self.nifty_units * today['open']
        })
        
        return next_state, reward, done, {}
    
    def execute_trade(self, action, today):
        """Execute a trade based on the action"""
        current_price = today['open']
        
        # Action is in range [-1, 1]
        # -1: sell all, 0: hold, 1: buy all available cash
        action = np.clip(action, -1, 1)
        
        if action > 0:  # Buy
            # Calculate how much to buy
            available_cash = self.cash
            max_units_to_buy = available_cash / current_price
            units_to_buy = max_units_to_buy * action
            
            if units_to_buy > 0:
                transaction_cost = units_to_buy * current_price * self.total_transaction_cost
                total_cost = units_to_buy * current_price + transaction_cost
                
                if total_cost <= available_cash:
                    self.cash -= total_cost
                    self.nifty_units += units_to_buy
                    reward = 0.01 * action  # Small positive reward for buying
                else:
                    reward = -0.01  # Penalty for invalid trade
            else:
                reward = 0
                
        elif action < 0:  # Sell
            # Calculate how much to sell
            units_to_sell = self.nifty_units * abs(action)
            
            if units_to_sell > 0:
                transaction_cost = units_to_sell * current_price * self.total_transaction_cost
                proceeds = units_to_sell * current_price - transaction_cost
                
                self.cash += proceeds
                self.nifty_units -= units_to_sell
                reward = 0.01 * abs(action)  # Small positive reward for selling
            else:
                reward = 0
                
        else:  # Hold
            reward = 0
        
        return reward
    
    def get_current_state(self):
        """Get the current state of the environment"""
        if self.current_step >= len(self.data):
            return None
        
        today = self.data.iloc[self.current_step]
        next_day = self.data.iloc[self.current_step + 1] if self.current_step + 1 < len(self.data) else today
        
        return self.get_close_state(today, next_day)
    
    def get_portfolio_value(self):
        """Get current portfolio value"""
        if self.current_step >= len(self.data):
            return self.cash
        
        current_price = self.data.iloc[self.current_step]['open']
        return self.cash + self.nifty_units * current_price
    
    def get_action_history(self):
        """Get the history of actions taken"""
        return self.action_history
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        if not self.action_history:
            return {}
        
        final_value = self.get_portfolio_value()
        total_return = (final_value / self.starting_net_worth) - 1
        
        # Calculate other metrics
        daily_returns = []
        for i in range(1, len(self.action_history)):
            prev_value = self.action_history[i-1]['net_worth']
            curr_value = self.action_history[i]['net_worth']
            daily_return = (curr_value / prev_value) - 1
            daily_returns.append(daily_return)
        
        if daily_returns:
            avg_daily_return = np.mean(daily_returns)
            volatility = np.std(daily_returns)
            sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0
        else:
            avg_daily_return = 0
            volatility = 0
            sharpe_ratio = 0
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'avg_daily_return': avg_daily_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len([a for a in self.action_history if a['action'] != 0])
        }