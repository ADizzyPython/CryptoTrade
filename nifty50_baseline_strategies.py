#!/usr/bin/env python3
"""
NIFTY50 Baseline Trading Strategies
Implementation of traditional trading strategies for performance comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, name: str, starting_capital: float = 1_000_000):
        self.name = name
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.position = 0  # Units of NIFTY50 held
        self.cash = starting_capital
        self.transaction_cost = 0.0011  # 0.11% total transaction cost
        self.trades = []
        self.portfolio_values = []
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, index: int) -> float:
        """
        Generate trading signal based on strategy logic
        
        Args:
            data: DataFrame with NIFTY50 data
            index: Current index in the data
            
        Returns:
            Signal between -1 (sell) and 1 (buy)
        """
        pass
    
    def execute_trade(self, signal: float, price: float, date: str) -> None:
        """
        Execute trade based on signal
        
        Args:
            signal: Trading signal (-1 to 1)
            price: Current price
            date: Trade date
        """
        if signal > 0:  # Buy signal
            # Calculate how much to buy
            available_cash = self.cash
            max_units = available_cash / price
            units_to_buy = max_units * signal
            
            if units_to_buy > 0:
                cost = units_to_buy * price
                transaction_cost = cost * self.transaction_cost
                total_cost = cost + transaction_cost
                
                if total_cost <= available_cash:
                    self.cash -= total_cost
                    self.position += units_to_buy
                    
                    self.trades.append({
                        'date': date,
                        'type': 'BUY',
                        'units': units_to_buy,
                        'price': price,
                        'cost': total_cost,
                        'cash_after': self.cash,
                        'position_after': self.position
                    })
                    
        elif signal < 0:  # Sell signal
            # Calculate how much to sell
            units_to_sell = self.position * abs(signal)
            
            if units_to_sell > 0:
                proceeds = units_to_sell * price
                transaction_cost = proceeds * self.transaction_cost
                net_proceeds = proceeds - transaction_cost
                
                self.cash += net_proceeds
                self.position -= units_to_sell
                
                self.trades.append({
                    'date': date,
                    'type': 'SELL',
                    'units': units_to_sell,
                    'price': price,
                    'proceeds': net_proceeds,
                    'cash_after': self.cash,
                    'position_after': self.position
                })
    
    def get_portfolio_value(self, price: float) -> float:
        """Get current portfolio value"""
        return self.cash + self.position * price
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with backtest results
        """
        self.trades = []
        self.portfolio_values = []
        
        for i in range(len(data)):
            row = data.iloc[i]
            price = row['open']
            date = str(row['timestamp'])
            
            # Generate signal
            signal = self.generate_signal(data, i)
            
            # Execute trade
            self.execute_trade(signal, price, date)
            
            # Record portfolio value
            portfolio_value = self.get_portfolio_value(price)
            self.portfolio_values.append({
                'date': date,
                'price': price,
                'cash': self.cash,
                'position': self.position,
                'portfolio_value': portfolio_value,
                'signal': signal
            })
        
        return self.calculate_performance()
    
    def calculate_performance(self) -> Dict:
        """Calculate performance metrics"""
        if not self.portfolio_values:
            return {}
        
        # Calculate returns
        final_value = self.portfolio_values[-1]['portfolio_value']
        total_return = (final_value / self.starting_capital) - 1
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(self.portfolio_values)):
            prev_value = self.portfolio_values[i-1]['portfolio_value']
            curr_value = self.portfolio_values[i]['portfolio_value']
            daily_return = (curr_value / prev_value) - 1
            daily_returns.append(daily_return)
        
        # Performance metrics
        avg_daily_return = np.mean(daily_returns) if daily_returns else 0
        volatility = np.std(daily_returns) if daily_returns else 0
        sharpe_ratio = (avg_daily_return / volatility) if volatility > 0 else 0
        
        # Calculate maximum drawdown
        max_drawdown = self.calculate_max_drawdown()
        
        return {
            'strategy': self.name,
            'total_return': total_return,
            'final_value': final_value,
            'avg_daily_return': avg_daily_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': self.calculate_win_rate(),
            'trades': self.trades,
            'portfolio_values': self.portfolio_values
        }
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.portfolio_values:
            return 0.0
        
        values = [pv['portfolio_value'] for pv in self.portfolio_values]
        peak = values[0]
        max_drawdown = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate (percentage of profitable trades)"""
        if not self.trades:
            return 0.0
        
        profitable_trades = 0
        for i in range(len(self.trades) - 1):
            if self.trades[i]['type'] == 'BUY' and i + 1 < len(self.trades):
                if self.trades[i + 1]['type'] == 'SELL':
                    buy_price = self.trades[i]['price']
                    sell_price = self.trades[i + 1]['price']
                    if sell_price > buy_price:
                        profitable_trades += 1
        
        return profitable_trades / (len(self.trades) / 2) if self.trades else 0.0

class BuyAndHoldStrategy(BaseStrategy):
    """Buy and Hold strategy - buy at start, hold until end"""
    
    def __init__(self, starting_capital: float = 1_000_000):
        super().__init__("Buy and Hold", starting_capital)
        self.bought = False
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> float:
        """Buy at the beginning and hold"""
        if not self.bought and index == 0:
            self.bought = True
            return 1.0  # Buy everything at start
        return 0.0  # Hold

class MovingAverageCrossoverStrategy(BaseStrategy):
    """Moving Average Crossover strategy"""
    
    def __init__(self, short_window: int = 5, long_window: int = 20, starting_capital: float = 1_000_000):
        super().__init__(f"MA Crossover ({short_window}/{long_window})", starting_capital)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> float:
        """Generate signal based on moving average crossover"""
        if index < self.long_window:
            return 0.0
        
        # Calculate moving averages
        short_ma = data.iloc[index - self.short_window + 1:index + 1]['close'].mean()
        long_ma = data.iloc[index - self.long_window + 1:index + 1]['close'].mean()
        
        # Generate signal
        if short_ma > long_ma:
            return 0.5  # Buy signal
        elif short_ma < long_ma:
            return -0.5  # Sell signal
        else:
            return 0.0  # Hold

class RSIStrategy(BaseStrategy):
    """RSI-based trading strategy"""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70, starting_capital: float = 1_000_000):
        super().__init__(f"RSI Strategy ({rsi_period}, {oversold}/{overbought})", starting_capital)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, data: pd.DataFrame, index: int) -> float:
        """Calculate RSI at given index"""
        if index < self.rsi_period:
            return 50.0  # Neutral RSI
        
        prices = data.iloc[index - self.rsi_period + 1:index + 1]['close']
        deltas = prices.diff()
        
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> float:
        """Generate signal based on RSI"""
        if index < self.rsi_period:
            return 0.0
        
        rsi = self.calculate_rsi(data, index)
        
        if rsi < self.oversold:
            return 0.6  # Buy signal (oversold)
        elif rsi > self.overbought:
            return -0.6  # Sell signal (overbought)
        else:
            return 0.0  # Hold

class MACDStrategy(BaseStrategy):
    """MACD-based trading strategy"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, starting_capital: float = 1_000_000):
        super().__init__(f"MACD Strategy ({fast_period}/{slow_period}/{signal_period})", starting_capital)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate_macd(self, data: pd.DataFrame, index: int) -> Tuple[float, float]:
        """Calculate MACD and signal line"""
        if index < self.slow_period:
            return 0.0, 0.0
        
        prices = data.iloc[:index + 1]['close']
        
        # Calculate EMAs
        ema_fast = prices.ewm(span=self.fast_period).mean().iloc[-1]
        ema_slow = prices.ewm(span=self.slow_period).mean().iloc[-1]
        
        # Calculate MACD
        macd = ema_fast - ema_slow
        
        # Calculate signal line (we need more data for this)
        if index < self.slow_period + self.signal_period:
            return macd, 0.0
        
        macd_series = []
        for i in range(self.slow_period, index + 1):
            prices_subset = data.iloc[:i + 1]['close']
            ema_fast_i = prices_subset.ewm(span=self.fast_period).mean().iloc[-1]
            ema_slow_i = prices_subset.ewm(span=self.slow_period).mean().iloc[-1]
            macd_series.append(ema_fast_i - ema_slow_i)
        
        macd_df = pd.Series(macd_series)
        signal_line = macd_df.ewm(span=self.signal_period).mean().iloc[-1]
        
        return macd, signal_line
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> float:
        """Generate signal based on MACD"""
        if index < self.slow_period + self.signal_period:
            return 0.0
        
        macd, signal_line = self.calculate_macd(data, index)
        
        if macd > signal_line:
            return 0.4  # Buy signal
        elif macd < signal_line:
            return -0.4  # Sell signal
        else:
            return 0.0  # Hold

class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands mean reversion strategy"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, starting_capital: float = 1_000_000):
        super().__init__(f"Bollinger Bands ({period}, {std_dev})", starting_capital)
        self.period = period
        self.std_dev = std_dev
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, index: int) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if index < self.period:
            price = data.iloc[index]['close']
            return price, price, price
        
        prices = data.iloc[index - self.period + 1:index + 1]['close']
        middle = prices.mean()
        std = prices.std()
        
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        
        return upper, middle, lower
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> float:
        """Generate signal based on Bollinger Bands"""
        if index < self.period:
            return 0.0
        
        current_price = data.iloc[index]['close']
        upper, middle, lower = self.calculate_bollinger_bands(data, index)
        
        if current_price < lower:
            return 0.5  # Buy signal (price below lower band)
        elif current_price > upper:
            return -0.5  # Sell signal (price above upper band)
        else:
            return 0.0  # Hold

class MomentumStrategy(BaseStrategy):
    """Momentum strategy based on price momentum"""
    
    def __init__(self, lookback_period: int = 10, starting_capital: float = 1_000_000):
        super().__init__(f"Momentum ({lookback_period})", starting_capital)
        self.lookback_period = lookback_period
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> float:
        """Generate signal based on momentum"""
        if index < self.lookback_period:
            return 0.0
        
        current_price = data.iloc[index]['close']
        past_price = data.iloc[index - self.lookback_period]['close']
        
        momentum = (current_price - past_price) / past_price
        
        # Generate signal based on momentum
        if momentum > 0.02:  # 2% positive momentum
            return 0.3  # Buy signal
        elif momentum < -0.02:  # 2% negative momentum
            return -0.3  # Sell signal
        else:
            return 0.0  # Hold

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy"""
    
    def __init__(self, lookback_period: int = 20, threshold: float = 0.02, starting_capital: float = 1_000_000):
        super().__init__(f"Mean Reversion ({lookback_period}, {threshold})", starting_capital)
        self.lookback_period = lookback_period
        self.threshold = threshold
    
    def generate_signal(self, data: pd.DataFrame, index: int) -> float:
        """Generate signal based on mean reversion"""
        if index < self.lookback_period:
            return 0.0
        
        current_price = data.iloc[index]['close']
        mean_price = data.iloc[index - self.lookback_period + 1:index + 1]['close'].mean()
        
        deviation = (current_price - mean_price) / mean_price
        
        if deviation < -self.threshold:
            return 0.4  # Buy signal (price below mean)
        elif deviation > self.threshold:
            return -0.4  # Sell signal (price above mean)
        else:
            return 0.0  # Hold

class BaselineStrategyManager:
    """Manager for running multiple baseline strategies"""
    
    def __init__(self, starting_capital: float = 1_000_000):
        self.starting_capital = starting_capital
        self.strategies = []
        self.results = {}
    
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add a strategy to the manager"""
        self.strategies.append(strategy)
    
    def add_all_strategies(self) -> None:
        """Add all baseline strategies"""
        self.strategies = [
            BuyAndHoldStrategy(self.starting_capital),
            MovingAverageCrossoverStrategy(5, 20, self.starting_capital),
            MovingAverageCrossoverStrategy(10, 50, self.starting_capital),
            RSIStrategy(14, 30, 70, self.starting_capital),
            MACDStrategy(12, 26, 9, self.starting_capital),
            BollingerBandsStrategy(20, 2.0, self.starting_capital),
            MomentumStrategy(10, self.starting_capital),
            MeanReversionStrategy(20, 0.02, self.starting_capital)
        ]
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """Run backtest for all strategies"""
        logger.info(f"Running backtest for {len(self.strategies)} strategies")
        
        self.results = {}
        
        for strategy in self.strategies:
            logger.info(f"Running backtest for {strategy.name}")
            
            # Reset strategy
            strategy.cash = strategy.starting_capital
            strategy.position = 0
            strategy.trades = []
            strategy.portfolio_values = []
            
            # Run backtest
            result = strategy.backtest(data)
            self.results[strategy.name] = result
            
            logger.info(f"Completed {strategy.name}: Return={result['total_return']:.2%}, Sharpe={result['sharpe_ratio']:.3f}")
        
        return self.results
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary of all strategies"""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for strategy_name, result in self.results.items():
            summary_data.append({
                'Strategy': strategy_name,
                'Total Return': result['total_return'],
                'Final Value': result['final_value'],
                'Sharpe Ratio': result['sharpe_ratio'],
                'Max Drawdown': result['max_drawdown'],
                'Volatility': result['volatility'],
                'Win Rate': result['win_rate'],
                'Total Trades': result['total_trades']
            })
        
        df = pd.DataFrame(summary_data)
        return df.sort_values('Total Return', ascending=False)
    
    def get_best_strategy(self) -> Tuple[str, Dict]:
        """Get the best performing strategy"""
        if not self.results:
            return None, None
        
        best_strategy = max(self.results.items(), key=lambda x: x[1]['total_return'])
        return best_strategy[0], best_strategy[1]
    
    def compare_with_llm_agent(self, llm_results: Dict) -> pd.DataFrame:
        """Compare baseline strategies with LLM agent results"""
        if not self.results:
            return pd.DataFrame()
        
        # Add LLM results to comparison
        comparison_data = []
        
        # Add baseline strategies
        for strategy_name, result in self.results.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Type': 'Baseline',
                'Total Return': result['total_return'],
                'Final Value': result['final_value'],
                'Sharpe Ratio': result['sharpe_ratio'],
                'Max Drawdown': result['max_drawdown'],
                'Volatility': result['volatility'],
                'Total Trades': result['total_trades']
            })
        
        # Add LLM agent results
        comparison_data.append({
            'Strategy': 'NIFTY50 LLM Agent',
            'Type': 'LLM-based',
            'Total Return': llm_results.get('total_return', 0),
            'Final Value': llm_results.get('final_value', 0),
            'Sharpe Ratio': llm_results.get('sharpe_ratio', 0),
            'Max Drawdown': llm_results.get('max_drawdown', 0),
            'Volatility': llm_results.get('volatility', 0),
            'Total Trades': llm_results.get('total_trades', 0)
        })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Total Return', ascending=False)

# Example usage
if __name__ == "__main__":
    # This would typically be run with actual NIFTY50 data
    
    # Create sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.02)  # Random walk
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    
    # Create strategy manager
    manager = BaselineStrategyManager(1_000_000)
    manager.add_all_strategies()
    
    # Run backtest
    results = manager.run_backtest(sample_data)
    
    # Get performance summary
    summary = manager.get_performance_summary()
    print("\nPerformance Summary:")
    print(summary.to_string(index=False))
    
    # Get best strategy
    best_name, best_result = manager.get_best_strategy()
    print(f"\nBest performing strategy: {best_name}")
    print(f"Total return: {best_result['total_return']:.2%}")
    print(f"Sharpe ratio: {best_result['sharpe_ratio']:.3f}")