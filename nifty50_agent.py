#!/usr/bin/env python3
"""
NIFTY50 Trading Agent
Main trading agent implementation for NIFTY50 index trading with multi-analyst architecture
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from argparse import Namespace

from nifty50_env import NIFTY50TradingEnv
from nifty50_analysts import NIFTY50AnalystManager
from nifty50_data_utils import NIFTY50DataManager
from nifty50_config import NIFTY50Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NIFTY50TradingAgent:
    """Main NIFTY50 trading agent with multi-analyst architecture"""
    
    def __init__(self, args: Namespace, llm_function: callable, config: NIFTY50Config = None):
        """
        Initialize the NIFTY50 trading agent
        
        Args:
            args: Configuration arguments
            llm_function: Function to call language model
            config: NIFTY50Config object (optional)
        """
        self.args = args
        self.llm_function = llm_function
        
        # Initialize configuration
        self.config = config or getattr(args, 'config', NIFTY50Config())
        if isinstance(self.config, str):
            self.config = NIFTY50Config(self.config)
        
        # Set model parameters from config
        self.model = getattr(args, 'model', self.config.model.model_name)
        self.seed = getattr(args, 'seed', self.config.model.seed)
        
        # Pass config to args for environment
        args.config = self.config
        
        # Initialize components
        self.env = NIFTY50TradingEnv(args)
        self.analyst_manager = NIFTY50AnalystManager()
        
        # Configuration flags from config
        self.use_tech = getattr(args, 'use_tech', self.config.analyst.use_technical)
        self.use_news = getattr(args, 'use_news', self.config.analyst.use_news)
        self.use_reflection = getattr(args, 'use_reflection', self.config.analyst.use_reflection)
        self.price_window = getattr(args, 'price_window', self.config.trading.price_window)
        self.reflection_window = getattr(args, 'reflection_window', self.config.trading.reflection_window)
        
        # Trading history and results
        self.trading_history = []
        self.analyst_reports = []
        self.performance_metrics = {}
        
        # Initialize environment
        self.current_state = self.env.reset()
        
        logger.info(f"Initialized NIFTY50 Trading Agent")
        logger.info(f"Trading period: {args.starting_date} to {args.ending_date}")
        logger.info(f"Use technical analysis: {self.use_tech}")
        logger.info(f"Use news analysis: {self.use_news}")
        logger.info(f"Use reflection: {self.use_reflection}")
        
    def run_trading_session(self) -> Dict[str, Any]:
        """
        Run a complete trading session
        
        Returns:
            Dictionary containing trading results and performance metrics
        """
        logger.info("Starting NIFTY50 trading session...")
        
        step = 0
        total_steps = self.env.total_steps
        
        while self.current_state is not None:
            step += 1
            logger.info(f"Trading step {step}/{total_steps - 1}")
            
            try:
                # Get current market state
                state = self.prepare_state_for_analysis(self.current_state)
                
                # Get recent trading history for reflection
                recent_history = self.get_recent_trading_history()
                
                # Run analyst analysis and get trading decision
                reasoning, action = self.analyst_manager.analyze_and_trade(
                    state, recent_history, self.llm_function, self.model, self.seed
                )
                
                # Get individual analyst reports for logging
                reports = self.analyst_manager.get_analyst_reports(
                    state, recent_history, self.llm_function, self.model, self.seed
                )
                
                # Store analyst reports
                self.analyst_reports.append({
                    'step': step,
                    'date': state['date'],
                    'reports': reports,
                    'reasoning': reasoning,
                    'action': action
                })
                
                # Execute trade
                next_state, reward, done, info = self.env.step(action)
                
                # Log trading action
                logger.info(f"Date: {state['date']}")
                logger.info(f"Action: {action:.4f}")
                logger.info(f"Reasoning: {reasoning}")
                logger.info(f"Portfolio value: ₹{state['net_worth']:.2f}")
                logger.info(f"ROI: {state['roi']:.2%}")
                
                # Update state
                self.current_state = next_state
                
                if done:
                    logger.info("Trading session completed!")
                    break
                    
            except Exception as e:
                logger.error(f"Error in trading step {step}: {e}")
                # Skip this step and continue
                next_state, reward, done, info = self.env.step(0)  # Hold position
                self.current_state = next_state
                
                if done:
                    break
        
        # Calculate final performance metrics
        self.performance_metrics = self.env.get_performance_metrics()
        self.trading_history = self.env.get_action_history()
        
        # Generate results summary
        results = {
            'performance_metrics': self.performance_metrics,
            'trading_history': self.trading_history,
            'analyst_reports': self.analyst_reports,
            'configuration': {
                'use_tech': self.use_tech,
                'use_news': self.use_news,
                'use_reflection': self.use_reflection,
                'price_window': self.price_window,
                'reflection_window': self.reflection_window,
                'model': self.model,
                'seed': self.seed
            }
        }
        
        logger.info("Trading session results:")
        logger.info(f"Final portfolio value: ₹{self.performance_metrics['final_value']:.2f}")
        logger.info(f"Total return: {self.performance_metrics['total_return']:.2%}")
        logger.info(f"Sharpe ratio: {self.performance_metrics['sharpe_ratio']:.4f}")
        logger.info(f"Total trades: {self.performance_metrics['total_trades']}")
        
        return results
    
    def prepare_state_for_analysis(self, state: Dict) -> Dict:
        """
        Prepare state data for analyst analysis
        
        Args:
            state: Raw state from environment
            
        Returns:
            Processed state ready for analysis
        """
        # If technical analysis is disabled, remove technical indicators
        if not self.use_tech:
            state = state.copy()
            state['technical'] = {}
            state['market_stats'] = {}
        
        # If news analysis is disabled, remove news data
        if not self.use_news:
            state = state.copy()
            state['news'] = 'N/A'
        
        return state
    
    def get_recent_trading_history(self) -> List[Dict]:
        """
        Get recent trading history for reflection analysis
        
        Returns:
            List of recent trading actions
        """
        if not self.use_reflection:
            return []
        
        # Get recent trading history based on reflection window
        recent_history = self.env.get_action_history()
        
        if len(recent_history) > self.reflection_window:
            recent_history = recent_history[-self.reflection_window:]
        
        return recent_history
    
    def save_results(self, results: Dict, output_path: str = None) -> None:
        """
        Save trading results to file
        
        Args:
            results: Trading results dictionary
            output_path: Path to save results (optional)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"nifty50_trading_results_{timestamp}.json"
        
        try:
            # Convert numpy types to Python types for JSON serialization
            results_json = json.dumps(results, indent=2, default=str)
            
            with open(output_path, 'w') as f:
                f.write(results_json)
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def load_results(self, input_path: str) -> Dict:
        """
        Load trading results from file
        
        Args:
            input_path: Path to load results from
            
        Returns:
            Trading results dictionary
        """
        try:
            with open(input_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Results loaded from {input_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return {}
    
    def generate_performance_report(self, results: Dict) -> str:
        """
        Generate a detailed performance report
        
        Args:
            results: Trading results dictionary
            
        Returns:
            Formatted performance report
        """
        metrics = results['performance_metrics']
        config = results['configuration']
        
        report = f"""
NIFTY50 Trading Performance Report
=================================

Configuration:
- Technical Analysis: {config['use_tech']}
- News Analysis: {config['use_news']}
- Reflection Analysis: {config['use_reflection']}
- Model: {config['model']}
- Price Window: {config['price_window']} days
- Reflection Window: {config['reflection_window']} trades

Performance Metrics:
- Starting Capital: ₹{self.config.trading.starting_capital:,.2f}
- Final Portfolio Value: ₹{metrics['final_value']:,.2f}
- Total Return: {metrics['total_return']:.2%}
- Average Daily Return: {metrics['avg_daily_return']:.4%}
- Volatility: {metrics['volatility']:.4f}
- Sharpe Ratio: {metrics['sharpe_ratio']:.4f}
- Total Trades: {metrics['total_trades']}

Trading Summary:
- Total Trading Days: {len(results['trading_history'])}
- Active Trading Days: {metrics['total_trades']}
- Average Trades per Day: {metrics['total_trades'] / len(results['trading_history']):.2f}
"""
        
        return report
    
    def analyze_trading_patterns(self, results: Dict) -> Dict:
        """
        Analyze trading patterns and behavior
        
        Args:
            results: Trading results dictionary
            
        Returns:
            Dictionary with pattern analysis
        """
        trading_history = results['trading_history']
        
        if not trading_history:
            return {}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(trading_history)
        
        # Calculate basic statistics
        actions = df['action'].values
        buy_actions = actions[actions > 0]
        sell_actions = actions[actions < 0]
        hold_actions = actions[actions == 0]
        
        # Analyze returns
        df['daily_return'] = df['net_worth'].pct_change()
        positive_returns = df['daily_return'][df['daily_return'] > 0]
        negative_returns = df['daily_return'][df['daily_return'] < 0]
        
        # Calculate win rate
        win_rate = len(positive_returns) / len(df) if len(df) > 0 else 0
        
        analysis = {
            'trading_behavior': {
                'buy_actions': len(buy_actions),
                'sell_actions': len(sell_actions),
                'hold_actions': len(hold_actions),
                'avg_buy_intensity': np.mean(buy_actions) if len(buy_actions) > 0 else 0,
                'avg_sell_intensity': np.mean(sell_actions) if len(sell_actions) > 0 else 0
            },
            'return_analysis': {
                'win_rate': win_rate,
                'avg_positive_return': np.mean(positive_returns) if len(positive_returns) > 0 else 0,
                'avg_negative_return': np.mean(negative_returns) if len(negative_returns) > 0 else 0,
                'best_day_return': df['daily_return'].max(),
                'worst_day_return': df['daily_return'].min()
            },
            'portfolio_evolution': {
                'max_portfolio_value': df['net_worth'].max(),
                'min_portfolio_value': df['net_worth'].min(),
                'max_drawdown': self.calculate_max_drawdown(df['net_worth'].values)
            }
        }
        
        return analysis
    
    def calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            portfolio_values: Array of portfolio values
            
        Returns:
            Maximum drawdown as a percentage
        """
        if len(portfolio_values) < 2:
            return 0.0
        
        peak = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown

# Utility functions for running the agent
def create_nifty50_args(starting_date: str, ending_date: str, config: NIFTY50Config = None, **kwargs) -> Namespace:
    """
    Create arguments namespace for NIFTY50 trading agent
    
    Args:
        starting_date: Start date in YYYY-MM-DD format
        ending_date: End date in YYYY-MM-DD format
        config: NIFTY50Config object (optional)
        **kwargs: Additional configuration parameters
        
    Returns:
        Namespace with configuration
    """
    if config is None:
        config = NIFTY50Config()
    
    args = Namespace()
    args.dataset = 'nifty50'
    args.starting_date = pd.to_datetime(starting_date)
    args.ending_date = pd.to_datetime(ending_date)
    args.config = config
    args.news_dir = kwargs.get('news_dir', config.data.news_dir)
    args.use_tech = kwargs.get('use_tech', config.analyst.use_technical)
    args.use_news = kwargs.get('use_news', config.analyst.use_news)
    args.use_reflection = kwargs.get('use_reflection', config.analyst.use_reflection)
    args.price_window = kwargs.get('price_window', config.trading.price_window)
    args.reflection_window = kwargs.get('reflection_window', config.trading.reflection_window)
    args.model = kwargs.get('model', config.model.model_name)
    args.seed = kwargs.get('seed', config.model.seed)
    
    return args

def dummy_llm_function(prompt: str, model: str, seed: int) -> str:
    """
    Dummy LLM function for testing
    
    Args:
        prompt: Input prompt
        model: Model name
        seed: Random seed
        
    Returns:
        Dummy response
    """
    return f"This is a dummy response for model {model} with seed {seed}. Technical analysis shows bullish trend."

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = NIFTY50Config()
    config.backtest.starting_date = "2024-01-01"
    config.backtest.ending_date = "2024-12-31"
    
    args = create_nifty50_args(
        starting_date=config.backtest.starting_date,
        ending_date=config.backtest.ending_date,
        config=config
    )
    
    # Initialize agent with dummy LLM
    agent = NIFTY50TradingAgent(args, dummy_llm_function, config)
    
    # Run trading session
    results = agent.run_trading_session()
    
    # Save results
    agent.save_results(results, "nifty50_test_results.json")
    
    # Generate performance report
    report = agent.generate_performance_report(results)
    print(report)
    
    # Analyze trading patterns
    patterns = agent.analyze_trading_patterns(results)
    print("\nTrading Pattern Analysis:")
    print(json.dumps(patterns, indent=2, default=str))