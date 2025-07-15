#!/usr/bin/env python3
"""
NIFTY50 Strategy Comparison
Compare LLM-based NIFTY50 trading agent with baseline strategies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional

from nifty50_baseline_strategies import BaselineStrategyManager
from nifty50_agent import NIFTY50TradingAgent, create_nifty50_args, dummy_llm_function
from nifty50_config import NIFTY50Config, create_conservative_preset, create_aggressive_preset, create_balanced_preset
from nifty50_data_utils import NIFTY50DataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NIFTY50StrategyComparison:
    """Compare different trading strategies for NIFTY50"""
    
    def __init__(self, starting_capital: float = 1_000_000):
        self.starting_capital = starting_capital
        self.baseline_manager = BaselineStrategyManager(starting_capital)
        self.baseline_manager.add_all_strategies()
        
        # Results storage
        self.baseline_results = {}
        self.llm_results = {}
        self.comparison_results = {}
    
    def run_baseline_comparison(self, data: pd.DataFrame) -> Dict:
        """Run comparison of baseline strategies"""
        logger.info("Running baseline strategy comparison")
        
        # Run baseline strategies
        self.baseline_results = self.baseline_manager.run_backtest(data)
        
        # Get performance summary
        summary = self.baseline_manager.get_performance_summary()
        
        logger.info("Baseline strategy comparison completed")
        return self.baseline_results
    
    def run_llm_agent_comparison(self, data: pd.DataFrame, configs: Dict[str, NIFTY50Config]) -> Dict:
        """Run comparison of LLM agents with different configurations"""
        logger.info("Running LLM agent comparison")
        
        self.llm_results = {}
        
        # Extract date range from data
        start_date = data.iloc[0]['timestamp']
        end_date = data.iloc[-1]['timestamp']
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        
        for config_name, config in configs.items():
            logger.info(f"Running LLM agent with {config_name} configuration")
            
            try:
                # Create arguments
                args = create_nifty50_args(
                    starting_date=str(start_date),
                    ending_date=str(end_date),
                    config=config
                )
                
                # Initialize agent
                agent = NIFTY50TradingAgent(args, dummy_llm_function, config)
                
                # Run trading session
                results = agent.run_trading_session()
                
                # Store results
                self.llm_results[f"LLM Agent ({config_name})"] = results['performance_metrics']
                
                logger.info(f"Completed LLM agent with {config_name} configuration")
                
            except Exception as e:
                logger.error(f"Error running LLM agent with {config_name}: {e}")
                self.llm_results[f"LLM Agent ({config_name})"] = {
                    'total_return': 0,
                    'final_value': self.starting_capital,
                    'sharpe_ratio': 0,
                    'volatility': 0,
                    'total_trades': 0
                }
        
        return self.llm_results
    
    def create_comprehensive_comparison(self) -> pd.DataFrame:
        """Create comprehensive comparison of all strategies"""
        comparison_data = []
        
        # Add baseline strategies
        for strategy_name, result in self.baseline_results.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Type': 'Baseline',
                'Total Return': result['total_return'],
                'Final Value': result['final_value'],
                'Sharpe Ratio': result['sharpe_ratio'],
                'Max Drawdown': result['max_drawdown'],
                'Volatility': result['volatility'],
                'Total Trades': result['total_trades'],
                'Win Rate': result.get('win_rate', 0)
            })
        
        # Add LLM agent results
        for agent_name, result in self.llm_results.items():
            comparison_data.append({
                'Strategy': agent_name,
                'Type': 'LLM-based',
                'Total Return': result.get('total_return', 0),
                'Final Value': result.get('final_value', self.starting_capital),
                'Sharpe Ratio': result.get('sharpe_ratio', 0),
                'Max Drawdown': result.get('max_drawdown', 0),
                'Volatility': result.get('volatility', 0),
                'Total Trades': result.get('total_trades', 0),
                'Win Rate': 0  # Would need to calculate from trading history
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Total Return', ascending=False)
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        comparison_df = self.create_comprehensive_comparison()
        
        if comparison_df.empty:
            return "No results available for comparison"
        
        report = f"""
NIFTY50 STRATEGY COMPARISON REPORT
=================================

Total Strategies Compared: {len(comparison_df)}
- Baseline Strategies: {len(comparison_df[comparison_df['Type'] == 'Baseline'])}
- LLM-based Strategies: {len(comparison_df[comparison_df['Type'] == 'LLM-based'])}

TOP 5 PERFORMING STRATEGIES:
"""
        
        top_5 = comparison_df.head(5)
        for idx, row in top_5.iterrows():
            report += f"""
{row['Strategy']}:
  - Total Return: {row['Total Return']:.2%}
  - Final Value: ₹{row['Final Value']:,.2f}
  - Sharpe Ratio: {row['Sharpe Ratio']:.3f}
  - Max Drawdown: {row['Max Drawdown']:.2%}
  - Total Trades: {row['Total Trades']}
"""
        
        # Best performing by category
        best_baseline = comparison_df[comparison_df['Type'] == 'Baseline'].iloc[0] if not comparison_df[comparison_df['Type'] == 'Baseline'].empty else None
        best_llm = comparison_df[comparison_df['Type'] == 'LLM-based'].iloc[0] if not comparison_df[comparison_df['Type'] == 'LLM-based'].empty else None
        
        if best_baseline is not None:
            report += f"""
BEST BASELINE STRATEGY:
{best_baseline['Strategy']}
- Total Return: {best_baseline['Total Return']:.2%}
- Sharpe Ratio: {best_baseline['Sharpe Ratio']:.3f}
- Max Drawdown: {best_baseline['Max Drawdown']:.2%}
"""
        
        if best_llm is not None:
            report += f"""
BEST LLM-BASED STRATEGY:
{best_llm['Strategy']}
- Total Return: {best_llm['Total Return']:.2%}
- Sharpe Ratio: {best_llm['Sharpe Ratio']:.3f}
- Max Drawdown: {best_llm['Max Drawdown']:.2%}
"""
        
        # Performance statistics
        baseline_returns = comparison_df[comparison_df['Type'] == 'Baseline']['Total Return']
        llm_returns = comparison_df[comparison_df['Type'] == 'LLM-based']['Total Return']
        
        if not baseline_returns.empty and not llm_returns.empty:
            report += f"""
PERFORMANCE STATISTICS:
Baseline Strategies:
- Average Return: {baseline_returns.mean():.2%}
- Best Return: {baseline_returns.max():.2%}
- Worst Return: {baseline_returns.min():.2%}

LLM-based Strategies:
- Average Return: {llm_returns.mean():.2%}
- Best Return: {llm_returns.max():.2%}
- Worst Return: {llm_returns.min():.2%}
"""
        
        return report
    
    def create_performance_plots(self, save_path: str = "nifty50_performance_comparison.png"):
        """Create performance comparison plots"""
        comparison_df = self.create_comprehensive_comparison()
        
        if comparison_df.empty:
            logger.warning("No data available for plotting")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('NIFTY50 Strategy Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Total Return comparison
        ax1 = axes[0, 0]
        comparison_df_sorted = comparison_df.sort_values('Total Return', ascending=True)
        colors = ['blue' if x == 'Baseline' else 'red' for x in comparison_df_sorted['Type']]
        bars = ax1.barh(comparison_df_sorted['Strategy'], comparison_df_sorted['Total Return'], color=colors, alpha=0.7)
        ax1.set_xlabel('Total Return')
        ax1.set_title('Total Return by Strategy')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        # Add percentage labels
        for i, (bar, return_val) in enumerate(zip(bars, comparison_df_sorted['Total Return'])):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{return_val:.1%}', ha='left', va='center', fontsize=8)
        
        # 2. Sharpe Ratio comparison
        ax2 = axes[0, 1]
        comparison_df_sorted = comparison_df.sort_values('Sharpe Ratio', ascending=True)
        colors = ['blue' if x == 'Baseline' else 'red' for x in comparison_df_sorted['Type']]
        bars = ax2.barh(comparison_df_sorted['Strategy'], comparison_df_sorted['Sharpe Ratio'], color=colors, alpha=0.7)
        ax2.set_xlabel('Sharpe Ratio')
        ax2.set_title('Sharpe Ratio by Strategy')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        # Add value labels
        for i, (bar, sharpe_val) in enumerate(zip(bars, comparison_df_sorted['Sharpe Ratio'])):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{sharpe_val:.2f}', ha='left', va='center', fontsize=8)
        
        # 3. Max Drawdown comparison
        ax3 = axes[1, 0]
        comparison_df_sorted = comparison_df.sort_values('Max Drawdown', ascending=True)
        colors = ['blue' if x == 'Baseline' else 'red' for x in comparison_df_sorted['Type']]
        bars = ax3.barh(comparison_df_sorted['Strategy'], comparison_df_sorted['Max Drawdown'], color=colors, alpha=0.7)
        ax3.set_xlabel('Max Drawdown')
        ax3.set_title('Maximum Drawdown by Strategy')
        
        # Add percentage labels
        for i, (bar, drawdown_val) in enumerate(zip(bars, comparison_df_sorted['Max Drawdown'])):
            ax3.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{drawdown_val:.1%}', ha='left', va='center', fontsize=8)
        
        # 4. Risk-Return scatter plot
        ax4 = axes[1, 1]
        baseline_mask = comparison_df['Type'] == 'Baseline'
        llm_mask = comparison_df['Type'] == 'LLM-based'
        
        ax4.scatter(comparison_df[baseline_mask]['Volatility'], 
                   comparison_df[baseline_mask]['Total Return'], 
                   color='blue', alpha=0.7, label='Baseline', s=100)
        ax4.scatter(comparison_df[llm_mask]['Volatility'], 
                   comparison_df[llm_mask]['Total Return'], 
                   color='red', alpha=0.7, label='LLM-based', s=100)
        
        ax4.set_xlabel('Volatility')
        ax4.set_ylabel('Total Return')
        ax4.set_title('Risk-Return Profile')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add strategy names as annotations
        for idx, row in comparison_df.iterrows():
            ax4.annotate(row['Strategy'], 
                        (row['Volatility'], row['Total Return']), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance plots saved to {save_path}")
    
    def save_results(self, filepath: str = "nifty50_strategy_comparison.json"):
        """Save comparison results to JSON file"""
        results = {
            'baseline_results': self.baseline_results,
            'llm_results': self.llm_results,
            'comparison_summary': self.create_comprehensive_comparison().to_dict('records'),
            'metadata': {
                'starting_capital': self.starting_capital,
                'comparison_date': datetime.now().isoformat(),
                'total_strategies': len(self.baseline_results) + len(self.llm_results)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")

def run_comprehensive_comparison():
    """Run a comprehensive comparison of all strategies"""
    logger.info("Starting comprehensive NIFTY50 strategy comparison")
    
    # Create sample data for demonstration
    # In practice, you would load real NIFTY50 data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Create more realistic price data with some trend and volatility
    returns = np.random.normal(0.0008, 0.02, 100)  # Daily returns with slight positive bias
    prices = 20000 * np.exp(np.cumsum(returns))  # Starting at 20,000 (typical NIFTY50 level)
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, 100)),
        'low': prices * (1 - np.random.uniform(0, 0.02, 100)),
        'close': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
        'volume': np.random.randint(100000, 1000000, 100)
    })
    
    # Add some technical indicators
    sample_data['sma_5'] = sample_data['close'].rolling(5).mean()
    sample_data['sma_20'] = sample_data['close'].rolling(20).mean()
    sample_data['rsi'] = 50 + np.random.uniform(-20, 20, 100)  # Mock RSI
    sample_data['macd'] = np.random.uniform(-100, 100, 100)  # Mock MACD
    sample_data['macd_signal'] = sample_data['macd'].rolling(9).mean()
    sample_data['volatility'] = sample_data['close'].rolling(20).std()
    
    # Initialize comparison
    comparison = NIFTY50StrategyComparison(1_000_000)
    
    # Run baseline comparison
    baseline_results = comparison.run_baseline_comparison(sample_data)
    
    # Create different configurations for LLM agents
    configs = {
        'Conservative': create_conservative_preset(),
        'Balanced': create_balanced_preset(),
        'Aggressive': create_aggressive_preset()
    }
    
    # Run LLM agent comparison
    llm_results = comparison.run_llm_agent_comparison(sample_data, configs)
    
    # Generate comprehensive report
    report = comparison.generate_performance_report()
    print(report)
    
    # Create performance plots
    comparison.create_performance_plots()
    
    # Save results
    comparison.save_results()
    
    # Print comparison table
    comparison_df = comparison.create_comprehensive_comparison()
    print("\nDETAILED COMPARISON TABLE:")
    print("=" * 100)
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    logger.info("Comprehensive comparison completed")

if __name__ == "__main__":
    run_comprehensive_comparison()