#!/usr/bin/env python3
"""
NIFTY50 2-Month Simple Baseline Comparison
Run comparison without plotting dependencies
"""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nifty50_config import NIFTY50Config, create_conservative_preset, create_aggressive_preset, create_balanced_preset
from nifty50_baseline_strategies import BaselineStrategyManager
from nifty50_agent import NIFTY50TradingAgent, create_nifty50_args
from utils import get_chat

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleLLMFunction:
    """Simple LLM function wrapper"""
    
    def __init__(self):
        self.call_count = 0
        
    def __call__(self, prompt: str, model: str, seed: int) -> str:
        self.call_count += 1
        try:
            response = get_chat(prompt, model, seed, temperature=0.0, max_tokens=512)
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error: Unable to get response. Using conservative hold strategy."

def create_extended_sample_data(start_date: str, end_date: str):
    """Create extended sample data for 2 months"""
    
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate business days (trading days)
    date_range = pd.bdate_range(start=start, end=end, freq='D')
    
    logger.info(f"Creating sample data for {len(date_range)} trading days")
    
    # Create realistic NIFTY50 data
    np.random.seed(42)
    
    # Starting price around 22,000 (typical NIFTY50 level in 2024)
    starting_price = 22000
    
    # Generate realistic returns (Indian market characteristics)
    daily_returns = np.random.normal(0.0005, 0.015, len(date_range))  # 0.05% daily return, 1.5% volatility
    
    # Add some trend and seasonality
    trend = np.linspace(0, 0.1, len(date_range))  # 10% upward trend over period
    seasonal = 0.02 * np.sin(np.linspace(0, 4*np.pi, len(date_range)))  # Some cyclical movement
    
    # Combine effects
    cumulative_returns = np.cumsum(daily_returns + trend/len(date_range) + seasonal/len(date_range))
    prices = starting_price * np.exp(cumulative_returns)
    
    # Add some realistic market events (sudden drops/rallies)
    market_events = np.random.choice(len(date_range), size=3, replace=False)
    for event in market_events:
        event_magnitude = np.random.uniform(-0.03, 0.03)  # ±3% market event
        prices[event:] *= (1 + event_magnitude)
    
    # Create OHLC data
    opens = prices
    highs = prices * (1 + np.random.uniform(0.001, 0.02, len(date_range)))
    lows = prices * (1 - np.random.uniform(0.001, 0.02, len(date_range)))
    closes = prices * (1 + np.random.uniform(-0.01, 0.01, len(date_range)))
    
    # Volumes (typical NIFTY50 trading volumes)
    volumes = np.random.randint(500000, 5000000, len(date_range))
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'timestamp': date_range,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'name': '^NSEI'
    })
    
    # Add technical indicators
    sample_data['sma_5'] = sample_data['close'].rolling(5).mean()
    sample_data['sma_10'] = sample_data['close'].rolling(10).mean()
    sample_data['sma_20'] = sample_data['close'].rolling(20).mean()
    sample_data['sma_50'] = sample_data['close'].rolling(50).mean()
    
    # MACD
    ema_12 = sample_data['close'].ewm(span=12).mean()
    ema_26 = sample_data['close'].ewm(span=26).mean()
    sample_data['macd'] = ema_12 - ema_26
    sample_data['macd_signal'] = sample_data['macd'].ewm(span=9).mean()
    
    # RSI
    delta = sample_data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    sample_data['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sample_data['bb_middle'] = sample_data['close'].rolling(20).mean()
    bb_std = sample_data['close'].rolling(20).std()
    sample_data['bb_upper'] = sample_data['bb_middle'] + (2 * bb_std)
    sample_data['bb_lower'] = sample_data['bb_middle'] - (2 * bb_std)
    
    # Volatility
    sample_data['volatility'] = sample_data['close'].rolling(20).std()
    
    logger.info(f"Sample data created: {len(sample_data)} rows")
    logger.info(f"Price range: ₹{sample_data['close'].min():.2f} to ₹{sample_data['close'].max():.2f}")
    logger.info(f"Total return: {((sample_data['close'].iloc[-1] / sample_data['close'].iloc[0]) - 1) * 100:.2f}%")
    
    return sample_data

def run_baseline_strategies(sample_data: pd.DataFrame, starting_capital: int = 1000000):
    """Run baseline strategies"""
    
    print(f"\n{'='*60}")
    print(f"📊 RUNNING BASELINE STRATEGIES")
    print(f"{'='*60}")
    
    # Initialize baseline strategy manager
    baseline_manager = BaselineStrategyManager(starting_capital)
    baseline_manager.add_all_strategies()
    
    # Run backtest
    baseline_results = baseline_manager.run_backtest(sample_data)
    
    # Get performance summary
    summary = baseline_manager.get_performance_summary()
    
    print(f"\n📋 BASELINE STRATEGY RESULTS:")
    print(f"{'='*60}")
    
    for _, row in summary.iterrows():
        print(f"{row['Strategy']:<25} | Return: {row['Total Return']:>8.2%} | Sharpe: {row['Sharpe Ratio']:>6.3f} | Trades: {row['Total Trades']:>3}")
    
    return baseline_results, summary

def run_llm_strategies(sample_data: pd.DataFrame, start_date: str, end_date: str, starting_capital: int = 1000000):
    """Run LLM-based strategies"""
    
    print(f"\n{'='*60}")
    print(f"🤖 RUNNING LLM STRATEGIES (GEMINI)")
    print(f"{'='*60}")
    
    llm_function = SimpleLLMFunction()
    llm_results = {}
    
    # Create different configurations
    configs = {
        'Conservative': create_conservative_preset(),
        'Balanced': create_balanced_preset(),
        'Aggressive': create_aggressive_preset()
    }
    
    # Update all configs to use Gemini
    for config in configs.values():
        config.model.model_name = "google/gemini-2.5-flash-preview-05-20"
        config.trading.starting_capital = starting_capital
    
    # Technical only configuration
    tech_config = create_balanced_preset()
    tech_config.model.model_name = "google/gemini-2.5-flash-preview-05-20"
    tech_config.analyst.use_technical = True
    tech_config.analyst.use_news = False
    tech_config.analyst.use_reflection = True
    tech_config.analyst.technical_weight = 0.7
    tech_config.analyst.reflection_weight = 0.3
    tech_config.trading.starting_capital = starting_capital
    configs['Technical Only'] = tech_config
    
    # News only configuration
    news_config = create_balanced_preset()
    news_config.model.model_name = "google/gemini-2.5-flash-preview-05-20"
    news_config.analyst.use_technical = False
    news_config.analyst.use_news = True
    news_config.analyst.use_reflection = True
    news_config.analyst.news_weight = 0.7
    news_config.analyst.reflection_weight = 0.3
    news_config.trading.starting_capital = starting_capital
    configs['News Only'] = news_config
    
    # Run each LLM configuration
    for config_name, config in configs.items():
        print(f"\n🔄 Running {config_name} configuration...")
        
        try:
            # Create arguments
            args = create_nifty50_args(
                starting_date=start_date,
                ending_date=end_date,
                config=config
            )
            
            # Save sample data with proper filename
            sample_data.to_csv('data/nifty50/sample_data.csv', index=False)
            
            # Initialize agent
            agent = NIFTY50TradingAgent(args, llm_function, config)
            
            # Run trading session
            results = agent.run_trading_session()
            
            # Store results
            llm_results[config_name] = results['performance_metrics']
            
            print(f"✅ {config_name}: {results['performance_metrics']['total_return']:.2%} return")
            
        except Exception as e:
            print(f"❌ {config_name} failed: {e}")
            llm_results[config_name] = {
                'total_return': 0,
                'final_value': starting_capital,
                'sharpe_ratio': 0,
                'volatility': 0,
                'total_trades': 0
            }
    
    print(f"\n📋 LLM STRATEGY RESULTS:")
    print(f"{'='*60}")
    
    for strategy, result in llm_results.items():
        print(f"{strategy:<25} | Return: {result['total_return']:>8.2%} | Sharpe: {result['sharpe_ratio']:>6.3f} | Trades: {result['total_trades']:>3}")
    
    print(f"\n🤖 Total AI Calls: {llm_function.call_count}")
    
    return llm_results

def create_comparison_table(baseline_summary: pd.DataFrame, llm_results: dict):
    """Create comprehensive comparison table"""
    
    comparison_data = []
    
    # Add baseline strategies
    for _, row in baseline_summary.iterrows():
        comparison_data.append({
            'Strategy': row['Strategy'],
            'Type': 'Baseline',
            'Total Return': row['Total Return'],
            'Final Value': row['Final Value'],
            'Sharpe Ratio': row['Sharpe Ratio'],
            'Max Drawdown': row['Max Drawdown'],
            'Volatility': row['Volatility'],
            'Total Trades': row['Total Trades']
        })
    
    # Add LLM strategies
    for strategy, result in llm_results.items():
        comparison_data.append({
            'Strategy': f"LLM {strategy}",
            'Type': 'LLM-based',
            'Total Return': result['total_return'],
            'Final Value': result['final_value'],
            'Sharpe Ratio': result['sharpe_ratio'],
            'Max Drawdown': result.get('max_drawdown', 0),
            'Volatility': result['volatility'],
            'Total Trades': result['total_trades']
        })
    
    df = pd.DataFrame(comparison_data)
    return df.sort_values('Total Return', ascending=False)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='NIFTY50 2-Month Simple Baseline Comparison')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-02-29', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=int, default=1000000, help='Starting capital in INR')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🚀 NIFTY50 2-MONTH BASELINE COMPARISON")
    print(f"{'='*80}")
    print(f"📅 Period: {args.start_date} to {args.end_date}")
    print(f"💰 Starting Capital: ₹{args.capital:,.2f}")
    print(f"🤖 Model: google/gemini-2.5-flash-preview-05-20")
    
    try:
        # Create sample data
        sample_data = create_extended_sample_data(args.start_date, args.end_date)
        
        # Save sample data
        os.makedirs('data/nifty50', exist_ok=True)
        sample_data.to_csv('data/nifty50/2month_sample_data.csv', index=False)
        
        # Run baseline strategies
        baseline_results, baseline_summary = run_baseline_strategies(sample_data, args.capital)
        
        # Run LLM strategies
        llm_results = run_llm_strategies(sample_data, args.start_date, args.end_date, args.capital)
        
        # Create comparison table
        comparison_df = create_comparison_table(baseline_summary, llm_results)
        
        print(f"\n{'='*80}")
        print(f"📊 COMPREHENSIVE COMPARISON TABLE")
        print(f"{'='*80}")
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        
        print(comparison_df.round(4).to_string(index=False))
        
        # Summary statistics
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY STATISTICS")
        print(f"{'='*80}")
        
        baseline_returns = comparison_df[comparison_df['Type'] == 'Baseline']['Total Return']
        llm_returns = comparison_df[comparison_df['Type'] == 'LLM-based']['Total Return']
        
        print(f"Baseline Strategies ({len(baseline_returns)} strategies):")
        print(f"  • Best Return: {baseline_returns.max():.2%}")
        print(f"  • Worst Return: {baseline_returns.min():.2%}")
        print(f"  • Average Return: {baseline_returns.mean():.2%}")
        
        print(f"\nLLM-based Strategies ({len(llm_returns)} strategies):")
        print(f"  • Best Return: {llm_returns.max():.2%}")
        print(f"  • Worst Return: {llm_returns.min():.2%}")
        print(f"  • Average Return: {llm_returns.mean():.2%}")
        
        # Market benchmark
        market_return = ((sample_data['close'].iloc[-1] / sample_data['close'].iloc[0]) - 1)
        print(f"\nMarket Benchmark (Buy & Hold):")
        print(f"  • Market Return: {market_return:.2%}")
        
        # Top performers
        print(f"\n🏆 TOP 5 PERFORMERS:")
        top_5 = comparison_df.head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. {row['Strategy']:<25} | {row['Total Return']:>8.2%} | {row['Type']}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"2month_simple_results_{timestamp}.json"
        
        results = {
            'parameters': {
                'start_date': args.start_date,
                'end_date': args.end_date,
                'starting_capital': args.capital,
                'model': 'google/gemini-2.5-flash-preview-05-20'
            },
            'market_benchmark': market_return,
            'baseline_results': baseline_results,
            'llm_results': llm_results,
            'comparison_table': comparison_df.to_dict('records')
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{'='*80}")
        print(f"✅ COMPARISON COMPLETED!")
        print(f"{'='*80}")
        print(f"📄 Results saved to: {results_file}")
        print(f"📈 Data saved to: data/nifty50/2month_sample_data.csv")
        
    except Exception as e:
        logger.error(f"Error running comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()