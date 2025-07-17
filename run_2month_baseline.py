#!/usr/bin/env python3
"""
NIFTY50 2-Month Baseline Comparison
Run comprehensive comparison with baseline strategies for 2 months of 2024
"""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nifty50_strategy_comparison import NIFTY50StrategyComparison
from nifty50_config import NIFTY50Config, create_conservative_preset, create_aggressive_preset, create_balanced_preset
from nifty50_baseline_strategies import BaselineStrategyManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

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
    # Slightly positive bias with higher volatility
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
    
    # Market statistics
    sample_data['market_cap'] = sample_data['close'] * 1000000  # Approximate
    sample_data['daily_return'] = sample_data['close'].pct_change()
    sample_data['high_low_spread'] = (sample_data['high'] - sample_data['low']) / sample_data['open']
    
    logger.info(f"Sample data created: {len(sample_data)} rows")
    logger.info(f"Price range: ₹{sample_data['close'].min():.2f} to ₹{sample_data['close'].max():.2f}")
    logger.info(f"Total return: {((sample_data['close'].iloc[-1] / sample_data['close'].iloc[0]) - 1) * 100:.2f}%")
    
    return sample_data

def run_2month_baseline_comparison(start_date: str, end_date: str, starting_capital: int = 1000000):
    """Run 2-month baseline comparison"""
    
    print(f"\n{'='*80}")
    print(f"🚀 NIFTY50 2-MONTH BASELINE COMPARISON")
    print(f"{'='*80}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"💰 Starting Capital: ₹{starting_capital:,.2f}")
    print(f"🤖 Model: google/gemini-2.5-flash-preview-05-20")
    
    # Create extended sample data
    sample_data = create_extended_sample_data(start_date, end_date)
    
    # Save sample data
    sample_data.to_csv('data/nifty50/2month_sample_data.csv', index=False)
    logger.info("Sample data saved to data/nifty50/2month_sample_data.csv")
    
    # Initialize comparison
    comparison = NIFTY50StrategyComparison(starting_capital)
    
    # Run baseline strategies
    print(f"\n{'='*60}")
    print(f"📊 RUNNING BASELINE STRATEGIES")
    print(f"{'='*60}")
    
    baseline_results = comparison.run_baseline_comparison(sample_data)
    
    # Create LLM agent configurations
    print(f"\n{'='*60}")
    print(f"🤖 RUNNING LLM AGENT CONFIGURATIONS")
    print(f"{'='*60}")
    
    # Update model in configs
    configs = {}
    
    # Conservative
    conservative_config = create_conservative_preset()
    conservative_config.model.model_name = "google/gemini-2.5-flash-preview-05-20"
    configs['Conservative'] = conservative_config
    
    # Balanced
    balanced_config = create_balanced_preset()
    balanced_config.model.model_name = "google/gemini-2.5-flash-preview-05-20"
    configs['Balanced'] = balanced_config
    
    # Aggressive
    aggressive_config = create_aggressive_preset()
    aggressive_config.model.model_name = "google/gemini-2.5-flash-preview-05-20"
    configs['Aggressive'] = aggressive_config
    
    # Technical only
    tech_config = create_balanced_preset()
    tech_config.model.model_name = "google/gemini-2.5-flash-preview-05-20"
    tech_config.analyst.use_technical = True
    tech_config.analyst.use_news = False
    tech_config.analyst.use_reflection = True
    tech_config.analyst.technical_weight = 0.7
    tech_config.analyst.reflection_weight = 0.3
    configs['Technical Only'] = tech_config
    
    # News only
    news_config = create_balanced_preset()
    news_config.model.model_name = "google/gemini-2.5-flash-preview-05-20"
    news_config.analyst.use_technical = False
    news_config.analyst.use_news = True
    news_config.analyst.use_reflection = True
    news_config.analyst.news_weight = 0.7
    news_config.analyst.reflection_weight = 0.3
    configs['News Only'] = news_config
    
    # Run LLM agent comparison
    llm_results = comparison.run_llm_agent_comparison(sample_data, configs)
    
    # Generate comprehensive report
    print(f"\n{'='*80}")
    print(f"📋 COMPREHENSIVE COMPARISON REPORT")
    print(f"{'='*80}")
    
    report = comparison.generate_performance_report()
    print(report)
    
    # Create detailed comparison table
    comparison_df = comparison.create_comprehensive_comparison()
    
    print(f"\n{'='*80}")
    print(f"📊 DETAILED COMPARISON TABLE")
    print(f"{'='*80}")
    
    # Format the table nicely
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    
    print(comparison_df.round(4).to_string(index=False))
    
    # Create performance plots
    print(f"\n📈 Creating performance plots...")
    comparison.create_performance_plots("2month_baseline_comparison.png")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"2month_baseline_results_{timestamp}.json"
    comparison.save_results(results_file)
    
    # Create summary statistics
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    baseline_returns = comparison_df[comparison_df['Type'] == 'Baseline']['Total Return']
    llm_returns = comparison_df[comparison_df['Type'] == 'LLM-based']['Total Return']
    
    print(f"Baseline Strategies ({len(baseline_returns)} strategies):")
    print(f"  • Best Return: {baseline_returns.max():.2%}")
    print(f"  • Worst Return: {baseline_returns.min():.2%}")
    print(f"  • Average Return: {baseline_returns.mean():.2%}")
    print(f"  • Std Deviation: {baseline_returns.std():.2%}")
    
    print(f"\nLLM-based Strategies ({len(llm_returns)} strategies):")
    print(f"  • Best Return: {llm_returns.max():.2%}")
    print(f"  • Worst Return: {llm_returns.min():.2%}")
    print(f"  • Average Return: {llm_returns.mean():.2%}")
    print(f"  • Std Deviation: {llm_returns.std():.2%}")
    
    # Market benchmark
    market_return = ((sample_data['close'].iloc[-1] / sample_data['close'].iloc[0]) - 1)
    print(f"\nMarket Benchmark (Buy & Hold NIFTY50):")
    print(f"  • Market Return: {market_return:.2%}")
    
    # Count strategies that beat the market
    beat_market_baseline = len(baseline_returns[baseline_returns > market_return])
    beat_market_llm = len(llm_returns[llm_returns > market_return])
    
    print(f"\nStrategies that Beat Market:")
    print(f"  • Baseline: {beat_market_baseline}/{len(baseline_returns)} ({beat_market_baseline/len(baseline_returns)*100:.1f}%)")
    print(f"  • LLM-based: {beat_market_llm}/{len(llm_returns)} ({beat_market_llm/len(llm_returns)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"✅ COMPARISON COMPLETED!")
    print(f"{'='*80}")
    print(f"📄 Results saved to: {results_file}")
    print(f"📊 Plot saved to: 2month_baseline_comparison.png")
    print(f"📈 Data saved to: data/nifty50/2month_sample_data.csv")
    
    return comparison_df, sample_data

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='NIFTY50 2-Month Baseline Comparison')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-02-29', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=int, default=1000000, help='Starting capital in INR')
    
    args = parser.parse_args()
    
    # Run comparison
    try:
        comparison_df, sample_data = run_2month_baseline_comparison(
            args.start_date, 
            args.end_date, 
            args.capital
        )
        
        print(f"\n🎯 Top 3 Strategies:")
        top_3 = comparison_df.head(3)
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            print(f"{i}. {row['Strategy']}: {row['Total Return']:.2%} return")
        
    except Exception as e:
        logger.error(f"Error running comparison: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()