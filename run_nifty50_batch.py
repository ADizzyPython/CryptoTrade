#!/usr/bin/env python3
"""
NIFTY50 Batch Execution Script
Run multiple NIFTY50 trading configurations and compare results
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nifty50_agent import NIFTY50TradingAgent, create_nifty50_args
from nifty50_config import NIFTY50Config, create_conservative_preset, create_aggressive_preset, create_balanced_preset
from nifty50_strategy_comparison import NIFTY50StrategyComparison
from utils import get_chat
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nifty50_batch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_llm_function():
    """Create the LLM function wrapper"""
    def llm_function(prompt: str, model: str, seed: int) -> str:
        try:
            response = get_chat(prompt, model, seed, temperature=0.0, max_tokens=512)
            return response
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return "Error: Unable to get LLM response. Using conservative strategy."
    
    return llm_function

class NIFTY50BatchRunner:
    """Batch runner for multiple NIFTY50 configurations"""
    
    def __init__(self, start_date: str, end_date: str, starting_capital: float = 1_000_000):
        self.start_date = start_date
        self.end_date = end_date
        self.starting_capital = starting_capital
        self.llm_function = create_llm_function()
        self.results = {}
        self.comparison_results = {}
        
    def create_test_configurations(self) -> Dict[str, NIFTY50Config]:
        """Create various test configurations"""
        configs = {}
        
        # 1. Standard presets
        configs['Conservative'] = create_conservative_preset()
        configs['Balanced'] = create_balanced_preset()
        configs['Aggressive'] = create_aggressive_preset()
        
        # 2. Technical analysis focused
        tech_config = create_balanced_preset()
        tech_config.analyst.use_technical = True
        tech_config.analyst.use_news = False
        tech_config.analyst.use_reflection = True
        tech_config.analyst.technical_weight = 0.7
        tech_config.analyst.reflection_weight = 0.3
        configs['Technical Only'] = tech_config
        
        # 3. News analysis focused
        news_config = create_balanced_preset()
        news_config.analyst.use_technical = False
        news_config.analyst.use_news = True
        news_config.analyst.use_reflection = True
        news_config.analyst.news_weight = 0.7
        news_config.analyst.reflection_weight = 0.3
        configs['News Only'] = news_config
        
        # 4. High frequency trading
        high_freq_config = create_aggressive_preset()
        high_freq_config.trading.max_position_size = 0.95
        high_freq_config.trading.min_cash_reserve = 0.05
        high_freq_config.analyst.min_confidence_threshold = 0.2
        configs['High Frequency'] = high_freq_config
        
        # 5. Conservative long-term
        long_term_config = create_conservative_preset()
        long_term_config.trading.max_position_size = 0.5
        long_term_config.trading.min_cash_reserve = 0.3
        long_term_config.analyst.min_confidence_threshold = 0.8
        configs['Long Term'] = long_term_config
        
        # 6. Equal weight analysts
        equal_weight_config = create_balanced_preset()
        equal_weight_config.analyst.technical_weight = 0.33
        equal_weight_config.analyst.news_weight = 0.33
        equal_weight_config.analyst.reflection_weight = 0.34
        configs['Equal Weight'] = equal_weight_config
        
        # 7. Different models (if available)
        gpt4_config = create_balanced_preset()
        gpt4_config.model.model_name = "openai/gpt-4o"
        gpt4_config.model.temperature = 0.0
        configs['GPT-4'] = gpt4_config
        
        # 8. Higher temperature for more randomness
        random_config = create_balanced_preset()
        random_config.model.temperature = 0.3
        configs['Random'] = random_config
        
        return configs
    
    def run_single_configuration(self, config_name: str, config: NIFTY50Config) -> Dict:
        """Run a single configuration"""
        logger.info(f"Running configuration: {config_name}")
        
        try:
            # Create arguments
            args = create_nifty50_args(
                starting_date=self.start_date,
                ending_date=self.end_date,
                config=config
            )
            
            # Initialize agent
            agent = NIFTY50TradingAgent(args, self.llm_function, config)
            
            # Run trading session
            results = agent.run_trading_session()
            
            # Add configuration info to results
            results['config_name'] = config_name
            results['config_summary'] = {
                'use_technical': config.analyst.use_technical,
                'use_news': config.analyst.use_news,
                'use_reflection': config.analyst.use_reflection,
                'model': config.model.model_name,
                'max_position_size': config.trading.max_position_size,
                'min_confidence': config.analyst.min_confidence_threshold
            }
            
            logger.info(f"Completed {config_name}: Return={results['performance_metrics']['total_return']:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running configuration {config_name}: {e}")
            return {
                'config_name': config_name,
                'error': str(e),
                'performance_metrics': {
                    'total_return': 0,
                    'final_value': self.starting_capital,
                    'sharpe_ratio': 0,
                    'volatility': 0,
                    'total_trades': 0
                }
            }
    
    def run_batch_execution(self, configs: Dict[str, NIFTY50Config] = None) -> Dict:
        """Run batch execution with multiple configurations"""
        if configs is None:
            configs = self.create_test_configurations()
        
        logger.info(f"Starting batch execution with {len(configs)} configurations")
        
        self.results = {}
        
        for config_name, config in configs.items():
            result = self.run_single_configuration(config_name, config)
            self.results[config_name] = result
        
        logger.info("Batch execution completed")
        return self.results
    
    def run_with_baseline_comparison(self, configs: Dict[str, NIFTY50Config] = None):
        """Run batch execution with baseline strategy comparison"""
        logger.info("Starting comprehensive comparison with baseline strategies")
        
        # Create sample data for comparison
        # In practice, you would load real NIFTY50 data
        import numpy as np
        
        np.random.seed(42)
        date_range = pd.date_range(self.start_date, self.end_date, freq='D')
        
        # Create realistic price data
        returns = np.random.normal(0.0008, 0.02, len(date_range))
        prices = 20000 * np.exp(np.cumsum(returns))
        
        sample_data = pd.DataFrame({
            'timestamp': date_range,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.02, len(date_range))),
            'low': prices * (1 - np.random.uniform(0, 0.02, len(date_range))),
            'close': prices * (1 + np.random.uniform(-0.01, 0.01, len(date_range))),
            'volume': np.random.randint(100000, 1000000, len(date_range))
        })
        
        # Add technical indicators
        sample_data['sma_5'] = sample_data['close'].rolling(5).mean()
        sample_data['sma_20'] = sample_data['close'].rolling(20).mean()
        sample_data['rsi'] = 50 + np.random.uniform(-20, 20, len(date_range))
        sample_data['macd'] = np.random.uniform(-100, 100, len(date_range))
        sample_data['macd_signal'] = sample_data['macd'].rolling(9).mean()
        sample_data['volatility'] = sample_data['close'].rolling(20).std()
        
        # Initialize comparison
        comparison = NIFTY50StrategyComparison(self.starting_capital)
        
        # Run baseline comparison
        baseline_results = comparison.run_baseline_comparison(sample_data)
        
        # Run LLM agent comparison
        if configs is None:
            configs = self.create_test_configurations()
        
        llm_results = comparison.run_llm_agent_comparison(sample_data, configs)
        
        # Store comparison results
        self.comparison_results = {
            'baseline_results': baseline_results,
            'llm_results': llm_results,
            'comparison_df': comparison.create_comprehensive_comparison()
        }
        
        # Generate report
        report = comparison.generate_performance_report()
        print("\n" + "="*80)
        print("COMPREHENSIVE STRATEGY COMPARISON")
        print("="*80)
        print(report)
        
        # Create performance plots
        comparison.create_performance_plots("batch_performance_comparison.png")
        
        # Save results
        comparison.save_results("batch_comparison_results.json")
        
        return self.comparison_results
    
    def generate_batch_report(self) -> str:
        """Generate comprehensive batch report"""
        if not self.results:
            return "No results available"
        
        report = f"""
NIFTY50 BATCH EXECUTION REPORT
==============================

Execution Period: {self.start_date} to {self.end_date}
Starting Capital: ₹{self.starting_capital:,.2f}
Total Configurations: {len(self.results)}

PERFORMANCE SUMMARY:
"""
        
        # Sort results by return
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['performance_metrics']['total_return'],
            reverse=True
        )
        
        for i, (config_name, result) in enumerate(sorted_results, 1):
            metrics = result['performance_metrics']
            report += f"""
{i}. {config_name}:
   - Total Return: {metrics['total_return']:.2%}
   - Final Value: ₹{metrics['final_value']:,.2f}
   - Sharpe Ratio: {metrics['sharpe_ratio']:.4f}
   - Total Trades: {metrics['total_trades']}
"""
        
        # Performance statistics
        returns = [r['performance_metrics']['total_return'] for r in self.results.values()]
        sharpe_ratios = [r['performance_metrics']['sharpe_ratio'] for r in self.results.values()]
        
        report += f"""
OVERALL STATISTICS:
- Best Return: {max(returns):.2%}
- Worst Return: {min(returns):.2%}
- Average Return: {sum(returns)/len(returns):.2%}
- Best Sharpe: {max(sharpe_ratios):.4f}
- Average Sharpe: {sum(sharpe_ratios)/len(sharpe_ratios):.4f}
"""
        
        return report
    
    def save_batch_results(self, filepath: str = None):
        """Save batch results to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"nifty50_batch_results_{timestamp}.json"
        
        batch_data = {
            'execution_info': {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'starting_capital': self.starting_capital,
                'execution_time': datetime.now().isoformat()
            },
            'results': self.results,
            'comparison_results': self.comparison_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(batch_data, f, indent=2, default=str)
        
        logger.info(f"Batch results saved to {filepath}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run NIFTY50 Trading Agent Batch')
    
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, default='2024-03-31',
                       help='End date in YYYY-MM-DD format')
    parser.add_argument('--capital', type=float, default=1_000_000,
                       help='Starting capital in INR')
    parser.add_argument('--with-baseline', action='store_true',
                       help='Include baseline strategy comparison')
    parser.add_argument('--config-file', type=str, default=None,
                       help='JSON file with custom configurations')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    print("NIFTY50 Batch Execution")
    print("="*50)
    
    args = parse_arguments()
    
    # Initialize batch runner
    batch_runner = NIFTY50BatchRunner(args.start_date, args.end_date, args.capital)
    
    # Load custom configurations if specified
    custom_configs = None
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                config_data = json.load(f)
            # Process custom configurations
            logger.info(f"Loaded custom configurations from {args.config_file}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Run batch execution
    if args.with_baseline:
        results = batch_runner.run_with_baseline_comparison(custom_configs)
    else:
        results = batch_runner.run_batch_execution(custom_configs)
    
    # Generate report
    report = batch_runner.generate_batch_report()
    print(report)
    
    # Save results
    batch_runner.save_batch_results(args.output)
    
    logger.info("Batch execution completed successfully")

if __name__ == "__main__":
    main()