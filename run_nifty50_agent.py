#!/usr/bin/env python3
"""
NIFTY50 Trading Agent Execution Script
Main script to run the NIFTY50 trading agent with various configurations
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nifty50_agent import NIFTY50TradingAgent, create_nifty50_args
from nifty50_config import NIFTY50Config, create_conservative_preset, create_aggressive_preset, create_balanced_preset
from utils import get_chat  # Import the LLM function from utils
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nifty50_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_llm_function():
    """Create the LLM function wrapper"""
    def llm_function(prompt: str, model: str, seed: int) -> str:
        """
        LLM function that calls the chat completion API
        
        Args:
            prompt: The input prompt
            model: Model name to use
            seed: Random seed for reproducibility
            
        Returns:
            Model response
        """
        try:
            response = get_chat(prompt, model, seed, temperature=0.0, max_tokens=512)
            return response
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return "Error: Unable to get LLM response. Using conservative strategy."
    
    return llm_function

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run NIFTY50 Trading Agent')
    
    # Date arguments
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='Start date in YYYY-MM-DD format (default: 2024-01-01)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                       help='End date in YYYY-MM-DD format (default: 2024-12-31)')
    
    # Configuration arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (optional)')
    parser.add_argument('--preset', type=str, choices=['conservative', 'balanced', 'aggressive'],
                       default='balanced', help='Configuration preset to use (default: balanced)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='openai/gpt-4o',
                       help='Model to use (default: openai/gpt-4o)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Analysis flags
    parser.add_argument('--no-tech', action='store_true',
                       help='Disable technical analysis')
    parser.add_argument('--no-news', action='store_true',
                       help='Disable news analysis')
    parser.add_argument('--no-reflection', action='store_true',
                       help='Disable reflection analysis')
    
    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--save-config', action='store_true',
                       help='Save configuration to file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in dry-run mode (no actual execution)')
    
    # Data arguments
    parser.add_argument('--news-dir', type=str, default='data/selected_nifty50_202401_202501',
                       help='Directory containing news data')
    parser.add_argument('--data-dir', type=str, default='data/nifty50',
                       help='Directory for price data')
    
    return parser.parse_args()

def setup_configuration(args):
    """Setup configuration based on arguments"""
    config = None
    
    # Load configuration from file if specified
    if args.config:
        config = NIFTY50Config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    
    # Or use preset
    elif args.preset:
        if args.preset == 'conservative':
            config = create_conservative_preset()
        elif args.preset == 'aggressive':
            config = create_aggressive_preset()
        else:
            config = create_balanced_preset()
        logger.info(f"Using {args.preset} preset configuration")
    
    # Or use default
    else:
        config = NIFTY50Config()
        logger.info("Using default configuration")
    
    # Override configuration with command line arguments
    if args.model:
        config.model.model_name = args.model
    if args.seed:
        config.model.seed = args.seed
    
    # Set analysis flags
    config.analyst.use_technical = not args.no_tech
    config.analyst.use_news = not args.no_news
    config.analyst.use_reflection = not args.no_reflection
    
    # Set data directories
    config.data.news_dir = args.news_dir
    config.data.data_dir = args.data_dir
    
    # Update backtest dates
    config.backtest.starting_date = args.start_date
    config.backtest.ending_date = args.end_date
    
    # Validate configuration
    if not config.validate_config():
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    # Save configuration if requested
    if args.save_config:
        config_filename = f"nifty50_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        config.save_config(config_filename)
        logger.info(f"Configuration saved to {config_filename}")
    
    return config

def run_trading_agent(config: NIFTY50Config, args, llm_function):
    """Run the trading agent"""
    logger.info("Initializing NIFTY50 Trading Agent")
    
    # Create arguments
    agent_args = create_nifty50_args(
        starting_date=args.start_date,
        ending_date=args.end_date,
        config=config
    )
    
    # Print configuration summary
    logger.info("Configuration Summary:")
    logger.info(f"  - Trading Period: {args.start_date} to {args.end_date}")
    logger.info(f"  - Starting Capital: ₹{config.trading.starting_capital:,.2f}")
    logger.info(f"  - Model: {config.model.model_name}")
    logger.info(f"  - Technical Analysis: {config.analyst.use_technical}")
    logger.info(f"  - News Analysis: {config.analyst.use_news}")
    logger.info(f"  - Reflection Analysis: {config.analyst.use_reflection}")
    logger.info(f"  - Max Position Size: {config.trading.max_position_size:.1%}")
    logger.info(f"  - Transaction Cost: {config.get_total_transaction_cost():.4%}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual trading will be performed")
        logger.info("This would initialize the trading agent and run the simulation")
        logger.info("To run actual simulation, remove the --dry-run flag")
        
        # Show what would happen
        logger.info("\nDRY RUN SIMULATION:")
        logger.info(f"  - Would load data from {args.start_date} to {args.end_date}")
        logger.info(f"  - Would use {config.model.model_name} for analysis")
        logger.info(f"  - Would start with ₹{config.trading.starting_capital:,.2f}")
        logger.info(f"  - Would use analysts: Technical={config.analyst.use_technical}, News={config.analyst.use_news}, Reflection={config.analyst.use_reflection}")
        logger.info(f"  - Would apply {config.get_total_transaction_cost():.4%} transaction costs")
        logger.info("\nTo run actual trading simulation, use:")
        logger.info(f"  python run_nifty50_agent.py --start-date {args.start_date} --end-date {args.end_date} --preset {args.preset}")
        
        return None
    
    # Initialize agent
    try:
        agent = NIFTY50TradingAgent(agent_args, llm_function, config)
        logger.info("Agent initialized successfully")
        
        # Run trading session
        logger.info("Starting trading session...")
        results = agent.run_trading_session()
        
        # Generate performance report
        report = agent.generate_performance_report(results)
        print("\n" + "="*80)
        print("TRADING SESSION COMPLETED")
        print("="*80)
        print(report)
        
        # Analyze trading patterns
        patterns = agent.analyze_trading_patterns(results)
        print("\nTRADING PATTERN ANALYSIS:")
        print(json.dumps(patterns, indent=2, default=str))
        
        # Save results
        output_path = args.output
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"nifty50_results_{timestamp}.json"
        
        agent.save_results(results, output_path)
        logger.info(f"Results saved to {output_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error running trading agent: {e}")
        raise

def main():
    """Main execution function"""
    print("NIFTY50 Trading Agent")
    print("="*50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup configuration
    config = setup_configuration(args)
    
    # Print configuration
    if not args.dry_run:
        config.print_config()
    
    # Create LLM function
    llm_function = create_llm_function()
    
    # Test LLM function
    try:
        test_response = llm_function("Test prompt", config.model.model_name, config.model.seed)
        logger.info("LLM connection test successful")
    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        logger.error("Please check your API key and internet connection")
        sys.exit(1)
    
    # Run trading agent
    try:
        results = run_trading_agent(config, args, llm_function)
        
        if results:
            logger.info("Trading session completed successfully")
            
            # Print final summary
            metrics = results['performance_metrics']
            print(f"\nFINAL RESULTS:")
            print(f"Total Return: {metrics['total_return']:.2%}")
            print(f"Final Value: ₹{metrics['final_value']:,.2f}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"Total Trades: {metrics['total_trades']}")
            
    except KeyboardInterrupt:
        logger.info("Trading session interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Trading session failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()