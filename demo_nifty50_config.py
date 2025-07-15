#!/usr/bin/env python3
"""
NIFTY50 Configuration System Demo
Demonstrates how to use the configuration system for NIFTY50 trading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nifty50_config import (
    NIFTY50Config, 
    create_conservative_preset, 
    create_aggressive_preset, 
    create_balanced_preset
)
from nifty50_agent import NIFTY50TradingAgent, create_nifty50_args, dummy_llm_function
import pandas as pd

def demo_basic_config():
    """Demonstrate basic configuration usage"""
    print("="*60)
    print("BASIC CONFIGURATION DEMO")
    print("="*60)
    
    # Create default configuration
    config = NIFTY50Config()
    
    # Print current configuration
    config.print_config()
    
    # Validate configuration
    is_valid = config.validate_config()
    print(f"\nConfiguration is valid: {is_valid}")
    
    # Save configuration
    config.save_config("demo_config.json")
    print("\nConfiguration saved to demo_config.json")

def demo_preset_configurations():
    """Demonstrate preset configurations"""
    print("\n" + "="*60)
    print("PRESET CONFIGURATIONS DEMO")
    print("="*60)
    
    # Create different presets
    presets = {
        "Conservative": create_conservative_preset(),
        "Aggressive": create_aggressive_preset(),
        "Balanced": create_balanced_preset()
    }
    
    for name, config in presets.items():
        print(f"\n{name.upper()} PRESET:")
        print(f"- Max Position Size: {config.trading.max_position_size:.1%}")
        print(f"- Min Cash Reserve: {config.trading.min_cash_reserve:.1%}")
        print(f"- Stop Loss: {config.trading.stop_loss_threshold:.1%}")
        print(f"- Take Profit: {config.trading.take_profit_threshold:.1%}")
        print(f"- Max Daily Loss: {config.risk.max_daily_loss:.1%}")
        print(f"- Confidence Threshold: {config.analyst.min_confidence_threshold:.1%}")
        
        # Save preset
        config.create_preset(name.lower(), f"{name} trading strategy")
    
    # List all presets
    print(f"\nAvailable presets: {presets['Conservative'].list_presets()}")

def demo_custom_configuration():
    """Demonstrate custom configuration creation"""
    print("\n" + "="*60)
    print("CUSTOM CONFIGURATION DEMO")
    print("="*60)
    
    # Create custom configuration
    config = NIFTY50Config()
    
    # Customize trading parameters
    config.trading.starting_capital = 500_000  # 5 lakh INR
    config.trading.max_position_size = 0.7  # 70% max position
    config.trading.min_cash_reserve = 0.15  # 15% cash reserve
    config.trading.stop_loss_threshold = 0.04  # 4% stop loss
    config.trading.take_profit_threshold = 0.12  # 12% take profit
    
    # Customize analyst weights
    config.analyst.technical_weight = 0.5  # 50% technical
    config.analyst.news_weight = 0.3  # 30% news
    config.analyst.reflection_weight = 0.2  # 20% reflection
    config.analyst.min_confidence_threshold = 0.6  # 60% confidence
    
    # Customize risk parameters
    config.risk.max_daily_loss = 0.015  # 1.5% max daily loss
    config.risk.max_weekly_loss = 0.04  # 4% max weekly loss
    config.risk.position_sizing_method = "volatility"  # Volatility-based sizing
    
    # Customize model parameters
    config.model.model_name = "openai/gpt-4o"
    config.model.temperature = 0.1  # Slightly more random
    config.model.max_tokens = 1024  # Longer responses
    
    print("Custom configuration created:")
    print(f"- Starting Capital: ₹{config.trading.starting_capital:,.2f}")
    print(f"- Position Size: {config.trading.max_position_size:.1%}")
    print(f"- Cash Reserve: {config.trading.min_cash_reserve:.1%}")
    print(f"- Technical Weight: {config.analyst.technical_weight:.1%}")
    print(f"- News Weight: {config.analyst.news_weight:.1%}")
    print(f"- Reflection Weight: {config.analyst.reflection_weight:.1%}")
    print(f"- Model: {config.model.model_name}")
    print(f"- Temperature: {config.model.temperature}")
    
    # Validate custom configuration
    is_valid = config.validate_config()
    print(f"\nCustom configuration is valid: {is_valid}")
    
    # Save custom configuration
    config.save_config("custom_config.json")
    print("Custom configuration saved to custom_config.json")
    
    return config

def demo_configuration_with_agent(custom_config):
    """Demonstrate using configuration with trading agent"""
    print("\n" + "="*60)
    print("CONFIGURATION WITH TRADING AGENT DEMO")
    print("="*60)
    
    # Create arguments with custom configuration
    args = create_nifty50_args(
        starting_date="2024-01-01",
        ending_date="2024-01-31",  # Short demo period
        config=custom_config
    )
    
    print(f"Created trading arguments with custom config:")
    print(f"- Dataset: {args.dataset}")
    print(f"- Date range: {args.starting_date.date()} to {args.ending_date.date()}")
    print(f"- Use technical: {args.use_tech}")
    print(f"- Use news: {args.use_news}")
    print(f"- Use reflection: {args.use_reflection}")
    print(f"- Model: {args.model}")
    print(f"- Seed: {args.seed}")
    
    # Initialize agent with custom configuration
    try:
        agent = NIFTY50TradingAgent(args, dummy_llm_function, custom_config)
        print(f"\nSuccessfully initialized NIFTY50 Trading Agent with custom config")
        print(f"- Starting capital: ₹{agent.config.trading.starting_capital:,.2f}")
        print(f"- Transaction cost: {agent.env.total_transaction_cost:.4%}")
        print(f"- Price window: {agent.price_window} days")
        print(f"- Reflection window: {agent.reflection_window} trades")
        
        # Note: We're not running the actual trading session in this demo
        # as it would require real data and may take time
        print("\nAgent is ready to start trading!")
        
    except Exception as e:
        print(f"Error initializing agent: {e}")

def demo_parameter_updates():
    """Demonstrate dynamic parameter updates"""
    print("\n" + "="*60)
    print("DYNAMIC PARAMETER UPDATES DEMO")
    print("="*60)
    
    config = NIFTY50Config()
    
    # Show initial values
    print("Initial values:")
    print(f"- Starting capital: ₹{config.trading.starting_capital:,.2f}")
    print(f"- Max position size: {config.trading.max_position_size:.1%}")
    print(f"- Technical weight: {config.analyst.technical_weight:.1%}")
    
    # Update parameters
    config.update_parameter('trading', 'starting_capital', 2_000_000)
    config.update_parameter('trading', 'max_position_size', 0.6)
    config.update_parameter('analyst', 'technical_weight', 0.6)
    
    # Show updated values
    print("\nAfter updates:")
    print(f"- Starting capital: ₹{config.trading.starting_capital:,.2f}")
    print(f"- Max position size: {config.trading.max_position_size:.1%}")
    print(f"- Technical weight: {config.analyst.technical_weight:.1%}")
    
    # Get specific parameter
    capital = config.get_parameter('trading', 'starting_capital')
    print(f"\nRetrieved starting capital: ₹{capital:,.2f}")

def demo_transaction_costs():
    """Demonstrate transaction cost calculations"""
    print("\n" + "="*60)
    print("TRANSACTION COST CALCULATIONS DEMO")
    print("="*60)
    
    config = NIFTY50Config()
    
    # Show individual cost components
    print("Indian Market Transaction Costs:")
    print(f"- Brokerage Fee: {config.market.brokerage_fee:.4%}")
    print(f"- Securities Transaction Tax (STT): {config.market.stt_rate:.4%}")
    print(f"- Exchange Fee: {config.market.exchange_fee:.6%}")
    print(f"- SEBI Fee: {config.market.sebi_fee:.7%}")
    print(f"- GST Rate: {config.market.gst_rate:.1%}")
    
    # Calculate total transaction cost
    total_cost = config.get_total_transaction_cost()
    print(f"\nTotal Transaction Cost: {total_cost:.4%}")
    
    # Example calculation for ₹1,00,000 trade
    trade_amount = 100_000
    cost_amount = trade_amount * total_cost
    print(f"\nFor a ₹{trade_amount:,.2f} trade:")
    print(f"- Total transaction cost: ₹{cost_amount:.2f}")
    print(f"- Net amount after costs: ₹{trade_amount - cost_amount:.2f}")

def demo_position_sizing():
    """Demonstrate position sizing calculations"""
    print("\n" + "="*60)
    print("POSITION SIZING DEMO")
    print("="*60)
    
    config = NIFTY50Config()
    
    # Test with different portfolio values
    portfolio_values = [1_000_000, 500_000, 2_000_000, 10_000_000]
    
    print("Position Size Limits:")
    print(f"- Max position size: {config.trading.max_position_size:.1%}")
    print(f"- Min cash reserve: {config.trading.min_cash_reserve:.1%}")
    print()
    
    for portfolio_value in portfolio_values:
        max_position = config.get_position_size_limit(portfolio_value)
        print(f"Portfolio: ₹{portfolio_value:,.2f} → Max position: ₹{max_position:,.2f}")

def main():
    """Run all configuration demos"""
    print("NIFTY50 Configuration System Demo")
    print("="*60)
    
    # Run all demos
    demo_basic_config()
    demo_preset_configurations()
    custom_config = demo_custom_configuration()
    demo_configuration_with_agent(custom_config)
    demo_parameter_updates()
    demo_transaction_costs()
    demo_position_sizing()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("\nConfiguration files created:")
    print("- demo_config.json")
    print("- custom_config.json")
    print("- nifty50_preset_conservative.json")
    print("- nifty50_preset_aggressive.json")
    print("- nifty50_preset_balanced.json")
    print("\nYou can now use these configurations in your trading system!")

if __name__ == "__main__":
    main()