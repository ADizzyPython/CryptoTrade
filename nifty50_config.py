#!/usr/bin/env python3
"""
NIFTY50 Configuration System
Centralized configuration management for NIFTY50 trading parameters
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingParameters:
    """Trading strategy parameters"""
    starting_capital: float = 1_000_000.0  # Starting capital in INR
    price_window: int = 7  # Days of price history for analysis
    reflection_window: int = 3  # Number of past trades to reflect on
    max_position_size: float = 0.8  # Maximum position size (80% of capital)
    min_cash_reserve: float = 0.1  # Minimum cash reserve (10% of capital)
    stop_loss_threshold: float = 0.05  # Stop loss at 5% loss
    take_profit_threshold: float = 0.15  # Take profit at 15% gain
    volatility_threshold: float = 0.02  # High volatility threshold (2%)
    
@dataclass
class MarketParameters:
    """Indian market specific parameters"""
    trading_hours_start: str = "09:15"  # NSE trading start time
    trading_hours_end: str = "15:30"    # NSE trading end time
    market_timezone: str = "Asia/Kolkata"
    brokerage_fee: float = 0.0003  # 0.03% brokerage fee
    stt_rate: float = 0.001  # 0.1% Securities Transaction Tax
    exchange_fee: float = 0.0000325  # NSE exchange fee
    sebi_fee: float = 0.000001  # SEBI regulatory fee
    gst_rate: float = 0.18  # 18% GST on brokerage
    
@dataclass
class DataParameters:
    """Data source and processing parameters"""
    data_dir: str = "data/nifty50"
    news_dir: str = "data/selected_nifty50_202401_202501"
    price_data_source: str = "yfinance"  # "yfinance" or "nsepy"
    nifty_symbol: str = "^NSEI"
    sensex_symbol: str = "^BSESN"
    update_frequency: str = "daily"  # "daily", "hourly", "realtime"
    news_sentiment_threshold: float = 0.3  # Sentiment threshold for trading signals
    
@dataclass
class TechnicalIndicators:
    """Technical analysis parameters"""
    sma_periods: list = None  # Simple Moving Average periods
    ema_periods: list = None  # Exponential Moving Average periods
    macd_fast: int = 12  # MACD fast period
    macd_slow: int = 26  # MACD slow period
    macd_signal: int = 9  # MACD signal period
    rsi_period: int = 14  # RSI period
    bollinger_period: int = 20  # Bollinger Bands period
    bollinger_std: float = 2.0  # Bollinger Bands standard deviation
    volatility_period: int = 20  # Volatility calculation period
    
    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [5, 10, 15, 20, 30, 50]
        if self.ema_periods is None:
            self.ema_periods = [12, 26, 50]

@dataclass
class ModelParameters:
    """Language model parameters"""
    model_name: str = "google/gemini-2.5-flash-preview-05-20"
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int = 42
    retry_attempts: int = 3
    timeout: int = 30
    
@dataclass
class AnalystConfig:
    """Analyst configuration parameters"""
    use_technical: bool = True
    use_news: bool = True
    use_reflection: bool = True
    technical_weight: float = 0.4  # Weight for technical analysis
    news_weight: float = 0.3  # Weight for news analysis
    reflection_weight: float = 0.3  # Weight for reflection analysis
    min_confidence_threshold: float = 0.5  # Minimum confidence for trading action
    
@dataclass
class RiskManagement:
    """Risk management parameters"""
    max_daily_loss: float = 0.02  # Maximum daily loss (2%)
    max_weekly_loss: float = 0.05  # Maximum weekly loss (5%)
    max_monthly_loss: float = 0.10  # Maximum monthly loss (10%)
    position_sizing_method: str = "fixed"  # "fixed", "kelly", "volatility"
    risk_free_rate: float = 0.07  # Indian risk-free rate (7%)
    max_correlation_threshold: float = 0.8  # Maximum correlation threshold
    
@dataclass
class BacktestParameters:
    """Backtesting configuration"""
    starting_date: str = "2024-01-01"
    ending_date: str = "2024-12-31"
    benchmark_symbol: str = "^NSEI"  # Benchmark for comparison
    transaction_cost_model: str = "indian_market"  # Transaction cost model
    slippage_model: str = "linear"  # Slippage model
    slippage_rate: float = 0.0001  # 0.01% slippage
    
class NIFTY50Config:
    """Main configuration class for NIFTY50 trading system"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file or "nifty50_config.json"
        
        # Initialize with default parameters
        self.trading = TradingParameters()
        self.market = MarketParameters()
        self.data = DataParameters()
        self.technical = TechnicalIndicators()
        self.model = ModelParameters()
        self.analyst = AnalystConfig()
        self.risk = RiskManagement()
        self.backtest = BacktestParameters()
        
        # Load configuration if file exists
        if os.path.exists(self.config_file):
            self.load_config()
        else:
            logger.info(f"Configuration file {self.config_file} not found. Using defaults.")
    
    def load_config(self, config_file: Optional[str] = None) -> None:
        """
        Load configuration from JSON file
        
        Args:
            config_file: Path to configuration file
        """
        if config_file:
            self.config_file = config_file
            
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration sections
            if 'trading' in config_data:
                self.trading = TradingParameters(**config_data['trading'])
            if 'market' in config_data:
                self.market = MarketParameters(**config_data['market'])
            if 'data' in config_data:
                self.data = DataParameters(**config_data['data'])
            if 'technical' in config_data:
                self.technical = TechnicalIndicators(**config_data['technical'])
            if 'model' in config_data:
                self.model = ModelParameters(**config_data['model'])
            if 'analyst' in config_data:
                self.analyst = AnalystConfig(**config_data['analyst'])
            if 'risk' in config_data:
                self.risk = RiskManagement(**config_data['risk'])
            if 'backtest' in config_data:
                self.backtest = BacktestParameters(**config_data['backtest'])
            
            logger.info(f"Configuration loaded from {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
    
    def save_config(self, config_file: Optional[str] = None) -> None:
        """
        Save current configuration to JSON file
        
        Args:
            config_file: Path to save configuration file
        """
        if config_file:
            self.config_file = config_file
            
        config_data = {
            'trading': asdict(self.trading),
            'market': asdict(self.market),
            'data': asdict(self.data),
            'technical': asdict(self.technical),
            'model': asdict(self.model),
            'analyst': asdict(self.analyst),
            'risk': asdict(self.risk),
            'backtest': asdict(self.backtest),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get_total_transaction_cost(self) -> float:
        """
        Calculate total transaction cost percentage
        
        Returns:
            Total transaction cost as percentage
        """
        base_cost = (self.market.brokerage_fee + 
                    self.market.stt_rate + 
                    self.market.exchange_fee + 
                    self.market.sebi_fee)
        
        # Add GST on brokerage
        gst_amount = self.market.brokerage_fee * self.market.gst_rate
        
        return base_cost + gst_amount
    
    def get_position_size_limit(self, current_capital: float) -> float:
        """
        Calculate position size limit based on current capital
        
        Args:
            current_capital: Current portfolio value
            
        Returns:
            Maximum position size in INR
        """
        max_position = current_capital * self.trading.max_position_size
        min_cash_required = current_capital * self.trading.min_cash_reserve
        
        return min(max_position, current_capital - min_cash_required)
    
    def validate_config(self) -> bool:
        """
        Validate configuration parameters
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate trading parameters
            assert 0 < self.trading.max_position_size <= 1.0, "Max position size must be between 0 and 1"
            assert 0 < self.trading.min_cash_reserve <= 1.0, "Min cash reserve must be between 0 and 1"
            assert self.trading.max_position_size + self.trading.min_cash_reserve <= 1.0, "Position size + cash reserve cannot exceed 100%"
            
            # Validate analyst weights
            total_weight = (self.analyst.technical_weight + 
                          self.analyst.news_weight + 
                          self.analyst.reflection_weight)
            assert abs(total_weight - 1.0) < 0.01, "Analyst weights must sum to 1.0"
            
            # Validate risk parameters
            assert 0 < self.risk.max_daily_loss <= 1.0, "Max daily loss must be between 0 and 1"
            assert 0 < self.risk.max_weekly_loss <= 1.0, "Max weekly loss must be between 0 and 1"
            assert 0 < self.risk.max_monthly_loss <= 1.0, "Max monthly loss must be between 0 and 1"
            
            # Validate technical indicators
            assert self.technical.macd_fast < self.technical.macd_slow, "MACD fast period must be less than slow period"
            assert self.technical.bollinger_std > 0, "Bollinger Bands standard deviation must be positive"
            
            logger.info("Configuration validation passed")
            return True
            
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return False
    
    def update_parameter(self, section: str, parameter: str, value: Any) -> None:
        """
        Update a specific parameter
        
        Args:
            section: Configuration section name
            parameter: Parameter name
            value: New value
        """
        try:
            section_obj = getattr(self, section)
            setattr(section_obj, parameter, value)
            logger.info(f"Updated {section}.{parameter} = {value}")
            
        except AttributeError:
            logger.error(f"Section '{section}' or parameter '{parameter}' not found")
    
    def get_parameter(self, section: str, parameter: str) -> Any:
        """
        Get a specific parameter value
        
        Args:
            section: Configuration section name
            parameter: Parameter name
            
        Returns:
            Parameter value
        """
        try:
            section_obj = getattr(self, section)
            return getattr(section_obj, parameter)
            
        except AttributeError:
            logger.error(f"Section '{section}' or parameter '{parameter}' not found")
            return None
    
    def create_preset(self, preset_name: str, description: str = "") -> None:
        """
        Create a configuration preset
        
        Args:
            preset_name: Name of the preset
            description: Description of the preset
        """
        preset_file = f"nifty50_preset_{preset_name}.json"
        
        config_data = {
            'trading': asdict(self.trading),
            'market': asdict(self.market),
            'data': asdict(self.data),
            'technical': asdict(self.technical),
            'model': asdict(self.model),
            'analyst': asdict(self.analyst),
            'risk': asdict(self.risk),
            'backtest': asdict(self.backtest),
            'metadata': {
                'preset_name': preset_name,
                'description': description,
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        }
        
        try:
            with open(preset_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Preset '{preset_name}' saved to {preset_file}")
            
        except Exception as e:
            logger.error(f"Error saving preset: {e}")
    
    def load_preset(self, preset_name: str) -> None:
        """
        Load a configuration preset
        
        Args:
            preset_name: Name of the preset to load
        """
        preset_file = f"nifty50_preset_{preset_name}.json"
        
        if os.path.exists(preset_file):
            self.load_config(preset_file)
            logger.info(f"Loaded preset '{preset_name}'")
        else:
            logger.error(f"Preset '{preset_name}' not found")
    
    def list_presets(self) -> list:
        """
        List available configuration presets
        
        Returns:
            List of available preset names
        """
        presets = []
        for file in os.listdir('.'):
            if file.startswith('nifty50_preset_') and file.endswith('.json'):
                preset_name = file.replace('nifty50_preset_', '').replace('.json', '')
                presets.append(preset_name)
        
        return presets
    
    def print_config(self) -> None:
        """Print current configuration"""
        print("\n" + "="*50)
        print("NIFTY50 TRADING CONFIGURATION")
        print("="*50)
        
        print(f"\nTRADING PARAMETERS:")
        print(f"  Starting Capital: ₹{self.trading.starting_capital:,.2f}")
        print(f"  Price Window: {self.trading.price_window} days")
        print(f"  Reflection Window: {self.trading.reflection_window} trades")
        print(f"  Max Position Size: {self.trading.max_position_size:.1%}")
        print(f"  Min Cash Reserve: {self.trading.min_cash_reserve:.1%}")
        
        print(f"\nMARKET PARAMETERS:")
        print(f"  Trading Hours: {self.market.trading_hours_start} - {self.market.trading_hours_end}")
        print(f"  Total Transaction Cost: {self.get_total_transaction_cost():.4%}")
        
        print(f"\nDATA PARAMETERS:")
        print(f"  Data Directory: {self.data.data_dir}")
        print(f"  News Directory: {self.data.news_dir}")
        print(f"  Price Data Source: {self.data.price_data_source}")
        
        print(f"\nANALYST CONFIG:")
        print(f"  Use Technical: {self.analyst.use_technical}")
        print(f"  Use News: {self.analyst.use_news}")
        print(f"  Use Reflection: {self.analyst.use_reflection}")
        print(f"  Technical Weight: {self.analyst.technical_weight:.1%}")
        print(f"  News Weight: {self.analyst.news_weight:.1%}")
        print(f"  Reflection Weight: {self.analyst.reflection_weight:.1%}")
        
        print(f"\nMODEL PARAMETERS:")
        print(f"  Model: {self.model.model_name}")
        print(f"  Temperature: {self.model.temperature}")
        print(f"  Seed: {self.model.seed}")
        
        print(f"\nRISK MANAGEMENT:")
        print(f"  Max Daily Loss: {self.risk.max_daily_loss:.1%}")
        print(f"  Max Weekly Loss: {self.risk.max_weekly_loss:.1%}")
        print(f"  Max Monthly Loss: {self.risk.max_monthly_loss:.1%}")
        
        print("="*50)

# Predefined configuration presets
def create_conservative_preset() -> NIFTY50Config:
    """Create a conservative trading configuration"""
    config = NIFTY50Config()
    config.trading.max_position_size = 0.6
    config.trading.min_cash_reserve = 0.2
    config.trading.stop_loss_threshold = 0.03
    config.trading.take_profit_threshold = 0.10
    config.analyst.min_confidence_threshold = 0.7
    config.risk.max_daily_loss = 0.01
    config.risk.max_weekly_loss = 0.03
    config.risk.max_monthly_loss = 0.08
    return config

def create_aggressive_preset() -> NIFTY50Config:
    """Create an aggressive trading configuration"""
    config = NIFTY50Config()
    config.trading.max_position_size = 0.9
    config.trading.min_cash_reserve = 0.05
    config.trading.stop_loss_threshold = 0.08
    config.trading.take_profit_threshold = 0.20
    config.analyst.min_confidence_threshold = 0.3
    config.risk.max_daily_loss = 0.03
    config.risk.max_weekly_loss = 0.08
    config.risk.max_monthly_loss = 0.15
    return config

def create_balanced_preset() -> NIFTY50Config:
    """Create a balanced trading configuration"""
    config = NIFTY50Config()
    # Uses default values which are already balanced
    return config

# Example usage
if __name__ == "__main__":
    # Create default configuration
    config = NIFTY50Config()
    
    # Print current configuration
    config.print_config()
    
    # Validate configuration
    if config.validate_config():
        print("\nConfiguration is valid!")
    
    # Save configuration
    config.save_config()
    
    # Create and save presets
    conservative_config = create_conservative_preset()
    conservative_config.create_preset("conservative", "Conservative trading strategy with low risk")
    
    aggressive_config = create_aggressive_preset()
    aggressive_config.create_preset("aggressive", "Aggressive trading strategy with high risk-reward")
    
    balanced_config = create_balanced_preset()
    balanced_config.create_preset("balanced", "Balanced trading strategy for moderate risk")
    
    # List available presets
    print(f"\nAvailable presets: {config.list_presets()}")