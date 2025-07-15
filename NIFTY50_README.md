# NIFTY50 Trading System

A comprehensive LLM-based trading system for the NIFTY50 index, built as an extension to the CryptoTrade framework. This system adapts the multi-analyst architecture from cryptocurrency trading to Indian stock market trading.

## Overview

The NIFTY50 Trading System provides:

- **Multi-Analyst Architecture**: Technical, News, and Reflection analysts
- **Configurable Trading Strategies**: Conservative, Balanced, and Aggressive presets
- **Comprehensive Backtesting**: Compare with traditional baseline strategies
- **Indian Market Specific**: Includes STT, brokerage fees, and NSE trading hours
- **Real-time Data Integration**: Uses yfinance for price data and news sentiment analysis

## Features

### 🔧 Core Components

- **Trading Agent** (`nifty50_agent.py`): Main trading orchestrator
- **Analysts** (`nifty50_analysts.py`): Technical, News, and Reflection analysis
- **Trading Environment** (`nifty50_env.py`): Simulated Indian market environment
- **Configuration System** (`nifty50_config.py`): Centralized parameter management
- **Data Utilities** (`nifty50_data_utils.py`): Data fetching and processing
- **Baseline Strategies** (`nifty50_baseline_strategies.py`): Traditional trading strategies

### 📊 Trading Strategies

**LLM-Based Strategies:**
- Conservative: Low risk, high confidence threshold
- Balanced: Moderate risk-reward balance
- Aggressive: High risk, high reward potential

**Baseline Strategies:**
- Buy and Hold
- Moving Average Crossover
- RSI Strategy
- MACD Strategy
- Bollinger Bands
- Momentum Strategy
- Mean Reversion Strategy

### 🎯 Indian Market Features

- **Transaction Costs**: Brokerage fees, STT, exchange fees, SEBI fees, GST
- **Trading Hours**: NSE trading hours (9:15 AM to 3:30 PM IST)
- **Currency**: All calculations in Indian Rupees (INR)
- **Market Holidays**: Indian market calendar awareness

## Installation

### Prerequisites

```bash
# Required Python packages
pip install pandas numpy yfinance openai matplotlib seaborn
```

### Setup

1. Clone the repository and navigate to the CryptoTrade directory
2. Set up your OpenAI API key in `utils.py`
3. Run the setup script:

```bash
./run_nifty50.sh setup
```

## Quick Start

### 1. Basic Trading Session

```bash
# Run with default settings (balanced preset, 2024 data)
./run_nifty50.sh run

# Run with custom settings
./run_nifty50.sh run --start-date 2024-01-01 --end-date 2024-03-31 --preset conservative
```

### 2. Batch Execution

```bash
# Run multiple configurations
./run_nifty50.sh batch

# Include baseline strategy comparison
./run_nifty50.sh batch --with-baseline
```

### 3. Strategy Comparison

```bash
# Compare LLM agents with baseline strategies
./run_nifty50.sh compare
```

### 4. Configuration Demo

```bash
# Explore configuration options
./run_nifty50.sh config --demo
```

## Configuration

### Configuration Presets

#### Conservative Preset
- **Max Position Size**: 60%
- **Min Cash Reserve**: 20%
- **Stop Loss**: 3%
- **Take Profit**: 10%
- **Confidence Threshold**: 70%

#### Balanced Preset (Default)
- **Max Position Size**: 80%
- **Min Cash Reserve**: 10%
- **Stop Loss**: 5%
- **Take Profit**: 15%
- **Confidence Threshold**: 50%

#### Aggressive Preset
- **Max Position Size**: 90%
- **Min Cash Reserve**: 5%
- **Stop Loss**: 8%
- **Take Profit**: 20%
- **Confidence Threshold**: 30%

### Custom Configuration

```python
from nifty50_config import NIFTY50Config

# Create custom configuration
config = NIFTY50Config()
config.trading.starting_capital = 500_000
config.trading.max_position_size = 0.7
config.analyst.technical_weight = 0.5
config.analyst.news_weight = 0.3
config.analyst.reflection_weight = 0.2

# Save configuration
config.save_config("my_config.json")
```

## Usage Examples

### Python API

```python
from nifty50_agent import NIFTY50TradingAgent, create_nifty50_args
from nifty50_config import create_balanced_preset
from utils import get_chat

# Create configuration
config = create_balanced_preset()

# Create arguments
args = create_nifty50_args(
    starting_date="2024-01-01",
    ending_date="2024-12-31",
    config=config
)

# Create LLM function
def llm_function(prompt, model, seed):
    return get_chat(prompt, model, seed)

# Initialize and run agent
agent = NIFTY50TradingAgent(args, llm_function, config)
results = agent.run_trading_session()

# Print results
print(f"Total Return: {results['performance_metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.4f}")
```

### Command Line Interface

```bash
# Basic usage
python run_nifty50_agent.py --start-date 2024-01-01 --end-date 2024-12-31

# Advanced usage
python run_nifty50_agent.py \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --preset aggressive \
    --model "openai/gpt-4o" \
    --no-news \
    --output results.json
```

## Data Requirements

### Price Data
- **Source**: Yahoo Finance via yfinance
- **Symbol**: ^NSEI (NIFTY50 index)
- **Frequency**: Daily OHLCV data
- **Auto-fetch**: System automatically fetches required data

### News Data
- **Directory**: `data/selected_nifty50_202401_202501/`
- **Format**: JSON files named as `YYYY-MM-DD.json`
- **Structure**: 
```json
{
  "organic": [
    {
      "title": "News headline",
      "description": "News content",
      "date": "Jan 4, 2024",
      "link": "https://example.com"
    }
  ]
}
```

## Performance Metrics

The system tracks comprehensive performance metrics:

- **Total Return**: Overall percentage return
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of trading actions taken

## Architecture

### Multi-Analyst System

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Technical      │    │  News           │    │  Reflection     │
│  Analyst        │    │  Analyst        │    │  Analyst        │
│                 │    │                 │    │                 │
│ • SMA/EMA       │    │ • Sentiment     │    │ • Past          │
│ • MACD          │    │ • Market Events │    │   Performance   │
│ • RSI           │    │ • Economic Data │    │ • Strategy      │
│ • Bollinger     │    │ • Policy News   │    │   Adjustment    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Trading        │
                    │  Coordinator    │
                    │                 │
                    │ • Weighted      │
                    │   Decision      │
                    │ • Risk          │
                    │   Management    │
                    │ • Execution     │
                    └─────────────────┘
```

### Data Flow

```
Market Data → Data Manager → Trading Environment → Analysts → Coordinator → Action
     ↑                                                                         ↓
News Data  ←────────────────────────────────────────────────────────→ Portfolio
```

## Comparison with Baseline Strategies

Run comprehensive comparisons to evaluate the LLM-based approach:

```bash
# Compare all strategies
./run_nifty50.sh compare

# Results include:
# - Performance metrics for each strategy
# - Risk-return analysis
# - Statistical significance tests
# - Visualization plots
```

## Transaction Costs

The system includes realistic Indian market transaction costs:

- **Brokerage Fee**: 0.03% per transaction
- **Securities Transaction Tax (STT)**: 0.1% on equity delivery
- **Exchange Fee**: 0.00345% (NSE)
- **SEBI Fee**: 0.0001%
- **GST**: 18% on brokerage
- **Total**: ~0.11% per transaction

## Risk Management

Built-in risk management features:

- **Position Sizing**: Configurable maximum position limits
- **Stop Loss**: Automatic loss prevention
- **Take Profit**: Profit booking mechanisms
- **Cash Reserve**: Minimum cash requirements
- **Volatility Adjustment**: Dynamic position sizing based on market volatility

## Logging and Monitoring

- **Comprehensive Logs**: All trading decisions and reasoning
- **Performance Tracking**: Real-time portfolio monitoring
- **Error Handling**: Robust error recovery mechanisms
- **Audit Trail**: Complete record of all trading activities

## File Structure

```
CryptoTrade/
├── nifty50_agent.py              # Main trading agent
├── nifty50_analysts.py           # Analyst implementations
├── nifty50_env.py                # Trading environment
├── nifty50_config.py             # Configuration system
├── nifty50_data_utils.py         # Data utilities
├── nifty50_baseline_strategies.py # Baseline strategies
├── nifty50_strategy_comparison.py # Strategy comparison
├── run_nifty50_agent.py          # Single execution script
├── run_nifty50_batch.py          # Batch execution script
├── run_nifty50.sh                # Shell script interface
├── demo_nifty50_config.py        # Configuration demo
├── NIFTY50_README.md             # This file
└── data/
    ├── nifty50/                  # Price data
    └── selected_nifty50_202401_202501/ # News data
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure OpenAI API key is properly configured in `utils.py`
2. **Data Not Found**: Run `./run_nifty50.sh setup` to create sample data
3. **Memory Issues**: Reduce batch size or date range for large datasets
4. **Network Errors**: Check internet connection for data fetching

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

- **Caching**: Implement data caching for repeated runs
- **Parallel Processing**: Use multiprocessing for batch executions
- **Memory Management**: Efficient data handling for large datasets
- **API Rate Limiting**: Respect OpenAI API rate limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the same terms as the original CryptoTrade project (CC BY-NC-SA).

## Disclaimer

This system is for educational and research purposes only. Past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.

## Support

For questions and support:
- Check the troubleshooting section
- Review the configuration documentation
- Examine the example scripts
- Use the demo mode to understand the system

## Future Enhancements

- **Real-time Trading**: Live market integration
- **Options Trading**: Extend to NIFTY50 options
- **Sector Analysis**: Individual stock analysis
- **Risk Models**: Advanced risk modeling
- **Performance Attribution**: Detailed performance analysis
- **Mobile Dashboard**: Real-time monitoring interface

---

*Built with ❤️ for the Indian trading community*