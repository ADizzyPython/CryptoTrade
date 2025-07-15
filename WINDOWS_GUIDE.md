# NIFTY50 Trading System - Windows PowerShell Guide

## Prerequisites

### 1. Install Python
- Download Python from https://python.org
- Make sure to check "Add Python to PATH" during installation
- Verify installation: `python --version`

### 2. Install Required Packages
```powershell
pip install pandas numpy yfinance openai matplotlib seaborn
```

### 3. Set up OpenAI API Key
- Edit `utils.py` and add your OpenAI API key
- Or set environment variable: `$env:OPENAI_API_KEY = "your-api-key"`

## Running the System

### Option 1: PowerShell Script (Recommended)

```powershell
# Set execution policy (run once)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run commands
.\run_nifty50.ps1 help
.\run_nifty50.ps1 config
.\run_nifty50.ps1 run -StartDate 2024-01-01 -EndDate 2024-03-31 -Preset balanced
.\run_nifty50.ps1 batch -WithBaseline
.\run_nifty50.ps1 compare
.\run_nifty50.ps1 demo
```

### Option 2: Batch File

```cmd
# Run from Command Prompt
run_nifty50.bat help
run_nifty50.bat config
run_nifty50.bat run --start-date 2024-01-01 --end-date 2024-03-31 --preset balanced
run_nifty50.bat batch --with-baseline
run_nifty50.bat compare
run_nifty50.bat demo
```

### Option 3: Direct Python Execution

```powershell
# Single trading session
python run_nifty50_agent.py --start-date 2024-01-01 --end-date 2024-12-31 --preset balanced

# Batch execution
python run_nifty50_batch.py --start-date 2024-01-01 --end-date 2024-12-31 --with-baseline

# Configuration demo
python demo_nifty50_config.py

# Strategy comparison
python nifty50_strategy_comparison.py
```

## PowerShell Commands

### Basic Usage

```powershell
# Show help
.\run_nifty50.ps1 help

# Run configuration demo
.\run_nifty50.ps1 config

# Setup environment
.\run_nifty50.ps1 setup
```

### Trading Sessions

```powershell
# Basic run with balanced preset
.\run_nifty50.ps1 run

# Conservative trading
.\run_nifty50.ps1 run -Preset conservative -StartDate 2024-01-01 -EndDate 2024-03-31

# Aggressive trading with custom settings
.\run_nifty50.ps1 run -Preset aggressive -Capital 500000 -NoNews

# Dry run (no actual execution)
.\run_nifty50.ps1 run -DryRun -StartDate 2024-01-01 -EndDate 2024-01-31
```

### Batch Execution

```powershell
# Run multiple configurations
.\run_nifty50.ps1 batch

# Include baseline strategy comparison
.\run_nifty50.ps1 batch -WithBaseline

# Custom date range
.\run_nifty50.ps1 batch -StartDate 2024-01-01 -EndDate 2024-06-30 -WithBaseline
```

### Analysis

```powershell
# Compare strategies
.\run_nifty50.ps1 compare

# Run demonstrations
.\run_nifty50.ps1 demo
```

## Parameter Reference

### PowerShell Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `-StartDate` | Start date (YYYY-MM-DD) | 2024-01-01 | `-StartDate 2024-01-01` |
| `-EndDate` | End date (YYYY-MM-DD) | 2024-12-31 | `-EndDate 2024-12-31` |
| `-Preset` | Configuration preset | balanced | `-Preset conservative` |
| `-Model` | LLM model | openai/gpt-4o | `-Model "openai/gpt-4o"` |
| `-Capital` | Starting capital (INR) | 1000000 | `-Capital 500000` |
| `-NoTech` | Disable technical analysis | false | `-NoTech` |
| `-NoNews` | Disable news analysis | false | `-NoNews` |
| `-NoReflection` | Disable reflection analysis | false | `-NoReflection` |
| `-DryRun` | Dry run mode | false | `-DryRun` |
| `-WithBaseline` | Include baseline comparison | false | `-WithBaseline` |
| `-Output` | Output file path | auto | `-Output "results.json"` |

### Configuration Presets

1. **Conservative** (`-Preset conservative`)
   - Max Position Size: 60%
   - Min Cash Reserve: 20%
   - Stop Loss: 3%
   - Take Profit: 10%

2. **Balanced** (`-Preset balanced`)
   - Max Position Size: 80%
   - Min Cash Reserve: 10%
   - Stop Loss: 5%
   - Take Profit: 15%

3. **Aggressive** (`-Preset aggressive`)
   - Max Position Size: 90%
   - Min Cash Reserve: 5%
   - Stop Loss: 8%
   - Take Profit: 20%

## Example Workflows

### 1. Quick Start
```powershell
# First time setup
.\run_nifty50.ps1 setup

# Run configuration demo to understand the system
.\run_nifty50.ps1 config

# Run a sample trading session
.\run_nifty50.ps1 run -StartDate 2024-01-01 -EndDate 2024-01-31 -DryRun
```

### 2. Full Analysis
```powershell
# Run comprehensive comparison
.\run_nifty50.ps1 batch -WithBaseline -StartDate 2024-01-01 -EndDate 2024-06-30

# Compare specific strategies
.\run_nifty50.ps1 compare
```

### 3. Custom Configuration
```powershell
# Create custom configuration
python -c "
from nifty50_config import NIFTY50Config
config = NIFTY50Config()
config.trading.starting_capital = 500000
config.trading.max_position_size = 0.7
config.save_config('my_config.json')
"

# Use custom configuration
python run_nifty50_agent.py --config my_config.json --start-date 2024-01-01 --end-date 2024-12-31
```

## Troubleshooting

### Common Issues

1. **Execution Policy Error**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **Python Not Found**
   - Reinstall Python with "Add to PATH" option
   - Or use full path: `C:\Python39\python.exe`

3. **Package Import Errors**
   ```powershell
   pip install --upgrade pandas numpy yfinance openai matplotlib seaborn
   ```

4. **API Key Issues**
   - Check `utils.py` for correct API key
   - Verify API key is valid and has credits

### Debug Mode

```powershell
# Enable verbose output
$env:PYTHONPATH = "."
python run_nifty50_agent.py --start-date 2024-01-01 --end-date 2024-01-31 --dry-run
```

## File Structure

```
CryptoTrade/
├── run_nifty50.ps1           # PowerShell script
├── run_nifty50.bat           # Batch file
├── run_nifty50_agent.py      # Single execution
├── run_nifty50_batch.py      # Batch execution
├── demo_nifty50_config.py    # Configuration demo
├── nifty50_*.py              # Core system files
└── data/
    ├── nifty50/              # Price data
    └── selected_nifty50_202401_202501/ # News data
```

## Performance Tips

1. **Use shorter date ranges for faster execution**
2. **Enable only necessary analysts** (use -NoTech, -NoNews, -NoReflection)
3. **Use dry-run mode for testing**
4. **Monitor API usage and costs**

## Next Steps

1. Run the configuration demo: `.\run_nifty50.ps1 config`
2. Try a dry run: `.\run_nifty50.ps1 run -DryRun`
3. Execute a real trading session: `.\run_nifty50.ps1 run`
4. Compare with baselines: `.\run_nifty50.ps1 batch -WithBaseline`

---

*Happy Trading! 🚀*