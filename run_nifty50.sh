#!/bin/bash

# NIFTY50 Trading System Execution Script
# This script provides easy access to all NIFTY50 trading functionalities

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
START_DATE="2024-01-01"
END_DATE="2024-12-31"
PRESET="balanced"
MODEL="openai/gpt-4o"
CAPITAL=1000000

# Function to print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print header
print_header() {
    echo ""
    print_colored $BLUE "====================================================="
    print_colored $BLUE "  NIFTY50 TRADING SYSTEM"
    print_colored $BLUE "====================================================="
    echo ""
}

# Function to print usage
print_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  run           - Run single trading session"
    echo "  batch         - Run batch execution with multiple configurations"
    echo "  compare       - Compare with baseline strategies"
    echo "  config        - Configuration management"
    echo "  demo          - Run demonstration examples"
    echo "  help          - Show this help message"
    echo ""
    echo "Options for 'run' command:"
    echo "  --start-date DATE     Start date (YYYY-MM-DD) [default: $START_DATE]"
    echo "  --end-date DATE       End date (YYYY-MM-DD) [default: $END_DATE]"
    echo "  --preset PRESET       Configuration preset: conservative|balanced|aggressive [default: $PRESET]"
    echo "  --model MODEL         LLM model to use [default: $MODEL]"
    echo "  --capital AMOUNT      Starting capital in INR [default: $CAPITAL]"
    echo "  --no-tech             Disable technical analysis"
    echo "  --no-news             Disable news analysis"
    echo "  --no-reflection       Disable reflection analysis"
    echo "  --dry-run             Run in dry-run mode"
    echo "  --output FILE         Output file path"
    echo ""
    echo "Examples:"
    echo "  $0 run --start-date 2024-01-01 --end-date 2024-03-31 --preset conservative"
    echo "  $0 batch --with-baseline"
    echo "  $0 compare"
    echo "  $0 config --demo"
    echo "  $0 demo"
}

# Function to check dependencies
check_dependencies() {
    print_colored $YELLOW "Checking dependencies..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_colored $RED "Error: Python 3 is required but not installed."
        exit 1
    fi
    
    # Check if required Python packages are available
    python3 -c "import pandas, numpy, yfinance, openai" 2>/dev/null || {
        print_colored $RED "Error: Required Python packages not found."
        print_colored $YELLOW "Please install requirements: pip install pandas numpy yfinance openai"
        exit 1
    }
    
    print_colored $GREEN "Dependencies check passed."
}

# Function to run single trading session
run_single() {
    print_colored $BLUE "Starting single trading session..."
    
    local python_args="--start-date $START_DATE --end-date $END_DATE --preset $PRESET --model $MODEL"
    
    # Parse additional arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --start-date)
                START_DATE="$2"
                python_args=$(echo "$python_args" | sed "s/--start-date [^ ]*/--start-date $2/")
                shift 2
                ;;
            --end-date)
                END_DATE="$2"
                python_args=$(echo "$python_args" | sed "s/--end-date [^ ]*/--end-date $2/")
                shift 2
                ;;
            --preset)
                PRESET="$2"
                python_args=$(echo "$python_args" | sed "s/--preset [^ ]*/--preset $2/")
                shift 2
                ;;
            --model)
                MODEL="$2"
                python_args=$(echo "$python_args" | sed "s/--model [^ ]*/--model $2/")
                shift 2
                ;;
            --capital)
                CAPITAL="$2"
                python_args="$python_args --capital $2"
                shift 2
                ;;
            --no-tech)
                python_args="$python_args --no-tech"
                shift
                ;;
            --no-news)
                python_args="$python_args --no-news"
                shift
                ;;
            --no-reflection)
                python_args="$python_args --no-reflection"
                shift
                ;;
            --dry-run)
                python_args="$python_args --dry-run"
                shift
                ;;
            --output)
                python_args="$python_args --output $2"
                shift 2
                ;;
            *)
                print_colored $RED "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    print_colored $YELLOW "Configuration:"
    print_colored $YELLOW "  Start Date: $START_DATE"
    print_colored $YELLOW "  End Date: $END_DATE"
    print_colored $YELLOW "  Preset: $PRESET"
    print_colored $YELLOW "  Model: $MODEL"
    print_colored $YELLOW "  Capital: ₹$CAPITAL"
    echo ""
    
    # Run the trading agent
    python3 run_nifty50_agent.py $python_args
}

# Function to run batch execution
run_batch() {
    print_colored $BLUE "Starting batch execution..."
    
    local python_args="--start-date $START_DATE --end-date $END_DATE --capital $CAPITAL"
    
    # Parse additional arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --with-baseline)
                python_args="$python_args --with-baseline"
                shift
                ;;
            --start-date)
                START_DATE="$2"
                python_args=$(echo "$python_args" | sed "s/--start-date [^ ]*/--start-date $2/")
                shift 2
                ;;
            --end-date)
                END_DATE="$2"
                python_args=$(echo "$python_args" | sed "s/--end-date [^ ]*/--end-date $2/")
                shift 2
                ;;
            --capital)
                CAPITAL="$2"
                python_args=$(echo "$python_args" | sed "s/--capital [^ ]*/--capital $2/")
                shift 2
                ;;
            --output)
                python_args="$python_args --output $2"
                shift 2
                ;;
            *)
                print_colored $RED "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    print_colored $YELLOW "Batch Configuration:"
    print_colored $YELLOW "  Start Date: $START_DATE"
    print_colored $YELLOW "  End Date: $END_DATE"
    print_colored $YELLOW "  Capital: ₹$CAPITAL"
    echo ""
    
    # Run batch execution
    python3 run_nifty50_batch.py $python_args
}

# Function to run comparison with baseline strategies
run_compare() {
    print_colored $BLUE "Running strategy comparison..."
    
    python3 nifty50_strategy_comparison.py
}

# Function to manage configurations
manage_config() {
    print_colored $BLUE "Configuration Management"
    
    case $1 in
        --demo)
            print_colored $YELLOW "Running configuration demo..."
            python3 demo_nifty50_config.py
            ;;
        --list)
            print_colored $YELLOW "Available configuration presets:"
            echo "  - conservative: Low risk, conservative trading"
            echo "  - balanced: Moderate risk, balanced approach"
            echo "  - aggressive: High risk, aggressive trading"
            ;;
        --create)
            print_colored $YELLOW "Creating new configuration..."
            python3 -c "
from nifty50_config import NIFTY50Config
config = NIFTY50Config()
config.print_config()
config.save_config('custom_config.json')
print('Configuration saved to custom_config.json')
"
            ;;
        *)
            print_colored $YELLOW "Configuration options:"
            echo "  --demo    Run configuration demonstration"
            echo "  --list    List available presets"
            echo "  --create  Create new configuration file"
            ;;
    esac
}

# Function to run demonstration examples
run_demo() {
    print_colored $BLUE "Running NIFTY50 System Demonstration"
    
    print_colored $YELLOW "1. Configuration Demo:"
    python3 demo_nifty50_config.py
    
    echo ""
    print_colored $YELLOW "2. Strategy Comparison Demo:"
    python3 nifty50_strategy_comparison.py
    
    echo ""
    print_colored $YELLOW "3. Sample Trading Session (Dry Run):"
    python3 run_nifty50_agent.py --start-date 2024-01-01 --end-date 2024-01-31 --preset balanced --dry-run
}

# Function to setup environment
setup_environment() {
    print_colored $BLUE "Setting up NIFTY50 Trading Environment"
    
    # Create necessary directories
    mkdir -p data/nifty50
    mkdir -p data/selected_nifty50_202401_202501
    mkdir -p logs
    
    # Create sample data if it doesn't exist
    if [ ! -f "data/nifty50/sample_data.csv" ]; then
        print_colored $YELLOW "Creating sample data..."
        python3 -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample NIFTY50 data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
prices = 20000 + np.cumsum(np.random.randn(100) * 50)

df = pd.DataFrame({
    'timestamp': dates,
    'open': prices,
    'high': prices * 1.02,
    'low': prices * 0.98,
    'close': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
    'volume': np.random.randint(1000000, 10000000, 100),
    'name': '^NSEI'
})

df.to_csv('data/nifty50/sample_data.csv', index=False)
print('Sample data created')
"
    fi
    
    print_colored $GREEN "Environment setup completed."
}

# Main execution
main() {
    print_header
    
    # Check if command is provided
    if [ $# -eq 0 ]; then
        print_usage
        exit 1
    fi
    
    # Parse command
    case $1 in
        run)
            shift
            check_dependencies
            run_single "$@"
            ;;
        batch)
            shift
            check_dependencies
            run_batch "$@"
            ;;
        compare)
            shift
            check_dependencies
            run_compare "$@"
            ;;
        config)
            shift
            manage_config "$@"
            ;;
        demo)
            shift
            check_dependencies
            run_demo "$@"
            ;;
        setup)
            setup_environment
            ;;
        help|--help|-h)
            print_usage
            ;;
        *)
            print_colored $RED "Unknown command: $1"
            print_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"