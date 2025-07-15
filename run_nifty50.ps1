# NIFTY50 Trading System - PowerShell Script
# Windows PowerShell execution script for NIFTY50 trading system

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [string]$StartDate = "2024-01-01",
    [string]$EndDate = "2024-12-31",
    [string]$Preset = "balanced",
    [string]$Model = "openai/gpt-4o",
    [int]$Capital = 1000000,
    [switch]$NoTech,
    [switch]$NoNews,
    [switch]$NoReflection,
    [switch]$DryRun,
    [switch]$WithBaseline,
    [string]$Output,
    [string]$ConfigFile
)

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Blue"
$Cyan = "Cyan"

function Write-ColoredOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Show-Header {
    Write-Host ""
    Write-ColoredOutput "=====================================================" $Blue
    Write-ColoredOutput "  NIFTY50 TRADING SYSTEM" $Blue
    Write-ColoredOutput "=====================================================" $Blue
    Write-Host ""
}

function Show-Usage {
    Write-Host "NIFTY50 Trading System - PowerShell Interface"
    Write-Host ""
    Write-Host "Usage: .\run_nifty50.ps1 [COMMAND] [OPTIONS]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  run           - Run single trading session"
    Write-Host "  batch         - Run batch execution with multiple configurations"
    Write-Host "  compare       - Compare with baseline strategies"
    Write-Host "  config        - Configuration management"
    Write-Host "  demo          - Run demonstration examples"
    Write-Host "  help          - Show this help message"
    Write-Host ""
    Write-Host "Options for 'run' command:"
    Write-Host "  -StartDate DATE       Start date (YYYY-MM-DD) [default: 2024-01-01]"
    Write-Host "  -EndDate DATE         End date (YYYY-MM-DD) [default: 2024-12-31]"
    Write-Host "  -Preset PRESET        Configuration preset: conservative|balanced|aggressive [default: balanced]"
    Write-Host "  -Model MODEL          LLM model to use [default: openai/gpt-4o]"
    Write-Host "  -Capital AMOUNT       Starting capital in INR [default: 1000000]"
    Write-Host "  -NoTech               Disable technical analysis"
    Write-Host "  -NoNews               Disable news analysis"
    Write-Host "  -NoReflection         Disable reflection analysis"
    Write-Host "  -DryRun               Run in dry-run mode"
    Write-Host "  -Output FILE          Output file path"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run_nifty50.ps1 run -StartDate 2024-01-01 -EndDate 2024-03-31 -Preset conservative"
    Write-Host "  .\run_nifty50.ps1 batch -WithBaseline"
    Write-Host "  .\run_nifty50.ps1 compare"
    Write-Host "  .\run_nifty50.ps1 config"
    Write-Host "  .\run_nifty50.ps1 demo"
}

function Test-Dependencies {
    Write-ColoredOutput "Checking dependencies..." $Yellow
    
    # Check if Python is available
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColoredOutput "Python found: $pythonVersion" $Green
        } else {
            throw "Python not found"
        }
    } catch {
        Write-ColoredOutput "Error: Python 3 is required but not installed." $Red
        Write-ColoredOutput "Please install Python from https://python.org" $Yellow
        exit 1
    }
    
    # Check if required packages are available
    try {
        python -c "import pandas, numpy, yfinance, openai" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-ColoredOutput "Required packages found." $Green
        } else {
            throw "Required packages not found"
        }
    } catch {
        Write-ColoredOutput "Error: Required Python packages not found." $Red
        Write-ColoredOutput "Please install requirements:" $Yellow
        Write-ColoredOutput "pip install pandas numpy yfinance openai matplotlib seaborn" $Cyan
        exit 1
    }
    
    Write-ColoredOutput "Dependencies check passed." $Green
}

function Invoke-SingleRun {
    Write-ColoredOutput "Starting single trading session..." $Blue
    
    # Build Python arguments
    $pythonArgs = @(
        "--start-date", $StartDate,
        "--end-date", $EndDate,
        "--preset", $Preset,
        "--model", $Model
    )
    
    if ($NoTech) { $pythonArgs += "--no-tech" }
    if ($NoNews) { $pythonArgs += "--no-news" }
    if ($NoReflection) { $pythonArgs += "--no-reflection" }
    if ($DryRun) { $pythonArgs += "--dry-run" }
    if ($Output) { $pythonArgs += "--output", $Output }
    
    Write-ColoredOutput "Configuration:" $Yellow
    Write-ColoredOutput "  Start Date: $StartDate" $Yellow
    Write-ColoredOutput "  End Date: $EndDate" $Yellow
    Write-ColoredOutput "  Preset: $Preset" $Yellow
    Write-ColoredOutput "  Model: $Model" $Yellow
    Write-ColoredOutput "  Capital: ₹$Capital" $Yellow
    Write-Host ""
    
    # Run the trading agent
    python run_nifty50_agent.py @pythonArgs
}

function Invoke-BatchRun {
    Write-ColoredOutput "Starting batch execution..." $Blue
    
    # Build Python arguments
    $pythonArgs = @(
        "--start-date", $StartDate,
        "--end-date", $EndDate,
        "--capital", $Capital
    )
    
    if ($WithBaseline) { $pythonArgs += "--with-baseline" }
    if ($Output) { $pythonArgs += "--output", $Output }
    if ($ConfigFile) { $pythonArgs += "--config-file", $ConfigFile }
    
    Write-ColoredOutput "Batch Configuration:" $Yellow
    Write-ColoredOutput "  Start Date: $StartDate" $Yellow
    Write-ColoredOutput "  End Date: $EndDate" $Yellow
    Write-ColoredOutput "  Capital: ₹$Capital" $Yellow
    if ($WithBaseline) { Write-ColoredOutput "  Include Baseline: Yes" $Yellow }
    Write-Host ""
    
    # Run batch execution
    python run_nifty50_batch.py @pythonArgs
}

function Invoke-Compare {
    Write-ColoredOutput "Running strategy comparison..." $Blue
    python nifty50_strategy_comparison.py
}

function Invoke-Config {
    Write-ColoredOutput "Configuration Management" $Blue
    
    Write-ColoredOutput "Available configuration presets:" $Yellow
    Write-Host "  - conservative: Low risk, conservative trading"
    Write-Host "  - balanced: Moderate risk, balanced approach"
    Write-Host "  - aggressive: High risk, aggressive trading"
    Write-Host ""
    
    Write-ColoredOutput "Running configuration demo..." $Yellow
    python demo_nifty50_config.py
}

function Invoke-Demo {
    Write-ColoredOutput "Running NIFTY50 System Demonstration" $Blue
    
    Write-ColoredOutput "1. Configuration Demo:" $Yellow
    python demo_nifty50_config.py
    
    Write-Host ""
    Write-ColoredOutput "2. Strategy Comparison Demo:" $Yellow
    python nifty50_strategy_comparison.py
    
    Write-Host ""
    Write-ColoredOutput "3. Sample Trading Session (Dry Run):" $Yellow
    python run_nifty50_agent.py --start-date 2024-01-01 --end-date 2024-01-31 --preset balanced --dry-run
}

function Initialize-Environment {
    Write-ColoredOutput "Setting up NIFTY50 Trading Environment" $Blue
    
    # Create necessary directories
    if (!(Test-Path "data")) { New-Item -ItemType Directory -Path "data" }
    if (!(Test-Path "data/nifty50")) { New-Item -ItemType Directory -Path "data/nifty50" }
    if (!(Test-Path "data/selected_nifty50_202401_202501")) { New-Item -ItemType Directory -Path "data/selected_nifty50_202401_202501" }
    if (!(Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" }
    
    # Create sample data if it doesn't exist
    if (!(Test-Path "data/nifty50/sample_data.csv")) {
        Write-ColoredOutput "Creating sample data..." $Yellow
        python -c @"
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
"@
    }
    
    Write-ColoredOutput "Environment setup completed." $Green
}

# Main execution
Show-Header

switch ($Command.ToLower()) {
    "run" {
        Test-Dependencies
        Invoke-SingleRun
    }
    "batch" {
        Test-Dependencies
        Invoke-BatchRun
    }
    "compare" {
        Test-Dependencies
        Invoke-Compare
    }
    "config" {
        Invoke-Config
    }
    "demo" {
        Test-Dependencies
        Invoke-Demo
    }
    "setup" {
        Initialize-Environment
    }
    "help" {
        Show-Usage
    }
    default {
        Write-ColoredOutput "Unknown command: $Command" $Red
        Show-Usage
        exit 1
    }
}