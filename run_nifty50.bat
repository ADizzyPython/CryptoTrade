@echo off
REM NIFTY50 Trading System - Windows Batch Script
REM Simple batch interface for NIFTY50 trading system

echo.
echo =====================================================
echo   NIFTY50 TRADING SYSTEM
echo =====================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Parse command line arguments
if "%1"=="" goto :help
if "%1"=="help" goto :help
if "%1"=="run" goto :run
if "%1"=="batch" goto :batch
if "%1"=="compare" goto :compare
if "%1"=="config" goto :config
if "%1"=="demo" goto :demo
if "%1"=="setup" goto :setup

echo Unknown command: %1
goto :help

:run
echo Running single trading session...
python run_nifty50_agent.py %2 %3 %4 %5 %6 %7 %8 %9
goto :end

:batch
echo Running batch execution...
python run_nifty50_batch.py %2 %3 %4 %5 %6 %7 %8 %9
goto :end

:compare
echo Running strategy comparison...
python nifty50_strategy_comparison.py
goto :end

:config
echo Configuration Management
echo.
echo Available presets:
echo   - conservative: Low risk, conservative trading
echo   - balanced: Moderate risk, balanced approach  
echo   - aggressive: High risk, aggressive trading
echo.
echo Running configuration demo...
python demo_nifty50_config.py
goto :end

:demo
echo Running NIFTY50 System Demonstration
echo.
echo 1. Configuration Demo:
python demo_nifty50_config.py
echo.
echo 2. Strategy Comparison Demo:
python nifty50_strategy_comparison.py
echo.
echo 3. Sample Trading Session (Dry Run):
python run_nifty50_agent.py --start-date 2024-01-01 --end-date 2024-01-31 --preset balanced --dry-run
goto :end

:setup
echo Setting up NIFTY50 Trading Environment
if not exist "data" mkdir data
if not exist "data\nifty50" mkdir data\nifty50
if not exist "data\selected_nifty50_202401_202501" mkdir data\selected_nifty50_202401_202501
if not exist "logs" mkdir logs
echo Environment setup completed.
goto :end

:help
echo Usage: %0 [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   run           - Run single trading session
echo   batch         - Run batch execution with multiple configurations
echo   compare       - Compare with baseline strategies
echo   config        - Configuration management
echo   demo          - Run demonstration examples
echo   setup         - Setup environment
echo   help          - Show this help message
echo.
echo Examples:
echo   %0 run --start-date 2024-01-01 --end-date 2024-03-31 --preset conservative
echo   %0 batch --with-baseline
echo   %0 compare
echo   %0 config
echo   %0 demo
echo.
echo For detailed options, use Python scripts directly:
echo   python run_nifty50_agent.py --help
echo   python run_nifty50_batch.py --help
goto :end

:end
echo.
pause