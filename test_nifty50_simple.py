#!/usr/bin/env python3
"""
Simple NIFTY50 Test Script
Test the basic functionality without external dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test imports
try:
    from nifty50_config import NIFTY50Config
    print("✓ Configuration system imported successfully")
except ImportError as e:
    print(f"✗ Error importing configuration: {e}")
    sys.exit(1)

def test_configuration():
    """Test basic configuration functionality"""
    print("\n" + "="*50)
    print("TESTING CONFIGURATION SYSTEM")
    print("="*50)
    
    # Create default configuration
    config = NIFTY50Config()
    
    print(f"✓ Default configuration created")
    print(f"  Starting Capital: ₹{config.trading.starting_capital:,.2f}")
    print(f"  Max Position Size: {config.trading.max_position_size:.1%}")
    print(f"  Model: {config.model.model_name}")
    
    # Test validation
    is_valid = config.validate_config()
    print(f"✓ Configuration validation: {is_valid}")
    
    # Test transaction cost calculation
    total_cost = config.get_total_transaction_cost()
    print(f"✓ Total transaction cost: {total_cost:.4%}")
    
    # Test position sizing
    portfolio_value = 1_000_000
    max_position = config.get_position_size_limit(portfolio_value)
    print(f"✓ Max position for ₹{portfolio_value:,.2f}: ₹{max_position:,.2f}")
    
    return config

def test_data_availability():
    """Test if sample data is available"""
    print("\n" + "="*50)
    print("TESTING DATA AVAILABILITY")
    print("="*50)
    
    sample_data_path = "data/nifty50/sample_data.csv"
    
    if os.path.exists(sample_data_path):
        print(f"✓ Sample data found at: {sample_data_path}")
        
        # Try to read the file
        try:
            import pandas as pd
            df = pd.read_csv(sample_data_path)
            print(f"✓ Sample data loaded: {len(df)} rows")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Columns: {list(df.columns)}")
            return True
        except Exception as e:
            print(f"✗ Error reading sample data: {e}")
            return False
    else:
        print(f"✗ Sample data not found at: {sample_data_path}")
        return False

def test_basic_trading_setup():
    """Test basic trading setup without LLM"""
    print("\n" + "="*50)
    print("TESTING BASIC TRADING SETUP")
    print("="*50)
    
    try:
        from nifty50_config import NIFTY50Config
        from nifty50_agent import create_nifty50_args
        
        # Create configuration
        config = NIFTY50Config()
        
        # Create arguments
        args = create_nifty50_args(
            starting_date="2024-01-01",
            ending_date="2024-01-31",
            config=config
        )
        
        print(f"✓ Trading arguments created")
        print(f"  Dataset: {args.dataset}")
        print(f"  Date range: {args.starting_date.date()} to {args.ending_date.date()}")
        print(f"  Use technical: {args.use_tech}")
        print(f"  Use news: {args.use_news}")
        print(f"  Model: {args.model}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in basic trading setup: {e}")
        return False

def test_dummy_llm():
    """Test dummy LLM function"""
    print("\n" + "="*50)
    print("TESTING DUMMY LLM FUNCTION")
    print("="*50)
    
    try:
        from nifty50_agent import dummy_llm_function
        
        # Test dummy LLM
        response = dummy_llm_function("Test prompt", "gpt-4", 42)
        print(f"✓ Dummy LLM function works")
        print(f"  Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in dummy LLM: {e}")
        return False

def create_sample_news_data():
    """Create sample news data if missing"""
    print("\n" + "="*50)
    print("CREATING SAMPLE NEWS DATA")
    print("="*50)
    
    news_dir = "data/selected_nifty50_202401_202501"
    
    if not os.path.exists(news_dir):
        os.makedirs(news_dir)
        print(f"✓ Created news directory: {news_dir}")
    
    # Create sample news file
    sample_news_file = os.path.join(news_dir, "2024-01-01.json")
    
    if not os.path.exists(sample_news_file):
        import json
        sample_news = {
            "organic": [
                {
                    "title": "NIFTY50 Shows Strong Performance",
                    "description": "The NIFTY50 index demonstrated robust performance in today's trading session.",
                    "date": "Jan 1, 2024",
                    "link": "https://example.com/news1"
                },
                {
                    "title": "Market Outlook Positive",
                    "description": "Analysts predict continued growth in the Indian equity markets.",
                    "date": "Jan 1, 2024",
                    "link": "https://example.com/news2"
                }
            ]
        }
        
        with open(sample_news_file, 'w') as f:
            json.dump(sample_news, f, indent=2)
        
        print(f"✓ Created sample news file: {sample_news_file}")
        return True
    else:
        print(f"✓ Sample news file already exists: {sample_news_file}")
        return True

def main():
    """Run all tests"""
    print("NIFTY50 System - Simple Test")
    print("="*50)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Configuration
    try:
        test_configuration()
        tests_passed += 1
        print("✓ Configuration test passed")
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
    
    # Test 2: Data availability
    try:
        if test_data_availability():
            tests_passed += 1
            print("✓ Data availability test passed")
        else:
            print("✗ Data availability test failed")
    except Exception as e:
        print(f"✗ Data availability test failed: {e}")
    
    # Test 3: Basic trading setup
    try:
        if test_basic_trading_setup():
            tests_passed += 1
            print("✓ Basic trading setup test passed")
        else:
            print("✗ Basic trading setup test failed")
    except Exception as e:
        print(f"✗ Basic trading setup test failed: {e}")
    
    # Test 4: Dummy LLM
    try:
        if test_dummy_llm():
            tests_passed += 1
            print("✓ Dummy LLM test passed")
        else:
            print("✗ Dummy LLM test failed")
    except Exception as e:
        print(f"✗ Dummy LLM test failed: {e}")
    
    # Test 5: Sample news data
    try:
        if create_sample_news_data():
            tests_passed += 1
            print("✓ Sample news data test passed")
        else:
            print("✗ Sample news data test failed")
    except Exception as e:
        print(f"✗ Sample news data test failed: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install pandas numpy yfinance openai")
        print("2. Run: python test_nifty50_simple.py")
        print("3. Run: python run_nifty50_agent.py --dry-run")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)