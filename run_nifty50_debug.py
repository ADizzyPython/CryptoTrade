#!/usr/bin/env python3
"""
NIFTY50 Debug Mode - Shows AI Prompts and Responses
Automatic execution with full AI interaction visibility
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nifty50_agent import NIFTY50TradingAgent, create_nifty50_args
from nifty50_config import NIFTY50Config, create_balanced_preset
from utils import get_chat
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DebugLLMFunction:
    """Debug wrapper for LLM function to show prompts and responses"""
    
    def __init__(self):
        self.call_count = 0
        
    def __call__(self, prompt: str, model: str, seed: int) -> str:
        self.call_count += 1
        
        # Detect prompt type
        prompt_type = self.detect_prompt_type(prompt)
        
        print(f"\n{'='*80}")
        print(f"🤖 AI CALL #{self.call_count} - {prompt_type}")
        print(f"{'='*80}")
        print(f"📝 Model: {model}")
        print(f"🎯 Seed: {seed}")
        print(f"\n📤 PROMPT SENT:")
        print(f"{'─'*60}")
        print(prompt)
        print(f"{'─'*60}")
        
        # Call actual LLM
        start_time = time.time()
        try:
            response = get_chat(prompt, model, seed, temperature=0.0, max_tokens=512)
            elapsed = time.time() - start_time
            
            print(f"\n📥 RESPONSE RECEIVED (took {elapsed:.2f}s):")
            print(f"{'─'*60}")
            print(response)
            print(f"{'─'*60}")
            
            # Extract key information
            if "ACTION:" in response:
                action_line = [line for line in response.split('\n') if 'ACTION:' in line]
                if action_line:
                    print(f"🎯 TRADING DECISION: {action_line[0]}")
            
            return response
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"❌ ERROR: {str(e)}"
            print(f"\n📥 ERROR (took {elapsed:.2f}s):")
            print(f"{'─'*60}")
            print(error_msg)
            print(f"{'─'*60}")
            return f"Error: {str(e)}"
    
    def detect_prompt_type(self, prompt: str) -> str:
        """Detect what type of analysis prompt this is"""
        if "technical analyst" in prompt.lower():
            return "🔧 TECHNICAL ANALYSIS"
        elif "news analyst" in prompt.lower():
            return "📰 NEWS ANALYSIS"
        elif "reflection analyst" in prompt.lower():
            return "🤔 REFLECTION ANALYSIS"
        elif "experienced ETH cryptocurrency trader" in prompt.lower() or "experienced NIFTY50" in prompt.lower():
            return "💰 FINAL TRADING DECISION"
        else:
            return "❓ UNKNOWN ANALYSIS"

def run_debug_session(start_date: str, end_date: str, preset: str = 'balanced'):
    """Run debug session with full AI visibility"""
    
    print(f"\n{'='*80}")
    print(f"🚀 NIFTY50 DEBUG SESSION STARTING")
    print(f"{'='*80}")
    
    # Create configuration
    if preset == 'conservative':
        from nifty50_config import create_conservative_preset
        config = create_conservative_preset()
    elif preset == 'aggressive':
        from nifty50_config import create_aggressive_preset
        config = create_aggressive_preset()
    else:
        config = create_balanced_preset()
    
    # Show configuration
    print(f"📅 Trading Period: {start_date} to {end_date}")
    print(f"💰 Starting Capital: ₹{config.trading.starting_capital:,.2f}")
    print(f"🤖 Model: {config.model.model_name}")
    print(f"🔧 Technical Analysis: {config.analyst.use_technical}")
    print(f"📰 News Analysis: {config.analyst.use_news}")
    print(f"🤔 Reflection Analysis: {config.analyst.use_reflection}")
    print(f"⚖️ Max Position Size: {config.trading.max_position_size:.1%}")
    print(f"💸 Transaction Cost: {config.get_total_transaction_cost():.4%}")
    
    # Create debug LLM function
    debug_llm = DebugLLMFunction()
    
    # Create arguments
    args = create_nifty50_args(
        starting_date=start_date,
        ending_date=end_date,
        config=config
    )
    
    # Initialize agent
    try:
        print(f"\n{'='*80}")
        print(f"🔧 INITIALIZING TRADING AGENT")
        print(f"{'='*80}")
        
        agent = NIFTY50TradingAgent(args, debug_llm, config)
        
        print(f"\n{'='*80}")
        print(f"🎯 STARTING TRADING SESSION")
        print(f"{'='*80}")
        
        # Run trading session
        results = agent.run_trading_session()
        
        # Show final results
        print(f"\n{'='*80}")
        print(f"📊 FINAL RESULTS")
        print(f"{'='*80}")
        
        metrics = results['performance_metrics']
        
        print(f"🎯 Total Return: {metrics['total_return']:.2%}")
        print(f"💰 Final Value: ₹{metrics['final_value']:,.2f}")
        print(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"📈 Total Trades: {metrics['total_trades']}")
        print(f"🤖 Total AI Calls: {debug_llm.call_count}")
        
        # Generate performance report
        report = agent.generate_performance_report(results)
        print(f"\n{'='*80}")
        print(f"📋 PERFORMANCE REPORT")
        print(f"{'='*80}")
        print(report)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"nifty50_debug_results_{timestamp}.json"
        
        results['debug_info'] = {
            'total_ai_calls': debug_llm.call_count,
            'session_type': 'debug',
            'preset': preset
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='NIFTY50 Debug Mode')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-01-05', help='End date (YYYY-MM-DD)')
    parser.add_argument('--preset', type=str, default='balanced', choices=['conservative', 'balanced', 'aggressive'], help='Configuration preset')
    
    args = parser.parse_args()
    
    # Run debug session
    try:
        results = run_debug_session(args.start_date, args.end_date, args.preset)
        print(f"\n✅ Debug session completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Debug session failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()