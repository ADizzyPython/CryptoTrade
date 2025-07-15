#!/usr/bin/env python3
"""
NIFTY50 Live Console - Real-time Trading with AI Interaction Visibility
Shows all prompts sent to AI and responses received in real-time
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nifty50_agent import NIFTY50TradingAgent, create_nifty50_args
from nifty50_config import NIFTY50Config, create_balanced_preset
from utils import get_chat
import logging

# Configure logging for live console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('nifty50_live_console.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveConsole:
    """Live console for NIFTY50 trading with AI interaction visibility"""
    
    def __init__(self):
        self.step_counter = 0
        self.prompt_counter = 0
        self.total_cost = 0.0
        
    def print_separator(self, title="", char="=", width=80):
        """Print a separator line"""
        if title:
            title_line = f" {title} "
            padding = (width - len(title_line)) // 2
            line = char * padding + title_line + char * padding
            if len(line) < width:
                line += char
        else:
            line = char * width
        print(f"\n{line}")
    
    def print_colored(self, text, color_code=None):
        """Print colored text"""
        if color_code:
            print(f"\033[{color_code}m{text}\033[0m")
        else:
            print(text)
    
    def truncate_text(self, text, max_length=200):
        """Truncate text for display"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def create_live_llm_function(self):
        """Create LLM function with live logging"""
        def llm_function(prompt: str, model: str, seed: int) -> str:
            self.prompt_counter += 1
            
            # Show prompt being sent
            self.print_separator(f"AI REQUEST #{self.prompt_counter}", "─")
            self.print_colored(f"🤖 Model: {model}", "36")  # Cyan
            self.print_colored(f"🎯 Seed: {seed}", "36")    # Cyan
            self.print_colored(f"📝 Prompt Type: {self.detect_prompt_type(prompt)}", "35")  # Magenta
            
            # Show truncated prompt
            print(f"\n📤 PROMPT SENT:")
            self.print_colored(self.truncate_text(prompt, 500), "37")  # Light gray
            
            # Send to AI and measure time
            start_time = time.time()
            try:
                response = get_chat(prompt, model, seed, temperature=0.0, max_tokens=512)
                end_time = time.time()
                
                # Show response received
                print(f"\n📥 RESPONSE RECEIVED (took {end_time - start_time:.2f}s):")
                self.print_colored(response, "32")  # Green
                
                # Extract key info if it's a trading decision
                if "ACTION:" in response:
                    self.extract_trading_decision(response)
                
                return response
                
            except Exception as e:
                end_time = time.time()
                error_msg = f"❌ ERROR: {str(e)}"
                self.print_colored(error_msg, "31")  # Red
                return f"Error: {str(e)}"
        
        return llm_function
    
    def detect_prompt_type(self, prompt: str) -> str:
        """Detect what type of analysis prompt this is"""
        if "technical analyst" in prompt.lower():
            return "🔧 Technical Analysis"
        elif "news analyst" in prompt.lower():
            return "📰 News Analysis"
        elif "reflection analyst" in prompt.lower():
            return "🤔 Reflection Analysis"
        elif "experienced ETH cryptocurrency trader" in prompt.lower():
            return "💰 Final Trading Decision"
        else:
            return "❓ Unknown Analysis"
    
    def extract_trading_decision(self, response: str):
        """Extract and highlight trading decision from response"""
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('ACTION:'):
                action_value = line.replace('ACTION:', '').strip()
                self.print_colored(f"🎯 TRADING DECISION: {action_value}", "33")  # Yellow
                break
    
    def run_live_session(self, config: NIFTY50Config, start_date: str, end_date: str):
        """Run a live trading session with full visibility"""
        
        self.print_separator("NIFTY50 LIVE TRADING CONSOLE", "=")
        self.print_colored("🚀 Starting live trading session with AI interaction visibility", "32")
        
        # Create arguments
        args = create_nifty50_args(
            starting_date=start_date,
            ending_date=end_date,
            config=config
        )
        
        # Create live LLM function
        live_llm_function = self.create_live_llm_function()
        
        # Show configuration
        self.print_separator("CONFIGURATION", "─")
        print(f"📅 Trading Period: {start_date} to {end_date}")
        print(f"💰 Starting Capital: ₹{config.trading.starting_capital:,.2f}")
        print(f"🤖 Model: {config.model.model_name}")
        print(f"🔧 Technical Analysis: {config.analyst.use_technical}")
        print(f"📰 News Analysis: {config.analyst.use_news}")
        print(f"🤔 Reflection Analysis: {config.analyst.use_reflection}")
        print(f"⚖️ Max Position Size: {config.trading.max_position_size:.1%}")
        print(f"💸 Transaction Cost: {config.get_total_transaction_cost():.4%}")
        
        # Initialize agent
        try:
            self.print_separator("INITIALIZING AGENT", "─")
            agent = NIFTY50TradingAgent(args, live_llm_function, config)
            
            # Hook into the agent's step method to show trading steps
            original_step = agent.run_trading_session
            
            def monitored_run_trading_session():
                self.print_separator("STARTING TRADING SESSION", "=")
                
                step = 0
                current_state = agent.current_state
                
                while current_state is not None:
                    step += 1
                    self.step_counter = step
                    
                    # Show step header
                    self.print_separator(f"TRADING STEP {step}", "=")
                    self.print_colored(f"📊 Date: {current_state.get('date', 'N/A')}", "34")
                    self.print_colored(f"💰 Portfolio Value: ₹{current_state.get('net_worth', 0):.2f}", "34")
                    self.print_colored(f"📈 ROI: {current_state.get('roi', 0):.2%}", "34")
                    self.print_colored(f"💵 Cash: ₹{current_state.get('cash', 0):.2f}", "34")
                    self.print_colored(f"📊 NIFTY Units: {current_state.get('nifty_units', 0):.2f}", "34")
                    
                    # Show market data
                    if 'technical' in current_state:
                        tech = current_state['technical']
                        print(f"\n📈 Technical Indicators:")
                        print(f"  • SMA 5: ₹{tech.get('sma_5', 'N/A')}")
                        print(f"  • SMA 20: ₹{tech.get('sma_20', 'N/A')}")
                        print(f"  • MACD Signal: {tech.get('macd_signal', 'N/A')}")
                        print(f"  • RSI: {tech.get('rsi_value', 'N/A')}")
                    
                    # Show news count
                    if 'news' in current_state:
                        news = current_state['news']
                        if news != 'N/A':
                            print(f"\n📰 News Articles: {len(news)} articles available")
                        else:
                            print(f"\n📰 News Articles: No news available")
                    
                    # Wait for user input to continue (optional)
                    input("\n⏯️  Press Enter to continue with AI analysis...")
                    
                    # Now let the original method handle the AI calls
                    try:
                        # Get current state and trading history
                        state = agent.prepare_state_for_analysis(current_state)
                        recent_history = agent.get_recent_trading_history()
                        
                        # Run analyst analysis
                        reasoning, action = agent.analyst_manager.analyze_and_trade(
                            state, recent_history, live_llm_function, agent.model, agent.seed
                        )
                        
                        # Execute trade
                        next_state, reward, done, info = agent.env.step(action)
                        
                        # Show results
                        self.print_separator("TRADING RESULT", "─")
                        self.print_colored(f"🎯 Final Action: {action:.4f}", "33")
                        self.print_colored(f"💭 Reasoning: {reasoning}", "37")
                        self.print_colored(f"🎁 Reward: {reward:.6f}", "32")
                        
                        # Update state
                        current_state = next_state
                        
                        if done:
                            self.print_separator("TRADING SESSION COMPLETED", "=")
                            break
                            
                    except Exception as e:
                        self.print_colored(f"❌ Error in trading step {step}: {e}", "31")
                        # Skip this step
                        next_state, reward, done, info = agent.env.step(0)
                        current_state = next_state
                        
                        if done:
                            break
                
                # Get final results
                results = {
                    'performance_metrics': agent.env.get_performance_metrics(),
                    'trading_history': agent.env.get_action_history(),
                    'total_prompts': self.prompt_counter
                }
                
                return results
            
            # Run the monitored session
            results = monitored_run_trading_session()
            
            # Show final results
            self.print_separator("FINAL RESULTS", "=")
            metrics = results['performance_metrics']
            
            self.print_colored(f"🎯 Total Return: {metrics['total_return']:.2%}", "32")
            self.print_colored(f"💰 Final Value: ₹{metrics['final_value']:,.2f}", "32")
            self.print_colored(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']:.4f}", "32")
            self.print_colored(f"📈 Total Trades: {metrics['total_trades']}", "32")
            self.print_colored(f"🤖 Total AI Calls: {self.prompt_counter}", "36")
            
            # Save detailed log
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"nifty50_live_session_{timestamp}.json"
            
            with open(log_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Detailed session log saved to: {log_file}")
            
        except Exception as e:
            self.print_colored(f"❌ Error running live session: {e}", "31")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='NIFTY50 Live Trading Console')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-01-10', help='End date (YYYY-MM-DD)')
    parser.add_argument('--preset', type=str, default='balanced', choices=['conservative', 'balanced', 'aggressive'], help='Configuration preset')
    
    args = parser.parse_args()
    
    # Create configuration
    if args.preset == 'conservative':
        from nifty50_config import create_conservative_preset
        config = create_conservative_preset()
    elif args.preset == 'aggressive':
        from nifty50_config import create_aggressive_preset
        config = create_aggressive_preset()
    else:
        config = create_balanced_preset()
    
    # Create and run live console
    console = LiveConsole()
    console.run_live_session(config, args.start_date, args.end_date)

if __name__ == "__main__":
    main()