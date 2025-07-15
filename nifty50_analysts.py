#!/usr/bin/env python3
"""
NIFTY50 Analysts
Implementation of specialized analysts for NIFTY50 trading
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

class NIFTY50TechnicalAnalyst:
    """Analyzes NIFTY50 technical indicators and market statistics"""
    
    def __init__(self):
        self.delim = "\n" + "="*50 + "\n"
        
    def generate_technical_analysis(self, state: Dict) -> str:
        """
        Generate technical analysis prompt for NIFTY50 index
        
        Args:
            state: Current market state containing price data and technical indicators
            
        Returns:
            Formatted prompt for technical analyst
        """
        prompt = f"You are a NIFTY50 index trading analyst specializing in technical analysis. "
        prompt += f"The recent price data and technical indicators are given below:{self.delim}"
        
        # Current price information
        prompt += f"Current Date: {state['date']}\n"
        prompt += f"NIFTY50 Opening Price: ₹{state['open']:.2f}\n"
        prompt += f"Current Cash: ₹{state['cash']:.2f}\n"
        prompt += f"NIFTY50 Units Held: {state['nifty_units']:.2f}\n"
        prompt += f"Current Net Worth: ₹{state['net_worth']:.2f}\n"
        prompt += f"Portfolio ROI: {state['roi']:.2%}\n"
        prompt += f"Today's ROI: {state['today_roi']:.2%}\n"
        
        # Technical indicators
        technical = state.get('technical', {})
        if technical:
            prompt += f"\nTechnical Indicators:\n"
            prompt += f"- SMA 5: ₹{technical.get('sma_5', 'N/A')}\n"
            prompt += f"- SMA 20: ₹{technical.get('sma_20', 'N/A')}\n"
            prompt += f"- SMA Crossover Signal: {technical.get('sma_crossover_signal', 'N/A')}\n"
            prompt += f"- MACD Signal: {technical.get('macd_signal', 'N/A')}\n"
            prompt += f"- MACD Value: {technical.get('macd_value', 'N/A'):.4f}\n"
            prompt += f"- Bollinger Bands Signal: {technical.get('bollinger_bands_signal', 'N/A')}\n"
            prompt += f"- RSI Signal: {technical.get('rsi_signal', 'N/A')}\n"
            prompt += f"- RSI Value: {technical.get('rsi_value', 'N/A'):.2f}\n"
        
        # Market statistics
        market_stats = state.get('market_stats', {})
        if market_stats:
            prompt += f"\nMarket Statistics:\n"
            prompt += f"- Trading Volume: {market_stats.get('volume', 'N/A')}\n"
            prompt += f"- Volatility: {market_stats.get('volatility', 'N/A'):.4f}\n"
            prompt += f"- Market Cap: ₹{market_stats.get('market_cap', 'N/A')}\n"
            prompt += f"- High-Low Spread: {market_stats.get('high_low_spread', 'N/A'):.4f}\n"
            prompt += f"- Daily Return: {market_stats.get('daily_return', 'N/A'):.4f}\n"
        
        prompt += f"{self.delim}"
        prompt += f"Based on the technical indicators and market statistics above, analyze the current market trend for NIFTY50. "
        prompt += f"Consider the following in your analysis:\n"
        prompt += f"1. Moving average trends and crossover signals\n"
        prompt += f"2. MACD momentum and divergence patterns\n"
        prompt += f"3. RSI overbought/oversold conditions\n"
        prompt += f"4. Bollinger Bands squeeze or expansion\n"
        prompt += f"5. Volume analysis and market participation\n"
        prompt += f"6. Overall market volatility and risk assessment\n\n"
        prompt += f"Write one concise paragraph analyzing the technical indicators and estimating the NIFTY50 market trend accordingly."
        
        return prompt

class NIFTY50NewsAnalyst:
    """Analyzes NIFTY50 and Indian market news for sentiment analysis"""
    
    def __init__(self):
        self.delim = "\n" + "="*50 + "\n"
        
    def generate_news_analysis(self, state: Dict) -> str:
        """
        Generate news analysis prompt for NIFTY50 trading
        
        Args:
            state: Current market state containing news data
            
        Returns:
            Formatted prompt for news analyst
        """
        prompt = f"You are a NIFTY50 index trading analyst specializing in news and sentiment analysis. "
        prompt += f"You are required to analyze the following news articles related to NIFTY50, Indian stock market, and economy:"
        prompt += f"{self.delim}"
        
        news = state.get('news', [])
        if news == 'N/A' or not news:
            prompt += "No relevant news available for today.\n"
        else:
            for i, article in enumerate(news, 1):
                prompt += f"Article {i}:\n"
                prompt += f"Title: {article.get('title', 'N/A')}\n"
                prompt += f"Date: {article.get('date', 'N/A')}\n"
                prompt += f"Description: {article.get('description', 'N/A')}\n"
                prompt += f"Source: {article.get('link', 'N/A')}\n"
                prompt += f"---\n"
        
        prompt += f"{self.delim}"
        prompt += f"Based on the news articles above, analyze the market sentiment for NIFTY50. "
        prompt += f"Consider the following in your analysis:\n"
        prompt += f"1. Overall market sentiment (bullish, bearish, or neutral)\n"
        prompt += f"2. Key economic indicators and government policies\n"
        prompt += f"3. Corporate earnings and sectoral performance\n"
        prompt += f"4. Global market influences on Indian markets\n"
        prompt += f"5. Regulatory changes and their market impact\n"
        prompt += f"6. FII/DII investment flows and institutional behavior\n\n"
        prompt += f"Write one concise paragraph analyzing the news sentiment and estimating the NIFTY50 market trend accordingly."
        
        return prompt

class NIFTY50ReflectionAnalyst:
    """Analyzes past trading performance and provides strategic guidance"""
    
    def __init__(self):
        self.delim = "\n" + "="*50 + "\n"
        
    def generate_reflection_analysis(self, trading_history: List[Dict]) -> str:
        """
        Generate reflection analysis based on past trading performance
        
        Args:
            trading_history: List of past trading actions and results
            
        Returns:
            Formatted prompt for reflection analyst
        """
        prompt = f"You are a NIFTY50 index trading analyst specializing in performance analysis and strategic planning. "
        prompt += f"Your trading history and performance data are given in chronological order below:"
        prompt += f"{self.delim}"
        
        if not trading_history:
            prompt += "No trading history available yet.\n"
        else:
            for i, trade in enumerate(trading_history, 1):
                prompt += f"Trade {i}:\n"
                prompt += f"Date: {trade.get('date', 'N/A')}\n"
                prompt += f"Action: {trade.get('action', 'N/A'):.4f} (1=buy, -1=sell, 0=hold)\n"
                prompt += f"Price: ₹{trade.get('price', 'N/A'):.2f}\n"
                prompt += f"Cash: ₹{trade.get('cash', 'N/A'):.2f}\n"
                prompt += f"NIFTY50 Units: {trade.get('nifty_units', 'N/A'):.2f}\n"
                prompt += f"Net Worth: ₹{trade.get('net_worth', 'N/A'):.2f}\n"
                
                # Calculate daily return if possible
                if i > 1:
                    prev_worth = trading_history[i-2].get('net_worth', 0)
                    curr_worth = trade.get('net_worth', 0)
                    if prev_worth > 0:
                        daily_return = (curr_worth / prev_worth - 1) * 100
                        prompt += f"Daily Return: {daily_return:.2f}%\n"
                
                prompt += f"---\n"
        
        prompt += f"{self.delim}"
        prompt += f"Based on your recent trading performance, reflect on the following:\n"
        prompt += f"1. Which trading decisions were most successful and why?\n"
        prompt += f"2. What market conditions led to losses and how can they be avoided?\n"
        prompt += f"3. Are there patterns in your trading behavior that need adjustment?\n"
        prompt += f"4. How well did technical analysis and news sentiment align with actual market movements?\n"
        prompt += f"5. What strategic changes should be made for future trades?\n"
        prompt += f"6. Risk management lessons learned from recent performance\n\n"
        prompt += f"Write one concise paragraph reflecting on your recent performance and providing strategic guidance for future NIFTY50 trades."
        
        return prompt

class NIFTY50TradingCoordinator:
    """Coordinates all analysts and makes final trading decisions"""
    
    def __init__(self):
        self.technical_analyst = NIFTY50TechnicalAnalyst()
        self.news_analyst = NIFTY50NewsAnalyst()
        self.reflection_analyst = NIFTY50ReflectionAnalyst()
        self.delim = "\n" + "="*50 + "\n"
        
    def generate_final_trading_prompt(self, technical_analysis: str, news_analysis: str, 
                                    reflection_analysis: str, state: Dict) -> str:
        """
        Generate the final trading decision prompt combining all analyst inputs
        
        Args:
            technical_analysis: Output from technical analyst
            news_analysis: Output from news analyst  
            reflection_analysis: Output from reflection analyst
            state: Current market state
            
        Returns:
            Final trading prompt for decision making
        """
        base_prompt = f"You are an experienced NIFTY50 index trader and you are trying to maximize your overall profit by trading NIFTY50. "
        base_prompt += f"In each day, you will make an action to buy or sell NIFTY50 units. "
        base_prompt += f"You are assisted by three specialist analysts below and need to decide the final action."
        
        prompt = f"{base_prompt}\n\n"
        
        # Technical analysis
        prompt += f"TECHNICAL ANALYST REPORT:{self.delim}{technical_analysis}{self.delim}\n"
        
        # News analysis
        prompt += f"NEWS ANALYST REPORT:{self.delim}{news_analysis}{self.delim}\n"
        
        # Reflection analysis
        prompt += f"REFLECTION ANALYST REPORT:{self.delim}{reflection_analysis}{self.delim}\n"
        
        # Current portfolio status
        prompt += f"CURRENT PORTFOLIO STATUS:\n"
        prompt += f"- Current Cash: ₹{state['cash']:.2f}\n"
        prompt += f"- NIFTY50 Units Held: {state['nifty_units']:.2f}\n"
        prompt += f"- Current Net Worth: ₹{state['net_worth']:.2f}\n"
        prompt += f"- Portfolio ROI: {state['roi']:.2%}\n"
        prompt += f"- Opening Price: ₹{state['open']:.2f}\n"
        
        # Action instructions
        prompt += f"\nACTION INSTRUCTIONS:\n"
        prompt += f"Based on the three analyst reports above, decide your trading action for today.\n"
        prompt += f"Your action should be a number between -1 and 1:\n"
        prompt += f"- Action > 0: BUY (positive values indicate buying intensity)\n"
        prompt += f"- Action < 0: SELL (negative values indicate selling intensity) \n"
        prompt += f"- Action = 0: HOLD (maintain current position)\n\n"
        
        prompt += f"Consider the following in your decision:\n"
        prompt += f"1. Technical indicators and market momentum\n"
        prompt += f"2. News sentiment and market events\n"
        prompt += f"3. Your past trading performance and lessons learned\n"
        prompt += f"4. Risk management and position sizing\n"
        prompt += f"5. Current market conditions and volatility\n"
        prompt += f"6. Portfolio diversification and cash management\n\n"
        
        prompt += f"Please provide your response in the following format:\n"
        prompt += f"REASONING: [Explain your decision based on the analyst reports]\n"
        prompt += f"ACTION: [A single number between -1 and 1]\n"
        
        return prompt
    
    def parse_trading_response(self, response: str) -> tuple[str, float]:
        """
        Parse the trading response to extract reasoning and action
        
        Args:
            response: Raw response from the trading model
            
        Returns:
            Tuple of (reasoning, action) where action is a float between -1 and 1
        """
        try:
            lines = response.strip().split('\n')
            reasoning = ""
            action = 0.0
            
            for line in lines:
                line = line.strip()
                if line.startswith('REASONING:'):
                    reasoning = line.replace('REASONING:', '').strip()
                elif line.startswith('ACTION:'):
                    action_str = line.replace('ACTION:', '').strip()
                    try:
                        action = float(action_str)
                        action = np.clip(action, -1.0, 1.0)  # Ensure valid range
                    except ValueError:
                        action = 0.0
            
            return reasoning, action
            
        except Exception as e:
            print(f"Error parsing trading response: {e}")
            return "Error parsing response", 0.0

class NIFTY50AnalystManager:
    """Main manager for coordinating all NIFTY50 analysts"""
    
    def __init__(self):
        self.coordinator = NIFTY50TradingCoordinator()
        
    def analyze_and_trade(self, state: Dict, trading_history: List[Dict], 
                         llm_function, model: str = "gpt-4", seed: int = 42) -> tuple[str, float]:
        """
        Run full analysis cycle and generate trading decision
        
        Args:
            state: Current market state
            trading_history: Past trading actions and results
            llm_function: Function to call language model
            model: Model name to use
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (reasoning, action) for the trading decision
        """
        # Generate technical analysis
        technical_prompt = self.coordinator.technical_analyst.generate_technical_analysis(state)
        technical_analysis = llm_function(technical_prompt, model, seed).strip()
        
        # Generate news analysis
        news_prompt = self.coordinator.news_analyst.generate_news_analysis(state)
        news_analysis = llm_function(news_prompt, model, seed).strip()
        
        # Generate reflection analysis
        reflection_prompt = self.coordinator.reflection_analyst.generate_reflection_analysis(trading_history)
        reflection_analysis = llm_function(reflection_prompt, model, seed).strip()
        
        # Generate final trading decision
        final_prompt = self.coordinator.generate_final_trading_prompt(
            technical_analysis, news_analysis, reflection_analysis, state
        )
        trading_response = llm_function(final_prompt, model, seed).strip()
        
        # Parse the response
        reasoning, action = self.coordinator.parse_trading_response(trading_response)
        
        return reasoning, action
    
    def get_analyst_reports(self, state: Dict, trading_history: List[Dict], 
                          llm_function, model: str = "gpt-4", seed: int = 42) -> Dict[str, str]:
        """
        Get individual analyst reports without making trading decision
        
        Args:
            state: Current market state
            trading_history: Past trading actions and results
            llm_function: Function to call language model
            model: Model name to use
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing all analyst reports
        """
        # Generate all analyst reports
        technical_prompt = self.coordinator.technical_analyst.generate_technical_analysis(state)
        technical_analysis = llm_function(technical_prompt, model, seed).strip()
        
        news_prompt = self.coordinator.news_analyst.generate_news_analysis(state)
        news_analysis = llm_function(news_prompt, model, seed).strip()
        
        reflection_prompt = self.coordinator.reflection_analyst.generate_reflection_analysis(trading_history)
        reflection_analysis = llm_function(reflection_prompt, model, seed).strip()
        
        return {
            'technical_analysis': technical_analysis,
            'news_analysis': news_analysis,
            'reflection_analysis': reflection_analysis,
            'technical_prompt': technical_prompt,
            'news_prompt': news_prompt,
            'reflection_prompt': reflection_prompt
        }