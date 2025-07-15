#!/usr/bin/env python3
"""
NIFTY50 Data Utilities
Provides data fetching and processing utilities for NIFTY50 index data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
import requests
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NIFTY50DataFetcher:
    """Fetches and processes NIFTY50 index data from various sources"""
    
    def __init__(self, data_dir: str = "data/nifty50"):
        self.data_dir = data_dir
        self.nifty_symbol = "^NSEI"
        self.sensex_symbol = "^BSESN"
        os.makedirs(data_dir, exist_ok=True)
        
    def fetch_yfinance_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (^NSEI for NIFTY50)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Rename columns to match crypto format
            data = data.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add symbol name
            data['name'] = symbol
            
            # Calculate market cap (approximate for index)
            # For NIFTY50, market cap is the sum of market caps of all 50 stocks
            # This is a rough approximation using index value
            data['market_cap'] = data['close'] * 1000000  # Placeholder approximation
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_nifty50_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch NIFTY50 index data
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with processed NIFTY50 data
        """
        data = self.fetch_yfinance_data(self.nifty_symbol, start_date, end_date)
        
        if data.empty:
            return data
        
        # Process timestamps to match crypto format
        data['timestamp'] = pd.to_datetime(data['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Add time-based columns (Indian market hours: 9:15 AM to 3:30 PM IST)
        data['time_open'] = pd.to_datetime(data['timestamp']).dt.strftime('%Y-%m-%dT09:15:00.000Z')
        data['time_close'] = pd.to_datetime(data['timestamp']).dt.strftime('%Y-%m-%dT15:30:00.000Z')
        data['time_high'] = pd.to_datetime(data['timestamp']).dt.strftime('%Y-%m-%dT12:00:00.000Z')
        data['time_low'] = pd.to_datetime(data['timestamp']).dt.strftime('%Y-%m-%dT13:00:00.000Z')
        
        # Reorder columns to match crypto format
        columns_order = ['time_open', 'time_close', 'time_high', 'time_low', 'name', 
                        'open', 'high', 'low', 'close', 'volume', 'market_cap', 'timestamp']
        
        data = data[columns_order]
        
        return data
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for NIFTY50 data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        if data.empty:
            return data
        
        df = data.copy()
        
        # Simple Moving Averages
        for period in [5, 10, 15, 20, 30]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Standard deviation (volatility)
        df['volatility'] = df['close'].rolling(window=20).std()
        
        return df
    
    def save_daily_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save data to CSV file
        
        Args:
            data: DataFrame to save
            filename: Output filename
        """
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Saved {len(data)} records to {filepath}")
    
    def load_existing_data(self, filename: str) -> pd.DataFrame:
        """
        Load existing data from CSV file
        
        Args:
            filename: Input filename
            
        Returns:
            DataFrame with loaded data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()

class NIFTY50NewsProcessor:
    """Processes NIFTY50 news data for sentiment analysis"""
    
    def __init__(self, news_dir: str = "data/selected_nifty50_202401_202501"):
        self.news_dir = news_dir
        
    def load_news_for_date(self, date: str) -> List[Dict]:
        """
        Load news articles for a specific date
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            List of news articles
        """
        filepath = os.path.join(self.news_dir, f"{date}.json")
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    news_data = json.load(f)
                    return news_data.get('organic', [])
            except Exception as e:
                logger.error(f"Error loading news for {date}: {e}")
                return []
        else:
            logger.warning(f"News file not found for {date}")
            return []
    
    def process_news_for_sentiment(self, articles: List[Dict]) -> Dict:
        """
        Process news articles for sentiment analysis
        
        Args:
            articles: List of news articles
            
        Returns:
            Dictionary with sentiment summary
        """
        if not articles:
            return {
                'total_articles': 0,
                'positive_sentiment': 0,
                'negative_sentiment': 0,
                'neutral_sentiment': 0,
                'avg_sentiment': 0.0
            }
        
        # Basic sentiment analysis keywords
        positive_keywords = ['surge', 'gain', 'rise', 'rally', 'bull', 'growth', 'positive', 'up', 'high']
        negative_keywords = ['fall', 'drop', 'decline', 'bear', 'crash', 'down', 'low', 'negative', 'loss']
        
        sentiment_scores = []
        
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = f"{title} {description}"
            
            positive_count = sum(1 for word in positive_keywords if word in content)
            negative_count = sum(1 for word in negative_keywords if word in content)
            
            if positive_count > negative_count:
                sentiment_scores.append(1)
            elif negative_count > positive_count:
                sentiment_scores.append(-1)
            else:
                sentiment_scores.append(0)
        
        positive_count = sum(1 for score in sentiment_scores if score == 1)
        negative_count = sum(1 for score in sentiment_scores if score == -1)
        neutral_count = sum(1 for score in sentiment_scores if score == 0)
        
        return {
            'total_articles': len(articles),
            'positive_sentiment': positive_count,
            'negative_sentiment': negative_count,
            'neutral_sentiment': neutral_count,
            'avg_sentiment': np.mean(sentiment_scores) if sentiment_scores else 0.0
        }

class NIFTY50DataManager:
    """Main class for managing NIFTY50 data operations"""
    
    def __init__(self, data_dir: str = "data/nifty50"):
        self.data_fetcher = NIFTY50DataFetcher(data_dir)
        self.news_processor = NIFTY50NewsProcessor()
        self.data_dir = data_dir
        
    def update_nifty50_data(self, start_date: str, end_date: str) -> None:
        """
        Update NIFTY50 data for the specified date range
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        logger.info(f"Updating NIFTY50 data from {start_date} to {end_date}")
        
        # Fetch raw data
        raw_data = self.data_fetcher.fetch_nifty50_data(start_date, end_date)
        
        if raw_data.empty:
            logger.warning("No data fetched")
            return
        
        # Add technical indicators
        processed_data = self.data_fetcher.calculate_technical_indicators(raw_data)
        
        # Save to file
        filename = f"nifty50_daily_{start_date}_{end_date}.csv"
        self.data_fetcher.save_daily_data(processed_data, filename)
        
        logger.info(f"Updated NIFTY50 data: {len(processed_data)} records")
    
    def get_trading_data(self, date: str) -> Dict:
        """
        Get comprehensive trading data for a specific date
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            Dictionary with price data, technical indicators, and news sentiment
        """
        # Load price data (you might need to implement date-specific loading)
        price_data = self.data_fetcher.load_existing_data("nifty50_daily_2024-01-01_2024-12-31.csv")
        
        if price_data.empty:
            logger.warning(f"No price data available for {date}")
            return {}
        
        # Filter for specific date
        price_data['date'] = pd.to_datetime(price_data['timestamp']).dt.date
        target_date = pd.to_datetime(date).date()
        day_data = price_data[price_data['date'] == target_date]
        
        if day_data.empty:
            logger.warning(f"No price data for {date}")
            return {}
        
        # Get news sentiment
        news_articles = self.news_processor.load_news_for_date(date)
        news_sentiment = self.news_processor.process_news_for_sentiment(news_articles)
        
        # Combine data
        return {
            'date': date,
            'price_data': day_data.iloc[0].to_dict(),
            'news_sentiment': news_sentiment,
            'technical_indicators': {
                'sma_5': day_data['sma_5'].iloc[0] if 'sma_5' in day_data.columns else None,
                'sma_20': day_data['sma_20'].iloc[0] if 'sma_20' in day_data.columns else None,
                'macd': day_data['macd'].iloc[0] if 'macd' in day_data.columns else None,
                'rsi': day_data['rsi'].iloc[0] if 'rsi' in day_data.columns else None,
                'volatility': day_data['volatility'].iloc[0] if 'volatility' in day_data.columns else None
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize data manager
    data_manager = NIFTY50DataManager()
    
    # Update data for 2024
    data_manager.update_nifty50_data("2024-01-01", "2024-12-31")
    
    # Get trading data for a specific date
    trading_data = data_manager.get_trading_data("2024-01-15")
    print(json.dumps(trading_data, indent=2, default=str))