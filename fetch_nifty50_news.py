#!/usr/bin/env python
import requests
import os
import json
import re
from datetime import datetime, timedelta
import time
import urllib.parse

# Output directory
OUTDIR = 'data/selected_nifty50_202401_202501'
os.makedirs(OUTDIR, exist_ok=True)

# Test with first 3 dates
start_date = datetime(2024, 1, 1)
dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(365)]

def matches_target_date(result_date, target_date):
    """Check if result_date matches the target_date"""
    try:
        # Convert target_date from YYYY-MM-DD to datetime
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        
        # Parse result_date (format: "Jan 4, 2024", "4 Jan 2024", etc.)
        result_date_clean = result_date.strip()
        
        # Try different formats
        formats = [
            '%b %d, %Y',    # "Jan 4, 2024"
            '%d %b %Y',     # "4 Jan 2024"
            '%B %d, %Y',    # "January 4, 2024"
            '%d %B %Y',     # "4 January 2024"
        ]
        
        for fmt in formats:
            try:
                result_dt = datetime.strptime(result_date_clean, fmt)
                # Check if dates match
                return result_dt.date() == target_dt.date()
            except ValueError:
                continue
        
        return False
    except Exception as e:
        print(f"Error matching dates: {result_date} vs {target_date}: {e}")
        return False

def fetch_news_for_date(date):
    """Fetch news for a specific date using BrightData proxy with date-specific query"""
    
    proxies = {
        "http": "http://brd-customer-hl_c0565a8d-zone-serp_api1:atpvk7cmhpj2@brd.superproxy.io:33335",
        "https": "http://brd-customer-hl_c0565a8d-zone-serp_api1:atpvk7cmhpj2@brd.superproxy.io:33335"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Calculate next day for before parameter
    current_date = datetime.strptime(date, '%Y-%m-%d')
    next_date = (current_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Build the URL with date-specific query similar to curl command
    query = f'before:{next_date} after:{date} "nifty"'
    encoded_query = urllib.parse.quote(query)
    
    url = f'https://www.google.com/search?q={encoded_query}&location=India&uule=w+CAIQICIFSW5kaWE&num=20&brd_json=1'
    
    print(f'Fetching news for {date}...')
    print(f'Query: {query}')
    print(f'URL: {url}')
    
    try:
        response = requests.get(url, headers=headers, proxies=proxies, verify=False)
        
        if response.status_code == 200:
            try:
                # Try to parse as JSON first
                data = response.json()
                return process_json_response(data, date, query)
            except json.JSONDecodeError:
                # If not JSON, treat as HTML
                html_content = response.text
                return process_html_response(html_content, date, query)
        else:
            print(f'HTTP Error {response.status_code} for {date}')
            return None
        
    except Exception as e:
        print(f'Error fetching news for {date}: {e}')
        return None

def process_json_response(data, date, query):
    """Process JSON response from BrightData SERP API"""
    
    # Extract organic results
    organic_results = []
    if 'organic' in data:
        for result in data['organic']:
            # Extract date from extensions if available
            result_date = None
            if 'extensions' in result:
                for ext in result['extensions']:
                    if 'text' in ext and any(month in ext['text'] for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                        result_date = ext['text']
                        break
            
            # Only keep results that match the target date
            if result_date and matches_target_date(result_date, date):
                organic_results.append({
                    'title': result.get('title', ''),
                    'description': result.get('description', ''),
                    'link': result.get('link', ''),
                    'display_link': result.get('display_link', ''),
                    'date': result_date,
                    'rank': result.get('rank', 0)
                })
    
    # Build final structure
    result = {
        'general': {
            'search_engine': 'google',
            'query': query,
            'results_cnt': data.get('general', {}).get('results_cnt', 0),
            'search_time': data.get('general', {}).get('search_time', 0),
            'language': data.get('general', {}).get('language', 'en'),
            'country': data.get('general', {}).get('country', 'India'),
            'location': 'India',
            'timestamp': datetime.now().isoformat() + 'Z'
        },
        'organic': organic_results
    }
    
    return result

def process_html_response(html_content, date, query):
    """Process HTML response and extract organic results"""
    
    # Save debug HTML
    debug_file = os.path.join(OUTDIR, f'debug_{date}.html')
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f'Debug: Saved HTML to {debug_file}')
    
    # Extract organic results from HTML
    organic_results = []
    
    # Pattern to match search results
    result_pattern = r'<div[^>]*class="[^"]*g[^"]*"[^>]*>.*?<h3[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?</h3>.*?<span[^>]*>(.*?)</span>'
    matches = re.findall(result_pattern, html_content, re.DOTALL)
    
    for i, (link, title, description) in enumerate(matches[:10]):
        # Clean up text
        title = re.sub(r'<[^>]*>', '', title).strip()
        description = re.sub(r'<[^>]*>', '', description).strip()
        
        # Extract date if present in description
        date_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+,\s+\d{4}', description)
        result_date = date_match.group(0) if date_match else None
        
        # Only keep results that match the target date
        if result_date and matches_target_date(result_date, date):
            organic_results.append({
                'title': title,
                'description': description,
                'link': link,
                'display_link': link,
                'date': result_date,
                'rank': i + 1
            })
    
    # Build final structure
    result = {
        'general': {
            'search_engine': 'google',
            'query': query,
            'results_cnt': len(organic_results),
            'search_time': 0.0,
            'language': 'en',
            'country': 'India',
            'location': 'India',
            'timestamp': datetime.now().isoformat() + 'Z'
        },
        'input': {
            'original_url': '',
            'request_id': f'fetch_{date}_{int(time.time())}'
        },
        'navigation': [],
        'organic': organic_results
    }
    
    return result

def main():
    """Main function to fetch news for all dates"""
    print('Starting NIFTY50 news fetcher with BrightData SERP API...')
    
    for date in dates:
        outpath = os.path.join(OUTDIR, f'{date}.json')
        
        if os.path.exists(outpath):
            print(f'{outpath} already exists, skipping.')
            continue
        
        # Fetch news data
        data = fetch_news_for_date(date)
        
        if data:
            # Save to JSON file
            with open(outpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f'Saved {outpath} with {len(data["organic"])} organic results')
        else:
            print(f'No data for {date}')
        
        # Add delay between requests
        time.sleep(3)
    
    print('Completed!')

if __name__ == '__main__':
    main()