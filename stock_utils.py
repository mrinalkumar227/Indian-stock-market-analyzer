"""
Stock Utility Functions for NSE Stock Analysis
Contains helper functions for data fetching, technical indicators, and signal detection.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from nse_stocks import (
    NIFTY_50,
    get_all_indices,
    get_stocks_by_index,
    get_all_nse_stocks,
    get_index_count
)

# Backward compatibility alias
NIFTY_50_STOCKS = NIFTY_50


def fetch_stock_data(symbol: str, period: str = "1y") -> tuple[pd.DataFrame | None, str | None]:
    """
    Fetch stock data from Yahoo Finance for NSE stocks.
    
    Args:
        symbol: Stock symbol (without .NS suffix)
        period: Time period for data (default: 1 year)
    
    Returns:
        Tuple of (DataFrame with stock data, error message if any)
    """
    try:
        # Append .NS for NSE stocks
        nse_symbol = f"{symbol.upper()}.NS"
        ticker = yf.Ticker(nse_symbol)
        
        # Fetch historical data
        data = ticker.history(period=period)
        
        if data.empty:
            return None, f"No data found for symbol: {symbol}"
        
        return data, None
        
    except Exception as e:
        return None, f"Error fetching data for {symbol}: {str(e)}"


def get_stock_info(symbol: str) -> tuple[dict | None, str | None]:
    """
    Get current stock information including price and 52-week data.
    
    Args:
        symbol: Stock symbol (without .NS suffix)
    
    Returns:
        Tuple of (info dictionary, error message if any)
    """
    try:
        nse_symbol = f"{symbol.upper()}.NS"
        ticker = yf.Ticker(nse_symbol)
        info = ticker.info
        
        if not info or 'regularMarketPrice' not in info:
            # Try fetching from history as fallback
            hist = ticker.history(period="5d")
            if hist.empty:
                return None, f"No data found for symbol: {symbol}"
            
            # Calculate from historical data
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            # Get 52-week data
            yearly_hist = ticker.history(period="1y")
            if not yearly_hist.empty:
                week_52_high = yearly_hist['High'].max()
                week_52_low = yearly_hist['Low'].min()
            else:
                week_52_high = current_price
                week_52_low = current_price
            
            return {
                'symbol': symbol.upper(),
                'current_price': current_price,
                'previous_close': prev_close,
                'daily_change_pct': ((current_price - prev_close) / prev_close) * 100,
                'week_52_high': week_52_high,
                'week_52_low': week_52_low,
            }, None
        
        current_price = info.get('regularMarketPrice', info.get('currentPrice', 0))
        prev_close = info.get('regularMarketPreviousClose', info.get('previousClose', current_price))
        
        return {
            'symbol': symbol.upper(),
            'current_price': current_price,
            'previous_close': prev_close,
            'daily_change_pct': ((current_price - prev_close) / prev_close) * 100 if prev_close else 0,
            'week_52_high': info.get('fiftyTwoWeekHigh', 0),
            'week_52_low': info.get('fiftyTwoWeekLow', 0),
        }, None
        
    except Exception as e:
        return None, f"Error fetching info for {symbol}: {str(e)}"


def calculate_sma(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        data: DataFrame with 'Close' column
        window: Number of periods for SMA
    
    Returns:
        Series with SMA values
    """
    return data['Close'].rolling(window=window).mean()


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data: DataFrame with 'Close' column
        period: RSI period (default: 14)
    
    Returns:
        Series with RSI values
    """
    delta = data['Close'].diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def check_buy_signal(data: pd.DataFrame) -> dict:
    """
    Check for "Buy the Dip" signals based on:
    1. RSI < 30 (oversold)
    2. OR price > 5% below 20-day SMA with positive 200-day SMA trend
    
    Args:
        data: DataFrame with stock price data
    
    Returns:
        Dictionary with signal information
    """
    if len(data) < 200:
        return {
            'has_signal': False,
            'reason': 'Insufficient data for analysis',
            'rsi': None,
            'price_vs_sma20': None,
            'sma200_trend': None
        }
    
    # Calculate indicators
    rsi = calculate_rsi(data)
    sma_20 = calculate_sma(data, 20)
    sma_200 = calculate_sma(data, 200)
    
    current_rsi = rsi.iloc[-1]
    current_price = data['Close'].iloc[-1]
    current_sma_20 = sma_20.iloc[-1]
    current_sma_200 = sma_200.iloc[-1]
    prev_sma_200 = sma_200.iloc[-2] if len(sma_200) > 1 else current_sma_200
    
    # Calculate percentage below 20-day SMA
    price_vs_sma20_pct = ((current_price - current_sma_20) / current_sma_20) * 100
    
    # Check if 200-day SMA trend is positive (current > previous)
    sma_200_trend_positive = current_sma_200 > prev_sma_200
    
    # Check conditions
    rsi_oversold = current_rsi < 30
    price_dip_with_trend = (price_vs_sma20_pct < -5) and sma_200_trend_positive
    
    has_signal = rsi_oversold or price_dip_with_trend
    
    reasons = []
    if rsi_oversold:
        reasons.append(f"RSI oversold ({current_rsi:.1f})")
    if price_dip_with_trend:
        reasons.append(f"Price {abs(price_vs_sma20_pct):.1f}% below 20-SMA with positive trend")
    
    return {
        'has_signal': has_signal,
        'reason': ', '.join(reasons) if reasons else 'No signal',
        'rsi': current_rsi,
        'price_vs_sma20': price_vs_sma20_pct,
        'sma200_trend': 'Positive' if sma_200_trend_positive else 'Negative',
        'current_price': current_price
    }


def scan_stocks_for_dips(stocks: list[str], progress_callback=None) -> list[dict]:
    """
    Scan multiple stocks for "Buy the Dip" signals.
    
    Args:
        stocks: List of stock symbols
        progress_callback: Optional callback for progress updates
    
    Returns:
        List of dictionaries with scan results
    """
    results = []
    
    for i, symbol in enumerate(stocks):
        if progress_callback:
            progress_callback(i + 1, len(stocks), symbol)
        
        data, error = fetch_stock_data(symbol, period="1y")
        
        if error or data is None:
            results.append({
                'symbol': symbol,
                'has_signal': False,
                'reason': error or 'No data',
                'rsi': None,
                'price_vs_sma20': None,
                'sma200_trend': None,
                'current_price': None,
                'error': True
            })
            continue
        
        signal_info = check_buy_signal(data)
        signal_info['symbol'] = symbol
        signal_info['error'] = False
        results.append(signal_info)
    
    return results
