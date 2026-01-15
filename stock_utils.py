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


def get_fundamental_data(symbol: str) -> tuple[dict | None, str | None]:
    """
    Get comprehensive fundamental data for a stock.
    
    Args:
        symbol: Stock symbol (without .NS suffix)
    
    Returns:
        Tuple of (fundamental data dictionary, error message if any)
    """
    try:
        nse_symbol = f"{symbol.upper()}.NS"
        ticker = yf.Ticker(nse_symbol)
        info = ticker.info
        
        if not info:
            return None, f"No fundamental data found for symbol: {symbol}"
        
        # Helper function to safely get values
        def safe_get(key, default=None):
            value = info.get(key, default)
            return value if value is not None else default
        
        # Company Overview
        company_overview = {
            'name': safe_get('longName', safe_get('shortName', symbol.upper())),
            'sector': safe_get('sector', 'N/A'),
            'industry': safe_get('industry', 'N/A'),
            'website': safe_get('website', ''),
            'description': safe_get('longBusinessSummary', ''),
            'employees': safe_get('fullTimeEmployees', None),
            'country': safe_get('country', 'India'),
            'currency': safe_get('currency', 'INR'),
        }
        
        # Market Data & Valuation Metrics
        valuation = {
            'market_cap': safe_get('marketCap', None),
            'enterprise_value': safe_get('enterpriseValue', None),
            'trailing_pe': safe_get('trailingPE', None),
            'forward_pe': safe_get('forwardPE', None),
            'peg_ratio': safe_get('pegRatio', None),
            'price_to_book': safe_get('priceToBook', None),
            'price_to_sales': safe_get('priceToSalesTrailing12Months', None),
            'ev_to_revenue': safe_get('enterpriseToRevenue', None),
            'ev_to_ebitda': safe_get('enterpriseToEbitda', None),
        }
        
        # Profitability Metrics
        profitability = {
            'profit_margin': safe_get('profitMargins', None),
            'operating_margin': safe_get('operatingMargins', None),
            'gross_margin': safe_get('grossMargins', None),
            'ebitda_margin': safe_get('ebitdaMargins', None),
            'return_on_equity': safe_get('returnOnEquity', None),
            'return_on_assets': safe_get('returnOnAssets', None),
        }
        
        # Per Share Data
        per_share = {
            'eps_trailing': safe_get('trailingEps', None),
            'eps_forward': safe_get('forwardEps', None),
            'book_value': safe_get('bookValue', None),
            'revenue_per_share': safe_get('revenuePerShare', None),
            'dividend_rate': safe_get('dividendRate', None),
            'dividend_yield': safe_get('dividendYield', None),
            'payout_ratio': safe_get('payoutRatio', None),
            'five_year_avg_dividend_yield': safe_get('fiveYearAvgDividendYield', None),
        }
        
        # Financial Health
        financial_health = {
            'total_cash': safe_get('totalCash', None),
            'total_debt': safe_get('totalDebt', None),
            'debt_to_equity': safe_get('debtToEquity', None),
            'current_ratio': safe_get('currentRatio', None),
            'quick_ratio': safe_get('quickRatio', None),
            'free_cash_flow': safe_get('freeCashflow', None),
            'operating_cash_flow': safe_get('operatingCashflow', None),
        }
        
        # Income Statement Highlights
        income = {
            'total_revenue': safe_get('totalRevenue', None),
            'revenue_growth': safe_get('revenueGrowth', None),
            'gross_profit': safe_get('grossProfits', None),
            'ebitda': safe_get('ebitda', None),
            'net_income': safe_get('netIncomeToCommon', None),
            'earnings_growth': safe_get('earningsGrowth', None),
            'earnings_quarterly_growth': safe_get('earningsQuarterlyGrowth', None),
        }
        
        # Growth Metrics
        growth = {
            'revenue_growth': safe_get('revenueGrowth', None),
            'earnings_growth': safe_get('earningsGrowth', None),
            'earnings_quarterly_growth': safe_get('earningsQuarterlyGrowth', None),
        }
        
        # Analyst Recommendations
        analyst = {
            'target_high': safe_get('targetHighPrice', None),
            'target_low': safe_get('targetLowPrice', None),
            'target_mean': safe_get('targetMeanPrice', None),
            'target_median': safe_get('targetMedianPrice', None),
            'recommendation': safe_get('recommendationKey', None),
            'recommendation_mean': safe_get('recommendationMean', None),
            'num_analysts': safe_get('numberOfAnalystOpinions', None),
        }
        
        return {
            'symbol': symbol.upper(),
            'company': company_overview,
            'valuation': valuation,
            'profitability': profitability,
            'per_share': per_share,
            'financial_health': financial_health,
            'income': income,
            'growth': growth,
            'analyst': analyst,
        }, None
        
    except Exception as e:
        return None, f"Error fetching fundamental data for {symbol}: {str(e)}"


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


def get_market_trend() -> dict:
    """
    Determine the current market trend based on Nifty 50.
    
    Returns:
        Dictionary with trend status and details.
    """
    try:
        # Fetch Nifty 50 data
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="1y")
        
        if hist.empty:
            return {'status': 'Unknown', 'color': 'grey', 'reason': 'Data unavailable'}
        
        # Calculate SMAs
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        
        current_close = hist['Close'].iloc[-1]
        sma_50 = hist['SMA_50'].iloc[-1]
        sma_200 = hist['SMA_200'].iloc[-1]
        
        # Determine Trend
        if current_close > sma_200 and sma_50 > sma_200:
            status = "Uptrend"
            color = "green"
            reason = "Market is in a Confirmed Uptrend (Price > 200 DMA & 50 > 200)"
        elif current_close > sma_200:
            status = "Uptrend Under Pressure"
            color = "orange"
            reason = "Price above 200 DMA but momentum weakening"
        else:
            status = "Market in Correction"
            color = "red"
            reason = "Market is in a Downtrend (Price < 200 DMA)"
            
        return {
            'status': status,
            'color': color,
            'reason': reason,
            'current_price': current_close,
            'sma_50': sma_50,
            'sma_200': sma_200
        }
    except Exception as e:
        return {'status': 'Error', 'color': 'grey', 'reason': str(e)}


def calculate_rs_rating(stock_data: pd.DataFrame, benchmark_data: pd.DataFrame = None) -> float:
    """
    Calculate a simplified Relative Strength Rating.
    Compares stock performance vs Benchmark (Nifty 50) over 12 months.
    
    Returns:
        Float representing excess return vs benchmark.
    """
    if len(stock_data) < 250: # Need approx 1 year data
        return 0.0
        
    # If no benchmark provided, just return raw 1-yr return
    if benchmark_data is None or benchmark_data.empty:
        start_price = stock_data['Close'].iloc[0]
        end_price = stock_data['Close'].iloc[-1]
        return ((end_price - start_price) / start_price) * 100

    # Calculate Stock Return
    s_start = stock_data['Close'].iloc[0]
    s_end = stock_data['Close'].iloc[-1]
    stock_ret = ((s_end - s_start) / s_start) * 100
    
    # Calculate Benchmark Return (align dates roughly)
    b_start = benchmark_data['Close'].iloc[0]
    b_end = benchmark_data['Close'].iloc[-1]
    bench_ret = ((b_end - b_start) / b_start) * 100
    
    # RS Rating (Excess Return)
    return stock_ret - bench_ret


def check_canslim_criteria(symbol: str, data: pd.DataFrame, fundamentals: dict) -> dict:
    """
    Check CAN SLIM criteria for a stock and provide justification.
    
    Args:
        symbol: Stock symbol
        data: Historical price data (1y)
        fundamentals: Fundamental data dictionary (from get_fundamental_data)
        
    Returns:
        Dictionary with pass/fail status and detailed justification.
    """
    criteria = {
        'C': {'pass': False, 'reason': ''},
        'A': {'pass': False, 'reason': ''},
        'N': {'pass': False, 'reason': ''},
        'S': {'pass': False, 'reason': ''},
        'L': {'pass': False, 'reason': ''},
        'I': {'pass': False, 'reason': ''},
        'Overall': False,
        'Justification': []
    }
    
    if data is None or fundamentals is None:
        criteria['Justification'].append("Insufficient data for analysis")
        return criteria

    # --- C: Current Earnings ---
    # Look for quarterly growth > 20%
    growth = fundamentals.get('growth', {})
    qtr_growth = growth.get('earnings_quarterly_growth')
    
    if qtr_growth and qtr_growth > 0.20:
        criteria['C']['pass'] = True
        criteria['C']['reason'] = f"Strong Qtr EPS Growth: {qtr_growth*100:.1f}% (>20%)"
    else:
        val = f"{qtr_growth*100:.1f}%" if qtr_growth else "N/A"
        criteria['C']['reason'] = f"Weak Qtr EPS Growth: {val} (<20%)"

    # --- A: Annual Earnings ---
    # Look for annual earnings growth > 20%
    ann_growth = growth.get('earnings_growth')
    roe = fundamentals.get('profitability', {}).get('return_on_equity')
    
    if (ann_growth and ann_growth > 0.15) or (roe and roe > 0.17):
        criteria['A']['pass'] = True
        criteria['A']['reason'] = f"Solid Annual Growth: {ann_growth*100 if ann_growth else 0:.1f}% or ROE: {roe*100 if roe else 0:.1f}%"
    else:
        criteria['A']['reason'] = "Annual Growth/ROE below targets (>15% / >17%)"

    # --- N: New Highs / Catalysts ---
    # Price near 52-week high (< 15% below)
    current_price = data['Close'].iloc[-1]
    high_52 = data['High'].max()
    dist_from_high = ((high_52 - current_price) / high_52) * 100
    
    if dist_from_high < 15:
        criteria['N']['pass'] = True
        criteria['N']['reason'] = f"Price near 52W High ({dist_from_high:.1f}% below)"
    else:
        criteria['N']['reason'] = f"Price >15% below 52W High ({dist_from_high:.1f}%)"

    # --- S: Supply & Demand ---
    # Volume check > 5 Cr daily value (approx) to ensure liquidity
    # And ideally rising volume on up days
    avg_vol = data['Volume'].tail(20).mean()
    avg_price = data['Close'].tail(20).mean()
    avg_turnover = avg_vol * avg_price
    
    if avg_turnover > 50_000_000: # 5 Crores
        criteria['S']['pass'] = True
        criteria['S']['reason'] = f"Liquid: Avg Daily Turnover ₹{avg_turnover/1e7:.1f}Cr"
    else:
        criteria['S']['reason'] = f"Illiquid: Avg Turnover ₹{avg_turnover/1e7:.1f}Cr (<5Cr)"

    # --- L: Leader (Relative Strength) ---
    # 1 Year Performance vs Market
    # Ideally should pass if stock return > 20% in last year or outperforms index
    start_price = data['Close'].iloc[0]
    stock_ret = ((current_price - start_price) / start_price)
    
    if stock_ret > 0.20: # Absolute return > 20%
        criteria['L']['pass'] = True
        criteria['L']['reason'] = f"Relative Strength Leader: +{stock_ret*100:.1f}% in 1yr"
    elif stock_ret > 0:
        criteria['L']['reason'] = f"Positive but weak Returns: +{stock_ret*100:.1f}%"
    else:
        criteria['L']['reason'] = f"Laggard: {stock_ret*100:.1f}% return"

    # --- I: Institutional Sponsorship ---
    # Hard to get real-time FII data via yfinance free API
    # We check if 'heldPercentInstitutions' exists
    inst_hold = fundamentals.get('major_holders', {}).get('heldPercentInstitutions') # This key might not exist in our struct
    # Fallback: Check if Market Cap is decent (> 2000 Cr) implying inst interest
    mkt_cap = fundamentals.get('valuation', {}).get('market_cap', 0)
    
    if mkt_cap and mkt_cap > 20_000_000_000: # 2000 Cr
        criteria['I']['pass'] = True
        criteria['I']['reason'] = f"Inst. Base Likely (Large Cap: ₹{mkt_cap/1e9:.1f}B)"
    else:
         criteria['I']['reason'] = "Small Cap / Unknown Inst. Sponsorship"

    # --- Calculate Justification ---
    # We advocate for the trade if C, A, N, and L are strong (The Core)
    score = sum([1 for k in ['C', 'A', 'N', 'S', 'L', 'I'] if criteria[k]['pass']])
    
    justification = []
    if score >= 5:
        criteria['Overall'] = True
        justification.append(f"🔥 **STRONG BUY CANDIDATE** ({score}/6 Criteria)")
        justification.append("Details:")
        justification.append(f"- ✅ **Earnings**: {criteria['C']['reason']}")
        justification.append(f"- ✅ **Trend**: {criteria['N']['reason']} & {criteria['L']['reason']}")
    elif score >= 3:
        criteria['Overall'] = False # Strict CAN SLIM needs more
        justification.append(f"⚠️ **WATCHLIST CANDIDATE** ({score}/6 Criteria)")
        justification.append("Missing some key criteria:")
        if not criteria['C']['pass']: justification.append(f"- ❌ **Earnings**: {criteria['C']['reason']}")
        if not criteria['N']['pass']: justification.append(f"- ❌ **Trend**: {criteria['N']['reason']}")
    else:
        criteria['Overall'] = False
        justification.append(f"⛔ **AVOID** ({score}/6 Criteria)")
        justification.append(f"- Weak Technicals/Fundamentals.")

    criteria['Justification'] = justification
    
    return criteria
