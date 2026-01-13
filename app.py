"""
NSE Stock Market Analysis Application
A Streamlit dashboard for analyzing Indian stocks with technical indicators.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from stock_utils import (
    fetch_stock_data,
    get_stock_info,
    get_fundamental_data,
    calculate_sma,
    calculate_rsi,
    scan_stocks_for_dips,
    NIFTY_50_STOCKS
)
from nse_stocks import (
    get_all_indices,
    get_stocks_by_index,
    get_all_nse_stocks,
    get_index_count
)

# Page Configuration
st.set_page_config(
    page_title="NSE Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
    .buy-signal {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 8px;
        padding: 10px;
        color: white;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .fundamental-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #0f3460;
        margin: 5px 0;
    }
    .valuation-good { color: #00e676; }
    .valuation-caution { color: #ffab00; }
    .valuation-high { color: #ff5252; }
</style>
""", unsafe_allow_html=True)

# App Title
st.title("📈 NSE Stock Market Analyzer")
st.markdown("*Analyze Indian stocks with technical indicators, fundamental analysis, and find Buy the Dip opportunities*")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Stock Analyzer", "📊 Fundamental Analysis", "💰 Dip Finder"])

# ==================== TAB 1: STOCK ANALYZER ====================
with tab1:
    st.header("Individual Stock Analysis")
    
    # Input section
    col1, col2 = st.columns([2, 3])
    
    with col1:
        symbol = st.text_input(
            "Enter Stock Symbol",
            value="TCS",
            placeholder="e.g., TCS, INFY, RELIANCE",
            help="Enter NSE stock symbol without .NS suffix"
        ).upper().strip()
        
        analyze_btn = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)
    
    with col2:
        st.info("💡 **Tip:** Enter any NSE stock symbol like TCS, INFY, RELIANCE, HDFCBANK, etc.")
    
    if analyze_btn and symbol:
        with st.spinner(f"Fetching data for {symbol}..."):
            # Fetch stock data
            data, error = fetch_stock_data(symbol, period="1y")
            info, info_error = get_stock_info(symbol)
            
            if error:
                st.error(f"❌ {error}")
            elif data is not None and info is not None:
                # Display metrics
                st.subheader(f"📊 {symbol} - Key Metrics")
                
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric(
                        label="Current Price",
                        value=f"₹{info['current_price']:,.2f}",
                        delta=f"{info['daily_change_pct']:.2f}%"
                    )
                
                with metric_cols[1]:
                    st.metric(
                        label="52-Week High",
                        value=f"₹{info['week_52_high']:,.2f}"
                    )
                
                with metric_cols[2]:
                    st.metric(
                        label="52-Week Low",
                        value=f"₹{info['week_52_low']:,.2f}"
                    )
                
                with metric_cols[3]:
                    daily_change = info['daily_change_pct']
                    change_color = "🟢" if daily_change >= 0 else "🔴"
                    st.metric(
                        label="Daily Change",
                        value=f"{change_color} {daily_change:.2f}%"
                    )
                
                # Calculate technical indicators
                data['SMA_50'] = calculate_sma(data, 50)
                data['SMA_200'] = calculate_sma(data, 200)
                data['RSI'] = calculate_rsi(data)
                
                # Create interactive Plotly chart
                st.subheader("📈 Price History & Moving Averages (1 Year)")
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f'{symbol} Price with SMAs', 'RSI (14)')
                )
                
                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='Price',
                        increasing_line_color='#00c853',
                        decreasing_line_color='#ff1744'
                    ),
                    row=1, col=1
                )
                
                # 50-day SMA
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_50'],
                        name='50-day SMA',
                        line=dict(color='#2196F3', width=2)
                    ),
                    row=1, col=1
                )
                
                # 200-day SMA
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_200'],
                        name='200-day SMA',
                        line=dict(color='#FF9800', width=2)
                    ),
                    row=1, col=1
                )
                
                # RSI
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        name='RSI',
                        line=dict(color='#9C27B0', width=2)
                    ),
                    row=2, col=1
                )
                
                # RSI reference lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # Update layout
                fig.update_layout(
                    height=700,
                    template='plotly_dark',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                
                fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
                fig.update_yaxes(title_text="RSI", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical Summary
                st.subheader("📋 Technical Summary")
                
                current_rsi = data['RSI'].iloc[-1]
                current_sma_50 = data['SMA_50'].iloc[-1]
                current_sma_200 = data['SMA_200'].iloc[-1]
                
                summary_cols = st.columns(3)
                
                with summary_cols[0]:
                    rsi_status = "Oversold 🟢" if current_rsi < 30 else ("Overbought 🔴" if current_rsi > 70 else "Neutral ⚪")
                    st.metric("RSI (14)", f"{current_rsi:.2f}", rsi_status)
                
                with summary_cols[1]:
                    if pd.notna(current_sma_50):
                        st.metric("50-day SMA", f"₹{current_sma_50:,.2f}")
                    else:
                        st.metric("50-day SMA", "N/A")
                
                with summary_cols[2]:
                    if pd.notna(current_sma_200):
                        trend = "Bullish 🟢" if info['current_price'] > current_sma_200 else "Bearish 🔴"
                        st.metric("200-day SMA", f"₹{current_sma_200:,.2f}", trend)
                    else:
                        st.metric("200-day SMA", "N/A")

# ==================== TAB 2: FUNDAMENTAL ANALYSIS ====================
with tab2:
    st.header("📊 Fundamental Analysis")
    st.markdown("*Comprehensive fundamental metrics to evaluate stock value and financial health*")
    
    # Input section
    col1, col2 = st.columns([2, 3])
    
    with col1:
        fund_symbol = st.text_input(
            "Enter Stock Symbol",
            value="TCS",
            placeholder="e.g., TCS, INFY, RELIANCE",
            help="Enter NSE stock symbol without .NS suffix",
            key="fundamental_symbol"
        ).upper().strip()
        
        analyze_fund_btn = st.button("📊 Analyze Fundamentals", type="primary", use_container_width=True)
    
    with col2:
        st.info("💡 **Tip:** Fundamental analysis helps evaluate if a stock is undervalued or overvalued based on its financial performance.")
    
    if analyze_fund_btn and fund_symbol:
        with st.spinner(f"Fetching fundamental data for {fund_symbol}..."):
            fund_data, fund_error = get_fundamental_data(fund_symbol)
            stock_info, info_error = get_stock_info(fund_symbol)
            
            if fund_error:
                st.error(f"❌ {fund_error}")
            elif fund_data is not None:
                # Helper functions for formatting
                def format_large_number(num):
                    if num is None:
                        return "N/A"
                    if num >= 1e12:
                        return f"₹{num/1e12:.2f}T"
                    elif num >= 1e9:
                        return f"₹{num/1e9:.2f}B"
                    elif num >= 1e7:
                        return f"₹{num/1e7:.2f}Cr"
                    elif num >= 1e5:
                        return f"₹{num/1e5:.2f}L"
                    else:
                        return f"₹{num:,.2f}"
                
                def format_percentage(val):
                    if val is None:
                        return "N/A"
                    return f"{val * 100:.2f}%"
                
                def format_ratio(val):
                    if val is None:
                        return "N/A"
                    return f"{val:.2f}"
                
                # ==================== COMPANY OVERVIEW ====================
                st.subheader(f"🏢 {fund_data['company']['name']}")
                
                overview_cols = st.columns(4)
                with overview_cols[0]:
                    st.markdown(f"**Sector:** {fund_data['company']['sector']}")
                with overview_cols[1]:
                    st.markdown(f"**Industry:** {fund_data['company']['industry']}")
                with overview_cols[2]:
                    if fund_data['company']['employees']:
                        st.markdown(f"**Employees:** {fund_data['company']['employees']:,}")
                    else:
                        st.markdown("**Employees:** N/A")
                with overview_cols[3]:
                    if stock_info:
                        st.markdown(f"**Current Price:** ₹{stock_info['current_price']:,.2f}")
                
                # Company Description
                if fund_data['company']['description']:
                    with st.expander("📝 Company Description"):
                        st.write(fund_data['company']['description'])
                
                st.divider()
                
                # ==================== KEY VALUATION METRICS ====================
                st.subheader("📈 Valuation Metrics")
                
                val_cols = st.columns(5)
                valuation = fund_data['valuation']
                
                with val_cols[0]:
                    pe = valuation['trailing_pe']
                    pe_color = "🟢" if pe and pe < 20 else ("🟡" if pe and pe < 30 else "🔴") if pe else ""
                    st.metric("P/E Ratio (TTM)", format_ratio(pe), pe_color if pe else None)
                
                with val_cols[1]:
                    fwd_pe = valuation['forward_pe']
                    st.metric("Forward P/E", format_ratio(fwd_pe))
                
                with val_cols[2]:
                    pb = valuation['price_to_book']
                    pb_color = "🟢" if pb and pb < 3 else ("🟡" if pb and pb < 5 else "🔴") if pb else ""
                    st.metric("P/B Ratio", format_ratio(pb), pb_color if pb else None)
                
                with val_cols[3]:
                    st.metric("EV/EBITDA", format_ratio(valuation['ev_to_ebitda']))
                
                with val_cols[4]:
                    peg = valuation['peg_ratio']
                    peg_color = "🟢" if peg and peg < 1 else ("🟡" if peg and peg < 2 else "🔴") if peg else ""
                    st.metric("PEG Ratio", format_ratio(peg), peg_color if peg else None)
                
                # Market Cap and Enterprise Value
                val_cols2 = st.columns(4)
                with val_cols2[0]:
                    st.metric("Market Cap", format_large_number(valuation['market_cap']))
                with val_cols2[1]:
                    st.metric("Enterprise Value", format_large_number(valuation['enterprise_value']))
                with val_cols2[2]:
                    st.metric("Price to Sales", format_ratio(valuation['price_to_sales']))
                with val_cols2[3]:
                    st.metric("EV/Revenue", format_ratio(valuation['ev_to_revenue']))
                
                st.divider()
                
                # ==================== PROFITABILITY METRICS ====================
                st.subheader("💰 Profitability Metrics")
                
                prof_cols = st.columns(6)
                profitability = fund_data['profitability']
                
                with prof_cols[0]:
                    roe = profitability['return_on_equity']
                    roe_color = "🟢" if roe and roe > 0.15 else ("🟡" if roe and roe > 0.10 else "🔴") if roe else ""
                    st.metric("Return on Equity", format_percentage(roe), roe_color if roe else None)
                
                with prof_cols[1]:
                    roa = profitability['return_on_assets']
                    st.metric("Return on Assets", format_percentage(roa))
                
                with prof_cols[2]:
                    st.metric("Gross Margin", format_percentage(profitability['gross_margin']))
                
                with prof_cols[3]:
                    st.metric("Operating Margin", format_percentage(profitability['operating_margin']))
                
                with prof_cols[4]:
                    npm = profitability['profit_margin']
                    npm_color = "🟢" if npm and npm > 0.15 else ("🟡" if npm and npm > 0.08 else "🔴") if npm else ""
                    st.metric("Net Profit Margin", format_percentage(npm), npm_color if npm else None)
                
                with prof_cols[5]:
                    st.metric("EBITDA Margin", format_percentage(profitability['ebitda_margin']))
                
                # Create a bar chart for margins if data is available
                margins_data = {
                    'Gross Margin': profitability['gross_margin'],
                    'Operating Margin': profitability['operating_margin'],
                    'Net Margin': profitability['profit_margin'],
                    'EBITDA Margin': profitability['ebitda_margin']
                }
                valid_margins = {k: v * 100 for k, v in margins_data.items() if v is not None}
                
                if valid_margins:
                    import plotly.express as px
                    margin_df = pd.DataFrame({
                        'Margin Type': list(valid_margins.keys()),
                        'Percentage': list(valid_margins.values())
                    })
                    fig_margins = px.bar(
                        margin_df, 
                        x='Margin Type', 
                        y='Percentage',
                        title='Profitability Margins Comparison',
                        color='Percentage',
                        color_continuous_scale='Greens'
                    )
                    fig_margins.update_layout(
                        template='plotly_dark',
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig_margins, use_container_width=True)
                
                st.divider()
                
                # ==================== PER SHARE DATA ====================
                st.subheader("📊 Per Share Data")
                
                ps_cols = st.columns(5)
                per_share = fund_data['per_share']
                
                with ps_cols[0]:
                    st.metric("EPS (TTM)", f"₹{per_share['eps_trailing']:.2f}" if per_share['eps_trailing'] else "N/A")
                
                with ps_cols[1]:
                    st.metric("Forward EPS", f"₹{per_share['eps_forward']:.2f}" if per_share['eps_forward'] else "N/A")
                
                with ps_cols[2]:
                    st.metric("Book Value", f"₹{per_share['book_value']:.2f}" if per_share['book_value'] else "N/A")
                
                with ps_cols[3]:
                    div_yield = per_share['dividend_yield']
                    div_color = "🟢" if div_yield and div_yield > 0.02 else ""
                    st.metric("Dividend Yield", format_percentage(div_yield), div_color if div_yield else None)
                
                with ps_cols[4]:
                    st.metric("Dividend Rate", f"₹{per_share['dividend_rate']:.2f}" if per_share['dividend_rate'] else "N/A")
                
                st.divider()
                
                # ==================== FINANCIAL HEALTH ====================
                st.subheader("🏦 Financial Health")
                
                fh_cols = st.columns(5)
                fin_health = fund_data['financial_health']
                
                with fh_cols[0]:
                    dte = fin_health['debt_to_equity']
                    dte_color = "🟢" if dte and dte < 50 else ("🟡" if dte and dte < 100 else "🔴") if dte else ""
                    st.metric("Debt to Equity", format_ratio(dte), dte_color if dte else None)
                
                with fh_cols[1]:
                    cr = fin_health['current_ratio']
                    cr_color = "🟢" if cr and cr > 1.5 else ("🟡" if cr and cr > 1 else "🔴") if cr else ""
                    st.metric("Current Ratio", format_ratio(cr), cr_color if cr else None)
                
                with fh_cols[2]:
                    st.metric("Quick Ratio", format_ratio(fin_health['quick_ratio']))
                
                with fh_cols[3]:
                    st.metric("Total Cash", format_large_number(fin_health['total_cash']))
                
                with fh_cols[4]:
                    st.metric("Total Debt", format_large_number(fin_health['total_debt']))
                
                # Cash Flow metrics
                cf_cols = st.columns(3)
                with cf_cols[0]:
                    st.metric("Free Cash Flow", format_large_number(fin_health['free_cash_flow']))
                with cf_cols[1]:
                    st.metric("Operating Cash Flow", format_large_number(fin_health['operating_cash_flow']))
                with cf_cols[2]:
                    # Cash to Debt ratio if both available
                    if fin_health['total_cash'] and fin_health['total_debt'] and fin_health['total_debt'] > 0:
                        cash_to_debt = fin_health['total_cash'] / fin_health['total_debt']
                        st.metric("Cash to Debt Ratio", f"{cash_to_debt:.2f}")
                    else:
                        st.metric("Cash to Debt Ratio", "N/A")
                
                st.divider()
                
                # ==================== GROWTH METRICS ====================
                st.subheader("📈 Growth Metrics")
                
                growth_cols = st.columns(3)
                growth = fund_data['growth']
                
                with growth_cols[0]:
                    rev_growth = growth['revenue_growth']
                    rev_color = "🟢" if rev_growth and rev_growth > 0.10 else ("🟡" if rev_growth and rev_growth > 0 else "🔴") if rev_growth else ""
                    st.metric("Revenue Growth (YoY)", format_percentage(rev_growth), rev_color if rev_growth else None)
                
                with growth_cols[1]:
                    earn_growth = growth['earnings_growth']
                    earn_color = "🟢" if earn_growth and earn_growth > 0.10 else ("🟡" if earn_growth and earn_growth > 0 else "🔴") if earn_growth else ""
                    st.metric("Earnings Growth (YoY)", format_percentage(earn_growth), earn_color if earn_growth else None)
                
                with growth_cols[2]:
                    qtr_growth = growth['earnings_quarterly_growth']
                    qtr_color = "🟢" if qtr_growth and qtr_growth > 0 else "🔴" if qtr_growth else ""
                    st.metric("Quarterly Earnings Growth", format_percentage(qtr_growth), qtr_color if qtr_growth else None)
                
                st.divider()
                
                # ==================== ANALYST RECOMMENDATIONS ====================
                st.subheader("🎯 Analyst Recommendations")
                
                analyst = fund_data['analyst']
                
                if analyst['recommendation'] or analyst['target_mean']:
                    analyst_cols = st.columns(5)
                    
                    with analyst_cols[0]:
                        rec = analyst['recommendation']
                        if rec:
                            rec_display = rec.upper().replace('_', ' ')
                            rec_color = "🟢" if 'buy' in rec.lower() else ("🟡" if 'hold' in rec.lower() else "🔴")
                            st.metric("Recommendation", f"{rec_color} {rec_display}")
                        else:
                            st.metric("Recommendation", "N/A")
                    
                    with analyst_cols[1]:
                        st.metric("Target High", f"₹{analyst['target_high']:,.2f}" if analyst['target_high'] else "N/A")
                    
                    with analyst_cols[2]:
                        st.metric("Target Mean", f"₹{analyst['target_mean']:,.2f}" if analyst['target_mean'] else "N/A")
                    
                    with analyst_cols[3]:
                        st.metric("Target Low", f"₹{analyst['target_low']:,.2f}" if analyst['target_low'] else "N/A")
                    
                    with analyst_cols[4]:
                        st.metric("# Analysts", analyst['num_analysts'] if analyst['num_analysts'] else "N/A")
                    
                    # Price vs Target comparison
                    if stock_info and analyst['target_mean']:
                        current = stock_info['current_price']
                        target = analyst['target_mean']
                        upside = ((target - current) / current) * 100
                        
                        st.info(f"**Potential Upside/Downside:** {upside:+.2f}% (Current: ₹{current:,.2f} → Target: ₹{target:,.2f})")
                else:
                    st.warning("No analyst recommendations available for this stock.")

# ==================== TAB 3: DIP FINDER ====================
with tab3:
    st.header("💰 Buy the Dip Finder")
    st.markdown("""
    This screener identifies potential "Buy the Dip" opportunities based on:
    - **RSI < 30** (Oversold condition)
    - **OR** Price is **>5% below 20-day SMA** with positive 200-day SMA trend
    """)
    
    st.divider()
    
    # Index/Sector Selection
    st.subheader("📊 Select Stock Universe")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Get all available indices plus "All NSE Stocks" and "Custom" options
        index_options = ["Select an Index..."] + get_all_indices() + ["All NSE Stocks (~500+)"]
        selected_index = st.selectbox(
            "Choose Index/Sector",
            options=index_options,
            help="Select a predefined index or sector to scan"
        )
    
    with col2:
        # Show stock count for selected index
        if selected_index and selected_index not in ["Select an Index...", "All NSE Stocks (~500+)"]:
            count = get_index_count(selected_index)
            st.metric("Stocks in Index", count)
        elif selected_index == "All NSE Stocks (~500+)":
            st.metric("Stocks in Index", len(get_all_nse_stocks()))
    
    # Custom symbols input
    st.markdown("---")
    st.subheader("➕ Add Custom Symbols")
    custom_symbols_input = st.text_area(
        "Enter additional stock symbols (comma-separated)",
        placeholder="E.g., ZOMATO, PAYTM, DELHIVERY, NYKAA",
        help="Add any NSE stock symbols not in the selected index"
    )
    
    # Process custom symbols
    custom_symbols = []
    if custom_symbols_input:
        custom_symbols = [s.strip().upper() for s in custom_symbols_input.split(",") if s.strip()]
    
    # Combine stocks from index and custom
    if selected_index == "All NSE Stocks (~500+)":
        base_stocks = get_all_nse_stocks()
    elif selected_index and selected_index != "Select an Index...":
        base_stocks = get_stocks_by_index(selected_index)
    else:
        base_stocks = []
    
    # Merge and deduplicate
    all_stocks_to_scan = list(set(base_stocks + custom_symbols))
    all_stocks_to_scan.sort()
    
    # Display final stock list
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_stocks = st.multiselect(
            "Stocks to scan (you can remove any)",
            options=all_stocks_to_scan,
            default=all_stocks_to_scan,
            help="Final list of stocks to scan for dip opportunities"
        )
    
    with col2:
        st.metric("Total Stocks Selected", len(selected_stocks))
    
    with col3:
        st.write("")
        st.write("")
        if st.button("📋 Select All", use_container_width=True, key="select_all_btn"):
            selected_stocks = all_stocks_to_scan
            st.rerun()
    
    # Warning for large scans
    if len(selected_stocks) > 100:
        st.warning(f"⚡ Scanning {len(selected_stocks)} stocks may take a few minutes. Please be patient.")
    elif len(selected_stocks) > 50:
        st.info(f"ℹ️ Scanning {len(selected_stocks)} stocks. This may take a moment.")
    
    scan_btn = st.button("🔍 Scan for Dips", type="primary", use_container_width=True)
    
    if scan_btn and selected_stocks:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, symbol):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"Scanning {symbol}... ({current}/{total})")
        
        with st.spinner("Analyzing stocks..."):
            results = scan_stocks_for_dips(selected_stocks, progress_callback=update_progress)
        
        progress_bar.empty()
        status_text.empty()
        
        # Separate buy signals from others
        buy_signals = [r for r in results if r.get('has_signal') and not r.get('error')]
        no_signals = [r for r in results if not r.get('has_signal') and not r.get('error')]
        errors = [r for r in results if r.get('error')]
        
        # Display results
        if buy_signals:
            st.success(f"🎯 Found {len(buy_signals)} Buy the Dip Opportunities!")
            
            # Create DataFrame for buy signals
            buy_df = pd.DataFrame(buy_signals)
            buy_df = buy_df[['symbol', 'current_price', 'rsi', 'price_vs_sma20', 'sma200_trend', 'reason']]
            buy_df.columns = ['Symbol', 'Price (₹)', 'RSI', '% vs 20-SMA', '200-SMA Trend', 'Signal Reason']
            
            # Format the dataframe
            buy_df['Price (₹)'] = buy_df['Price (₹)'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
            buy_df['RSI'] = buy_df['RSI'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
            buy_df['% vs 20-SMA'] = buy_df['% vs 20-SMA'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            
            st.dataframe(
                buy_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Signal Reason": st.column_config.TextColumn("Signal Reason", width="large")
                }
            )
        else:
            st.warning("⚠️ No Buy the Dip signals found in the selected stocks.")
        
        # Show stocks without signals
        with st.expander(f"📊 Stocks without signals ({len(no_signals)})"):
            if no_signals:
                no_signal_df = pd.DataFrame(no_signals)
                no_signal_df = no_signal_df[['symbol', 'current_price', 'rsi', 'price_vs_sma20', 'sma200_trend']]
                no_signal_df.columns = ['Symbol', 'Price (₹)', 'RSI', '% vs 20-SMA', '200-SMA Trend']
                
                no_signal_df['Price (₹)'] = no_signal_df['Price (₹)'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
                no_signal_df['RSI'] = no_signal_df['RSI'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                no_signal_df['% vs 20-SMA'] = no_signal_df['% vs 20-SMA'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                
                st.dataframe(no_signal_df, use_container_width=True, hide_index=True)
        
        # Show errors
        if errors:
            with st.expander(f"⚠️ Errors ({len(errors)})"):
                for err in errors:
                    st.error(f"{err['symbol']}: {err['reason']}")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **NSE Stock Analyzer** helps you:
    
    - 📊 Analyze any NSE stock
    - 📈 View price charts with SMAs
    - 📋 **Fundamental Analysis**
    - 💰 Find "Buy the Dip" opportunities
    
    ---
    
    **Available Indices:**
    - Nifty 50, Next 50, Midcap 100, Smallcap 100
    - Bank Nifty, Nifty IT, Pharma, Auto
    - FMCG, Energy, Metal, Realty, PSU Bank
    - All NSE Stocks (500+)
    
    ---
    
    **Technical Indicators:**
    - **SMA 50/200**: Trend indicators
    - **RSI**: Momentum indicator
    
    ---
    
    **Fundamental Metrics:**
    - **Valuation**: P/E, P/B, EV/EBITDA, PEG
    - **Profitability**: ROE, ROA, Margins
    - **Financial Health**: Debt/Equity, Cash Flow
    - **Growth**: Revenue & Earnings Growth
    
    ---
    
    **Buy the Dip Signals:**
    - RSI < 30 (Oversold)
    - Price >5% below 20-SMA
    """)
    
    st.divider()
    st.caption("Data source: Yahoo Finance via yfinance")
    st.caption("Made with ❤️ using Streamlit")

