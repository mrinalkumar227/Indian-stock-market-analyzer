# Indian-stock-market-analyzer# NSE Stock Market Analyzer 📈

A Python-based stock analysis dashboard for Indian stocks (NSE) with technical indicators and algorithmic trading signals.

## Features

### 🔍 Stock Analyzer

- Real-time stock data from Yahoo Finance
- Key metrics: Current Price, 52-Week High/Low, Daily Change
- Interactive Plotly charts with candlestick patterns
- Technical indicators: 50-day & 200-day SMA, RSI (14)

### 💰 Dip Finder

Algorithmic screener for "Buy the Dip" opportunities:

- RSI < 30 (Oversold condition)
- Price > 5% below 20-day SMA with positive 200-day trend
- Scans Nifty 50 stocks in one click

## Quick Start

```bash
# Clone and navigate to the project
cd nse-stock-analyzer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deployment to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository and `app.py` as the main file
5. Click Deploy!

## Tech Stack

- **Streamlit** - Web dashboard
- **yfinance** - Stock data API
- **Plotly** - Interactive charts
- **Pandas/NumPy** - Data processing

## File Structure

```
nse-stock-analyzer/
├── app.py              # Main Streamlit application
├── stock_utils.py      # Utility functions & indicators
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Streamlit configuration
└── README.md           # This file
```

## License

MIT License - Feel free to use and modify!
