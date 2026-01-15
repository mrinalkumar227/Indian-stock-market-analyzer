
import sys
import os
import pandas as pd
sys.path.append(os.getcwd())

from ai_utils import train_intraday_model, predict_next_move
from nse_stocks import get_stocks_by_index

def scan_market():
    print("🚀 Starting AI Scan for Nifty 50...")
    stocks = get_stocks_by_index("Nifty 50")
    
    results = []
    
    # Analyze first 20 for speed in this demo, or all if fast enough
    # Let's do all, it might take 60s
    count = 0
    for symbol in stocks:
        count += 1
        print(f"[{count}/{len(stocks)}] Analyzing {symbol}...", end="\r")
        
        try:
            model, train_info, error = train_intraday_model(symbol)
            if not error:
                prob_up = predict_next_move(model, train_info['last_data'], train_info['feature_cols'])
                
                # We only care about strong signals for the "Trade" recommendation
                if prob_up > 0.60 or prob_up < 0.40:
                    signal = "Bullish" if prob_up > 0.5 else "Bearish"
                    conf = prob_up if prob_up > 0.5 else (1 - prob_up)
                    results.append({
                        'Symbol': symbol,
                        'Signal': signal,
                        'Confidence': conf
                    })
        except Exception as e:
            pass

    print("\n\n✅ Scan Complete.")
    
    if results:
        df = pd.DataFrame(results).sort_values(by='Confidence', ascending=False)
        print("\n🔥 TOP AI PICKS:")
        print(df.head(10).to_string(index=False))
    else:
        print("No strong signals found right now.")

if __name__ == "__main__":
    scan_market()
