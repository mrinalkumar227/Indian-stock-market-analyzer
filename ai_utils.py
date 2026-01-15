
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

def prepare_ai_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical indicators as features for the AI model.
    Optimization: Vectorized operations for speed.
    """
    df = data.copy()
    
    # 1. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. MACD (12, 26, 9)
    # Using simple EMAs
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 3. Bollinger Band Width
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    upper_band = sma_20 + (std_20 * 2)
    lower_band = sma_20 - (std_20 * 2)
    df['BB_Width'] = (upper_band - lower_band) / sma_20
    
    # 4. Returns (Momentum)
    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_5'] = df['Close'].pct_change(5)
    
    # 5. Volatility (ATR-ish proxy)
    df['Range'] = (df['High'] - df['Low']) / df['Close']
    
    # Target: 1 if Next Close > Current Close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df.dropna()

def train_intraday_model(symbol: str):
    """
    Fetch intraday data, train XGBoost model, and return model + recent data.
    """
    try:
        ticker = f"{symbol.upper()}"
        if not ticker.endswith('.NS'):
            ticker += ".NS"
            
        # Fetch 60 days of 15m data (max allowed by yfinance for 15m)
        # For 5m data, max is 60 days.
        data = yf.download(ticker, period='60d', interval='15m', progress=False)
        
        if data.empty or len(data) < 100:
            return None, None, "Insufficient intraday data (need >100 candles)"

        # Clean MultiIndex columns if present (yfinance update)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Feature Engineering
        df_model = prepare_ai_features(data)
        
        if len(df_model) < 50:
            return None, None, "Not enough data after feature engineering"

        # Define Features & Target
        feature_cols = ['RSI', 'MACD', 'Signal_Line', 'BB_Width', 'Return_1', 'Return_5', 'Range']
        X = df_model[feature_cols]
        y = df_model['Target']
        
        # Split Train/Test (Time-based split, no shuffle)
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            objective='binary:logistic',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        return model, {
            'accuracy': acc, 
            'last_data': df_model.iloc[-1].to_dict(), 
            'feature_cols': feature_cols
        }, None

    except Exception as e:
        return None, None, str(e)

def predict_next_move(model, last_data_point, feature_cols):
    """
    Predict probability of UP move for the next candle.
    """
    try:
        # Construct single-row DataFrame
        input_data = pd.DataFrame([last_data_point], columns=feature_cols)
        
        # Predict Probability
        # proba[0][1] is probability of class 1 (UP)
        prob_up = model.predict_proba(input_data)[0][1]
        
        return prob_up
    except Exception as e:
        return 0.5
