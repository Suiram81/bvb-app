
import time
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, date

# ALERTĂ DE CORECȚIE BVB
def check_correction_alert(bet_data):
    # Returnează o listă de alerte tehnice pe baza RSI și MACD pentru indicele BET.
    alerts = []
    try:
        close = bet_data["Close"]
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(14).mean()
        ma_down = down.rolling(14).mean().replace(0, 1e-9)
        rs = ma_up / ma_down
        rsi14 = 100 - (100 / (1 + rs))

        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()

        latest_rsi = rsi14.iloc[-1]
        latest_macd = macd.iloc[-1]
        latest_signal = signal.iloc[-1]

        if latest_rsi > 70:
            alerts.append("⚠️ RSI semnalează o zonă de supracumpărare")
        if latest_macd < latest_signal:
            alerts.append("⚠️ MACD a trecut sub semnal — posibil început de corecție")
    except Exception as e:
        alerts.append(f"Eroare la calculul alertei: {str(e)}")
    return alerts
