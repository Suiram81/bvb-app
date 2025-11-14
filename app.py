
import time
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, date

def alerta_corectie_bet(bet_data):
    if bet_data is None or bet_data.empty:
        return None

    bet_data["sma50"] = bet_data["BET_Close"].rolling(50).mean()
    bet_data["sma200"] = bet_data["BET_Close"].rolling(200).mean()
    delta = bet_data["BET_Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean().replace(0, 1e-9)
    rsi14 = 100 - (100 / (1 + rs))
    bet_data["rsi14"] = rsi14

    macd = bet_data["BET_Close"].ewm(span=12, adjust=False).mean() - bet_data["BET_Close"].ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    bet_data["macd"] = macd
    bet_data["signal"] = signal

    last_rsi = rsi14.iloc[-1]
    last_macd = macd.iloc[-1]
    last_signal = signal.iloc[-1]
    last_day_change = bet_data["BET_Close"].iloc[-1] - bet_data["BET_Close"].iloc[-2]

    alerta = "Niciun semnal de corecție"
    if last_rsi > 75:
        alerta = "Atenție: RSI peste 75 – piață supracumpărată"
    if last_macd < last_signal:
        alerta = "Alertă: MACD a coborât sub semnal – potențial început de corecție"
    if last_day_change < 0 and rsi14.iloc[-2] > 70 and last_rsi < 70:
        alerta = "Semnal: început de corecție pe BET"

    return alerta
