
# app_bvb_alerta.py

import time
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, date

APP_TITLE = "BVB Recommender Web v1.9.2 + Alertă corecție"

BET_TICKERS = ["^BETI","^BET"]
BET_CONSTITUENTS = [
    "ATB.RO","AQ.RO","TLV.RO","BRD.RO","TEL.RO","DIGI.RO","FP.RO","M.RO",
    "SNP.RO","ONE.RO","PE.RO","WINE.RO","SNN.RO","SNG.RO","TGN.RO","H2O.RO",
    "EL.RO","SFG.RO","TRP.RO","TTS.RO"
]
ETF_TICKERS = ["TVBETETF.RO", "PTENGETF.RO"]
TICKERS = BET_CONSTITUENTS + ETF_TICKERS
BET_PERIODS = {
    "1 zi": ("1d", "5m"), "5 zile": ("5d", "15m"), "1 luna": ("1mo", "1d"),
    "3 luni": ("3mo", "1d"), "6 luni": ("6mo", "1d"),
    "1 an": ("1y", "1d"), "5 ani": ("5y", "1wk"),
}
DEFAULT_SETTINGS = {"history_days": 250, "momentum_lookback": 30}
USER_PORTFOLIO = {"TLV.RO","SNP.RO","H2O.RO","EL.RO"}
TAX_SWITCHOVER = date(2026, 1, 1)

def net_rate_for_date(dstr):
    try:
        d = datetime.fromisoformat(str(dstr)).date()
    except Exception:
        d = date.today()
    return 0.84 if d >= TAX_SWITCHOVER else 0.92

@st.cache_data(ttl=120, show_spinner=False)
def fetch_symbol(sym, history_days, momentum_lookback):
    try:
        tk = yf.Ticker(sym)
        try: info = tk.info
        except: info = {}
        hist = tk.history(period=f"{history_days}d", interval="1d", auto_adjust=False, timeout=20)
        if hist is None or hist.empty: return None
        price_now = info.get("regularMarketPrice") or float(tk.fast_info.last_price)
        prev_close = info.get("previousClose") or float(hist["Close"].iloc[-2])
        day_change = (price_now - prev_close)/prev_close*100 if prev_close else 0.0
        p30 = float(hist["Close"].iloc[-1 - momentum_lookback]) if len(hist) > momentum_lookback else float(hist["Close"].iloc[0])
        momentum = (price_now/p30 - 1.0)*100 if p30 else 0.0
        returns = hist["Close"].pct_change().dropna()
        volatility = float(returns.std()*100) if not returns.empty else 0.0
        avg_volume = float(hist["Volume"].tail(30).mean()) if "Volume" in hist.columns else 0.0
        pe = info.get("trailingPE")
        try: dividends = tk.dividends
        except: dividends = None
        last_dividend = last_div_date = last_div_net_pct = None
        if dividends is not None and not dividends.empty:
            try:
                last_dividend = float(dividends.iloc[-1])
                last_div_date = str(dividends.index[-1].date())
                if price_now and last_dividend:
                    net_rate = net_rate_for_date(last_div_date)
                    last_div_net_pct = float(last_dividend * net_rate / float(price_now) * 100.0)
            except: pass
        name = info.get("shortName", sym)
        return {
            "symbol": sym, "name": name, "price": float(price_now),
            "day_change": float(day_change), "momentum": float(momentum),
            "volatility": float(volatility), "avg_volume": float(avg_volume),
            "pe": float(pe) if pe is not None else None,
            "yield": float(last_div_net_pct) if last_div_net_pct is not None else None,
            "last_dividend_net_pct": float(last_div_net_pct) if last_div_net_pct is not None else None,
            "last_div_date": last_div_date,
            "history": hist.reset_index()
        }
    except: return None

@st.cache_data(ttl=120, show_spinner=False)
def fetch_bet_data():
    for sym in BET_TICKERS:
        try:
            tk = yf.Ticker(sym)
            h = tk.history(period="6mo", interval="1d")
            if h is not None and not h.empty:
                return h
        except: pass
    return pd.DataFrame()

def compute_indicators(hist_df):
    try:
        close = hist_df["Close"]
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ma_up = up.rolling(14).mean()
        ma_down = down.rolling(14).mean().replace(0, 1e-9)
        rs = ma_up / ma_down
        rsi14 = 100 - (100/(1+rs))
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        return {
            "sma50_last": float(sma50.iloc[-1]),
            "sma200_last": float(sma200.iloc[-1]),
            "rsi14_last": float(rsi14.iloc[-1]),
            "macd_last": float(macd.iloc[-1]),
            "signal_last": float(signal.iloc[-1]),
        }
    except:
        return {}

def alerta_corectie_bet(bet_hist):
    if bet_hist is None or bet_hist.empty:
        return "Date BET indisponibile."
    indic = compute_indicators(bet_hist)
    if not indic:
        return "Date insuficiente pentru alertă."
    alerts = []
    if indic["rsi14_last"] > 70:
        alerts.append("RSI peste 70 - posibil supracumpărat.")
    if indic["macd_last"] < indic["signal_last"]:
        alerts.append("MACD sub semnal - semnal de slăbiciune.")
    if indic["sma50_last"] < indic["sma200_last"]:
        alerts.append("SMA50 a coborât sub SMA200 - risc de corecție.")
    return alerts if alerts else ["Niciun semnal major de corecție."]

# Afișare în aplicație
st.title("Alerte corecție tehnică")
bet_hist = fetch_bet_data()
mesaje = alerta_corectie_bet(bet_hist)
for m in mesaje:
    st.warning(m)
