
import time
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

APP_TITLE = "BVB Recommender Web v1"

BET_TICKERS = ["^BETI","^BET"]

TICKERS = [
    "TLV.RO","H2O.RO","SNP.RO","SNG.RO","BRD.RO",
    "TGN.RO","EL.RO","BVB.RO","DIGI.RO","FP.RO",
    "SNN.RO","TEL.RO","ONE.RO","COTE.RO","WINE.RO",
    "ALR.RO","PE.RO","ATB.RO","SMTL.RO","BIO.RO"
]

BET_PERIODS = {
    "1 zi": ("1d", "5m"),
    "5 zile": ("5d", "15m"),
    "1 luna": ("1mo", "1h"),
    "3 luni": ("3mo", "1d"),
    "6 luni": ("6mo", "1d"),
    "1 an": ("1y", "1d"),
    "5 ani": ("5y", "1wk"),
}

DEFAULT_SETTINGS = {
    "drop_alert_threshold_pct": -2.0,
    "market_drop_threshold_pct": -1.5,
    "breadth_threshold": 0.60,
    "history_days": 200,
    "momentum_lookback": 30
}

@st.cache_data(ttl=120, show_spinner=False)
def fetch_symbol(sym, history_days, momentum_lookback):
    try:
        tk = yf.Ticker(sym)
        try:
            info = tk.info
        except Exception:
            info = {}

        hist = tk.history(period=f"{history_days}d", interval="1d", auto_adjust=False, timeout=20)
        if hist is None or hist.empty:
            return None

        price_now = info.get("regularMarketPrice")
        if price_now is None:
            try:
                price_now = float(tk.fast_info.last_price)
            except Exception:
                price_now = float(hist["Close"].iloc[-1])

        prev_close = info.get("previousClose")
        if prev_close is None:
            prev_close = float(hist["Close"].iloc[-2]) if len(hist)>=2 else float(hist["Close"].iloc[-1])

        day_change = (price_now - prev_close)/prev_close*100 if prev_close else 0.0

        if len(hist) > momentum_lookback:
            p30 = float(hist["Close"].iloc[-1 - momentum_lookback])
        else:
            p30 = float(hist["Close"].iloc[0])
        momentum = (price_now/p30 - 1.0)*100 if p30 else 0.0

        returns = hist["Close"].pct_change().dropna()
        volatility = float(returns.std()*100) if not returns.empty else 0.0

        avg_volume = float(hist["Volume"].tail(30).mean()) if "Volume" in hist.columns else 0.0

        pe = info.get("trailingPE")
        dy = info.get("dividendYield")
        if dy is not None:
            dy = float(dy)*100.0

        name = info.get("shortName", sym)

        return {
            "symbol": sym,
            "name": name,
            "price": float(price_now) if price_now is not None else None,
            "day_change": float(day_change),
            "momentum": float(momentum),
            "volatility": float(volatility),
            "avg_volume": float(avg_volume),
            "pe": float(pe) if pe is not None else None,
            "yield": float(dy) if dy is not None else None,
            "history": hist.reset_index()
        }
    except Exception:
        return None

@st.cache_data(ttl=120, show_spinner=False)
def fetch_all(tickers, history_days, momentum_lookback):
    out = []
    for sym in tickers:
        d = fetch_symbol(sym, history_days, momentum_lookback)
        if d is not None:
            out.append(d)
    return out

def score_row(r):
    s = 0.0
    s += r["momentum"] * 0.5
    s += r["volatility"] * 0.1
    s += (r["avg_volume"]/1_000_000.0) * 0.2
    if r["pe"] is not None:
        s += max(0.0, 20.0 - r["pe"]) * 0.3
    if r["yield"] is not None:
        s += r["yield"] * 0.2
    return float(s)

def build_reason(r):
    parts = []
    if r["momentum"] > 0: parts.append(f"+{r['momentum']:.1f}% in 30z")
    if r["volatility"] > 5: parts.append(f"vol {r['volatility']:.1f}%")
    if r["avg_volume"] > 100_000: parts.append("volum ridicat")
    if r["pe"] is not None and r["pe"] < 15: parts.append(f"PE {r['pe']:.1f}")
    if r["yield"] is not None and r["yield"] > 2: parts.append(f"div {r['yield']:.1f}%")
    return "; ".join(parts) if parts else "-"

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
            "sma50_last": float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else None,
            "sma200_last": float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else None,
            "rsi14_last": float(rsi14.iloc[-1]) if not pd.isna(rsi14.iloc[-1]) else None,
            "macd_last": float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None,
            "signal_last": float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else None,
            "sma50_series": sma50,
            "sma200_series": sma200,
        }
    except Exception:
        return {}

def verdict(m):
    sma50 = m.get("sma50_last")
    sma200 = m.get("sma200_last")
    rsi = m.get("rsi14_last")
    macd = m.get("macd_last")
    sig = m.get("signal_last")

    reasons = []
    good = 0
    bad = 0
    if sma50 is not None and sma200 is not None:
        if sma50 > sma200:
            reasons.append("trend pozitiv SMA50 peste SMA200")
            good += 1
        else:
            reasons.append("trend slab SMA50 sub SMA200")
            bad += 1
    if rsi is not None:
        if 40 <= rsi <= 70:
            reasons.append("RSI zona ok")
            good += 1
        elif rsi < 30:
            reasons.append("RSI supravandut")
            good += 1
        else:
            reasons.append("RSI supracumparat")
            bad += 1
    if macd is not None and sig is not None:
        if macd > sig:
            reasons.append("MACD peste semnal")
            good += 1
        else:
            reasons.append("MACD sub semnal")
            bad += 1
    v = "OK de cumparat" if good >= bad else "De evitat acum"
    return v, "; ".join(reasons) if reasons else "Date insuficiente"

def bet_history(period="3mo", interval="1d"):
    for sym in BET_TICKERS:
        try:
            tk = yf.Ticker(sym)
            h = tk.history(period=period, interval=interval, timeout=20)
            if h is not None and not h.empty:
                return h
        except Exception:
            pass
    return pd.DataFrame()

st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title("BVB Recommender")
with st.sidebar:
    st.header("Setari")
    drop_thr = st.number_input("Prag scadere actiune %", value=DEFAULT_SETTINGS["drop_alert_threshold_pct"], step=0.1, format="%.2f")
    market_thr = st.number_input("Prag BET %", value=DEFAULT_SETTINGS["market_drop_threshold_pct"], step=0.1, format="%.2f")
    breadth_thr = st.number_input("Breadth scadere", value=float(DEFAULT_SETTINGS["breadth_threshold"]), step=0.05, format="%.2f")
    history_days = st.number_input("Zile istoric", value=DEFAULT_SETTINGS["history_days"], step=10)
    momentum_lb = st.number_input("Lookback momentum", value=DEFAULT_SETTINGS["momentum_lookback"], step=5)

rows = fetch_all(TICKERS, int(history_days), int(momentum_lb))
for r in rows:
    r["score"] = score_row(r)

rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)

df = pd.DataFrame([{
    "Nr": i+1,
    "Simbol": r["symbol"],
    "Denumire": r["name"],
    "Pret": round(r["price"],2) if r["price"] is not None else np.nan,
    "Delta zi %": round(r["day_change"],2),
    "Scor": round(r["score"],2),
    "Motiv": build_reason(r)
} for i, r in enumerate(rows_sorted)])

st.subheader("Recomandari ordonate 20 companii BVB")
st.dataframe(df, use_container_width=True, hide_index=True)

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Indice BET")
    choice = st.selectbox("Perioada", list(BET_PERIODS.keys()), index=2)
    period, interval = BET_PERIODS.get(choice, ("3mo","1d"))
    bet = bet_history(period, interval)
    if bet is None or bet.empty:
        st.write("Date indisponibile")
    else:
        st.line_chart(bet["Close"])

with col2:
    st.subheader("Detalii actiune")
    symbols = [r["symbol"] for r in rows_sorted]
    sel = st.selectbox("Alege simbol", symbols)
    row = next(r for r in rows_sorted if r["symbol"] == sel)
    h = row["history"].copy()
    h["Close"] = h["Close"].astype(float)
    st.metric("Pret curent RON", value=f"{row['price']:.2f}" if row['price'] is not None else "-")
    st.metric("Delta zi %", value=f"{row['day_change']:+.2f}%")
    st.metric("Momentum 30z %", value=f"{row['momentum']:+.2f}%")
    st.metric("Volatilitate %", value=f"{row['volatility']:.1f}%")
    st.metric("Volum mediu 30z", value=int(row['avg_volume']))

    st.line_chart(h.set_index("Date")["Close"] if "Date" in h.columns else h.set_index("Date") if "Date" in h.columns else h.set_index(h.columns[0])["Close"])

    m = compute_indicators(h.set_index(h.columns[0]))
    v, reason = verdict(m)
    st.write(f"Recomandare: {v}")
    st.write(f"Motiv: {reason}")
