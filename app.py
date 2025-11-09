
import time
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

APP_TITLE = "BVB Recommender Web v1.7 (full + Scara Tradeville)"

BET_TICKERS = ["^BETI","^BET"]

# 20 constituenti oficiali BET (noiembrie 2025)
TICKERS = [
    "ATB.RO","AQ.RO","TLV.RO","BRD.RO","TEL.RO","DIGI.RO","FP.RO","M.RO",
    "SNP.RO","ONE.RO","PE.RO","WINE.RO","SNN.RO","SNG.RO","TGN.RO","H2O.RO",
    "EL.RO","SFG.RO","TRP.RO","TTS.RO"
]

BET_PERIODS = {
    "1 zi": ("1d", "5m"),
    "5 zile": ("5d", "15m"),
    "1 luna": ("1mo", "1d"),
    "3 luni": ("3mo", "1d"),
    "6 luni": ("6mo", "1d"),
    "1 an": ("1y", "1d"),
    "5 ani": ("5y", "1wk"),
}

DEFAULT_SETTINGS = {
    "history_days": 250,
    "momentum_lookback": 30
}

DIVIDEND_TAX_NET_RATE = 0.92
USER_PORTFOLIO = {"TLV.RO","SNP.RO","H2O.RO","EL.RO"}

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

        try:
            dividends = tk.dividends
        except Exception:
            dividends = None
        last_dividend = None
        last_div_date = None
        last_div_net_pct = None
        if dividends is not None and not dividends.empty:
            try:
                last_dividend = float(dividends.iloc[-1])
                last_div_date = str(dividends.index[-1].date())
                if price_now:
                    last_div_net_pct = float(last_dividend * DIVIDEND_TAX_NET_RATE / float(price_now) * 100.0)
            except Exception:
                pass

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
            "yield": float(last_div_net_pct) if last_div_net_pct is not None else None,
            "last_dividend_net_pct": float(last_div_net_pct) if last_div_net_pct is not None else None,
            "last_div_date": last_div_date,
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
            h = tk.history(period=period, interval=interval, timeout=20, auto_adjust=False)
            if h is not None and not h.empty:
                return h[["Close"]].rename(columns={"Close": "BET_Close"}), "BET Yahoo"
        except Exception:
            pass
    return None, "NA"

def compute_bet_simulare(rows_sorted, period_key):
    period_days = {"1 zi":1, "5 zile":5, "1 luna":30, "3 luni":90, "6 luni":180, "1 an":365, "5 ani":365*5}
    days = period_days.get(period_key, 90)
    series = []
    for r in rows_sorted:
        h = r["history"].copy()
        if h.empty:
            continue
        h = h.rename(columns={h.columns[0]: "Date"})
        h = h.set_index("Date")
        s = h["Close"].astype(float).copy()
        s = s.iloc[-days:] if len(s) > days else s
        if len(s) == 0:
            continue
        base = s.iloc[0]
        if base == 0 or np.isnan(base):
            continue
        s_norm = s / base * 100.0
        series.append(s_norm.rename(r["symbol"]))
    if not series:
        return None
    dfw = pd.concat(series, axis=1, join="inner").dropna(how="all")
    if dfw.empty:
        return None
    simulare = dfw.mean(axis=1).to_frame(name="BET_Close")
    return simulare

def compute_recommendations(rows_sorted):
    scores = [r["score"] for r in rows_sorted if "score" in r]
    if len(scores) >= 4:
        q25, q75 = np.percentile(scores, [25, 75])
    else:
        q25, q75 = (np.min(scores), np.max(scores))
    rec_map = {}
    for r in rows_sorted:
        h = r["history"].copy()
        h = h.set_index(h.columns[0])
        v_text, _ = verdict(compute_indicators(h))
        s = r["score"]
        if v_text == "OK de cumparat" and s >= q75:
            rec = "Cumpara"
        elif s >= q25 and s < q75:
            rec = "Mentine"
        elif v_text == "De evitat acum" and s < q25:
            rec = "Vinde"
        else:
            rec = "Evalueaza"
        if r["symbol"] in USER_PORTFOLIO and rec == "Cumpara":
            rec = "Mentine"
        rec_map[r["symbol"]] = rec
    return rec_map

st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title("BVB Recommender")

with st.sidebar:
    st.header("Setari")
    history_days = st.number_input("Zile istoric", value=DEFAULT_SETTINGS["history_days"], step=10)
    momentum_lb = st.number_input("Lookback momentum", value=DEFAULT_SETTINGS["momentum_lookback"], step=5)

# date principale
rows = fetch_all(TICKERS, int(history_days), int(momentum_lb))
for r in rows:
    r["score"] = score_row(r)
rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)
rec_map = compute_recommendations(rows_sorted)

# tabel principal
st.subheader("Recomandari ordonate 20 companii BET")
df = pd.DataFrame([{
    "Nr": i+1,
    "Simbol": r["symbol"],
    "Denumire": r["name"],
    "Pret": round(r["price"],2) if r["price"] is not None else np.nan,
    "Delta zi %": round(r["day_change"],2),
    "Dividend net %": (round(r["last_dividend_net_pct"],2) if r.get("last_dividend_net_pct") is not None else np.nan),
    "Data dividend": r.get("last_div_date") if r.get("last_div_date") else "",
    "Scor": round(r["score"],2),
    "Recomandare": rec_map.get(r["symbol"], "Evalueaza"),
    "Motiv": build_reason(r)
} for i, r in enumerate(rows_sorted)])
st.dataframe(df, use_container_width=True, hide_index=True)

# coloane: BET + detalii actiune
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Indice BET")
    choice = st.selectbox("Perioada", list(BET_PERIODS.keys()), index=5, key="bet_period")
    period, interval = BET_PERIODS.get(choice, ("1y","1d"))
    bet_yahoo, _ = bet_history(period, interval)
    simulare = compute_bet_simulare(rows_sorted, choice)

    mode = st.selectbox("Mod afisare", ["Scara Tradeville", "Nivel oficial Yahoo", "Simulare BET normalizat 100"])

    if mode == "Scara Tradeville":
        data = simulare if simulare is not None else bet_yahoo
        anchor_default = float(bet_yahoo['BET_Close'].iloc[-1]) if bet_yahoo is not None else 22865.87
        anchor = st.number_input("Valoare curenta BET (Tradeville)", value=float(round(anchor_default,2)))
        label = "Sursa: Simulare BET ancorata la nivel Tradeville" if simulare is not None else "Sursa: BET Yahoo"
        if data is not None and not data.empty:
            # ancoreaza seria la valoarea dorita
            scale = anchor / float(data['BET_Close'].iloc[-1])
            data = data.copy()
            data['BET_Close'] = data['BET_Close'] * scale
            # calculeaza Var si Var%
            if len(data) >= 2:
                last = float(data['BET_Close'].iloc[-1])
                prev = float(data['BET_Close'].iloc[-2])
                var = last - prev
                varpct = (var / prev * 100.0) if prev else 0.0
            else:
                last = anchor
                var = 0.0
                varpct = 0.0
            # afiseaza metricele in stil Tradeville
            m1, m2, m3 = st.columns(3)
            m1.metric("Valoare", f"{last:,.2f}".replace(","," ").replace(".",","))
            m2.metric("Var", f"{var:+.2f}".replace(".",","))
            m3.metric("Var%", f"{varpct:+.2f}%".replace(".",","))
        else:
            st.write("Date indisponibile")
    elif mode == "Nivel oficial Yahoo":
        data = bet_yahoo
        label = "Sursa: BET Yahoo"
    else:
        data = simulare
        label = "Sursa: Simulare BET (100 la start)"

    if mode != "Scara Tradeville":
        if data is None or data.empty:
            st.write("Date indisponibile")
        else:
            # calculeaza si afiseaza metricele aici pe baza seriei
            if len(data) >= 2:
                last = float(data['BET_Close'].iloc[-1])
                prev = float(data['BET_Close'].iloc[-2])
                var = last - prev
                varpct = (var / prev * 100.0) if prev else 0.0
            else:
                last = data['BET_Close'].iloc[-1] if len(data) else 0.0
                var = 0.0
                varpct = 0.0
            m1, m2, m3 = st.columns(3)
            m1.metric("Valoare", f"{last:,.2f}".replace(","," ").replace(".",","))
            m2.metric("Var", f"{var:+.2f}".replace(".",","))
            m3.metric("Var%", f"{varpct:+.2f}%".replace(".",","))

    if data is not None and not data.empty:
        st.caption(label)
        st.line_chart(data["BET_Close"])

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
    st.metric("Ultimul dividend net %", value=(f"{row['last_dividend_net_pct']:.2f}%" if row.get('last_dividend_net_pct') is not None else "-"))
    st.metric("Data dividend", value=(row.get('last_div_date') or "-"))
    st.line_chart(h.set_index("Date")["Close"] if "Date" in h.columns else h.set_index(h.columns[0])["Close"])
