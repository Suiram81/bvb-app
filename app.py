
import time
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
import streamlit as st
from datetime import datetime, date

APP_TITLE = "BVB Recommender Web v1.9.2 (fix PTENGETF motiv + taxe 2026 + ETF-uri)"
BET_TICKERS = ["^BETI","^BET"]
BET_SCALE = 176.0  # factor implicit; va fi recalibrat dinamic dupa valoarea BET de pe BVB

BET_CONSTITUENTS = [
    "ATB.RO","AQ.RO","TLV.RO","BRD.RO","TEL.RO","DIGI.RO","FP.RO","M.RO",
    "SNP.RO","ONE.RO","PE.RO","WINE.RO","SNN.RO","SNG.RO","TGN.RO","H2O.RO",
    "EL.RO","SFG.RO","TRP.RO","TTS.RO"
]



AERO_CONSTITUENTS = {
    "DN.RO": "DN Agrar Group",
    "CMVX.RO": "COMVEX SA CONSTANTA",
    "BUCU.RO": "BUCUR OBOR SA BUCURESTI",
    "MET.RO": "META ESTATE TRUST",
    "AG.RO": "AGROLAND BUSINESS SYSTEM",
    "SCDM.RO": "UNIREA SHOPPING CENTER SA BUCURESTI",
    "IPRU.RO": "IPROEB SA BISTRITA",
    "MACO.RO": "MACOFIL SA TG. JIU",
    "FOJE.RO": "FORAJ SONDE SA VIDELE",
    "BENTO.RO": "2B Intelligent Soft",
    "PRSN.RO": "PROSPECTIUNI SA BUCURESTI",
    "HAI.RO": "Holde Agri Invest S.A. - Clasa A",
    "BUCV.RO": "BUCUR SA BUCURESTI",
    "GSH.RO": "Grup Serban Holding",
    "NRF.RO": "NOROFERT S.A.",
    "CC.RO": "CONNECTIONS CONSULT S.A.",
    "ASC.RO": "ASCENDIA S.A.",
    "JTG.RO": "JT GRUP OIL",
    "ALW.RO": "VISUAL FAN",
    "AST.RO": "ARCTIC STREAM",
    "AGRO.RO": "AGROSERV MARIUTA",
    "MIBO.RO": "Millenium Insurance Broker",
    "ATRD.RO": "ATELIERE CFR GRIVITA SA BUCURESTI",
    "HUNT.RO": "IHUNT TECHNOLOGY IMPORT-EXPORT",
    "2P.RO": "2PERFORMANT NETWORK",
    "BRNA.RO": "ROMNAV SA BRAILA",
    "SPX.RO": "SIPEX COMPANY",
    "AVIO.RO": "AVIOANE SA CRAIOVA",
    "LIH.RO": "LIFE IS HARD S.A.",
    "CLAIM.RO": "AIR CLAIM",
    "CODE.RO": "Softbinator Technologies",
    "REIT.RO": "Star Residence Invest",
    "BONA.RO": "BONAS IMPORT EXPORT",
    "MAM.RO": "MAMBRICOLAJ S.A.",
    "FRB.RO": "FIREBYTE GAMES",
}
AERO_TICKERS = list(AERO_CONSTITUENTS.keys())
ETF_TICKERS = ["TVBETETF.RO", "PTENGETF.RO"]

TICKERS = BET_CONSTITUENTS + ETF_TICKERS + AERO_TICKERS

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

USER_PORTFOLIO = {"TLV.RO","SNP.RO","H2O.RO","EL.RO"}

TAX_SWITCHOVER = date(2026, 1, 1)

def fetch_bet_last_from_bvb():
    """Incearca sa citeasca valoarea curenta a indicelui BET direct de pe site-ul BVB.

    Returneaza un float cu nivelul indicelui sau None daca nu reuseste.
    """
    try:
        import requests
        urls = [
            "https://www.bvb.ro/",
            "https://m.bvb.ro/financialinstruments/indices/indicesprofiles",
        ]
        for url in urls:
            try:
                r = requests.get(url, timeout=10)
            except Exception:
                continue
            if r.status_code != 200:
                continue
            html = r.text
            # incercam sa gasim tabele HTML
            try:
                tables = pd.read_html(html, decimal=",", thousands=".")
                for df in tables:
                    if df.empty or df.shape[1] < 2:
                        continue
                    # cautam randul care contine "BET"
                    mask = df.apply(lambda col: col.astype(str).str.contains(r"\bBET\b", case=False, regex=True))
                    rows = df[mask.any(axis=1)]
                    if not rows.empty:
                        # extragem toate valorile numerice din acel rand
                        vals = []
                        for v in rows.iloc[0].values:
                            s = str(v)
                            s = s.replace("\xa0", " ").strip()
                            try:
                                num = float(s.replace(".", "").replace(",", "."))
                                vals.append(num)
                            except Exception:
                                continue
                        if vals:
                            candidate = max(vals)
                            if candidate > 1000:
                                return candidate
            except Exception:
                pass

            # fallback: expresie regulata direct in HTML
            import re as _re
            m = _re.search(r"BET[^0-9]*([0-9\.\,]{4,})", html)
            if m:
                try:
                    s = m.group(1)
                    val = float(s.replace(".", "").replace(",", "."))
                    if val > 1000:
                        return val
                except Exception:
                    pass
    except Exception:
        return None
    return None
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
                if price_now and last_dividend:
                    net_rate = net_rate_for_date(last_div_date)
                    last_div_net_pct = float(last_dividend * net_rate / float(price_now) * 100.0)
            except Exception:
                pass

        try:
            pred_next_price, pred_next_change_pct = predict_next_day_linear(hist)
        except Exception:
            pred_next_price, pred_next_change_pct = None, None

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
            "history": hist.reset_index(),
            "pred_next_price": float(pred_next_price) if pred_next_price is not None else None,
            "pred_next_change_pct": float(pred_next_change_pct) if pred_next_change_pct is not None else None
        }
    except Exception:
        return None

@st.cache_data(ttl=120, show_spinner=False)
def fetch_all(tickers, history_days, momentum_lookback):
    out = []
    for sym in tickers:
        d = fetch_symbol(sym, history_days, momentum_lookback)
        if d is not None:
            d["no_data"] = False
            out.append(d)
        else:
            # adaugam placeholder pentru PTENGETF.RO cand lipsesc date
            if sym == "PTENGETF.RO":
                out.append({
                    "symbol": sym,
                    "name": "PTENGETF",
                    "price": None,
                    "day_change": 0.0,
                    "momentum": 0.0,
                    "volatility": 0.0,
                    "avg_volume": 0.0,
                    "pe": None,
                    "yield": None,
                    "last_dividend_net_pct": None,
                    "last_div_date": None,
                    "pred_next_price": None,
                    "pred_next_change_pct": None,
                    "history": pd.DataFrame(),
                    "no_data": True
                })
    return out

def score_row(r):
    if r.get("no_data"):
        return float("nan")
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
    if r.get("no_data"):
        return "nu ai date de a genera raportul"
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
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(14).mean()
        ma_down = down.rolling(14).mean().replace(0, 1e-9)
        rs = ma_up / ma_down
        rsi14 = 100 - (100 / (1 + rs))
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()

        max_60 = close.tail(60).max()
        if pd.notna(max_60) and max_60 > 0:
            drawdown_60 = (close.iloc[-1] / max_60 - 1.0) * 100.0
        else:
            drawdown_60 = None

        return {
            "sma50_last": float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else None,
            "sma200_last": float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else None,
            "rsi14_last": float(rsi14.iloc[-1]) if not pd.isna(rsi14.iloc[-1]) else None,
            "macd_last": float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None,
            "signal_last": float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else None,
            "drawdown_60": float(drawdown_60) if drawdown_60 is not None else None,
        }
    except Exception:
        return {}

def predict_next_day_linear(hist_df, min_points=15):
    """Model hibrid pe termen scurt calibrat pentru BVB.
    - 40% trend pe ultimele 20 de zile
    - 60% smoothing pe ultimele 5 zile
    - limitare dupa volatilitatea zilnica
    """
    try:
        closes = hist_df["Close"].astype(float).dropna()
        if len(closes) < min_points:
            return None, None

        # Pretul curent
        last_price = float(closes.iloc[-1])
        if last_price <= 0:
            return None, None

        # Trend pe ultimele 20 de zile
        tail_trend = closes.tail(min(20, len(closes)))
        n_trend = len(tail_trend)
        x_trend = np.arange(n_trend, dtype=float)
        y_trend = tail_trend.values
        a, b = np.polyfit(x_trend, y_trend, 1)
        trend_next = a * float(n_trend) + b

        # Smoothing pe ultimele 5 zile
        tail_smooth = closes.tail(min(5, len(closes)))
        smoothed = tail_smooth.ewm(alpha=0.5, adjust=False).mean().iloc[-1]

        # Baza: combinatie 40% trend, 60% smoothing
        base_next = float(0.4 * trend_next + 0.6 * smoothed)

        # Volatilitatea ultimelor 20 de zile (pct_change)
        rets = closes.pct_change().dropna()
        if len(rets) > 0:
            vol20 = rets.tail(min(20, len(rets))).std()
        else:
            vol20 = 0.0

        # Limitare miscari la aproximativ +/- 1.5 deviatii standard
        if vol20 and not np.isnan(vol20):
            max_up = last_price * (1.0 + 1.5 * vol20)
            max_down = last_price * (1.0 - 1.5 * vol20)
            base_next = max(min(base_next, max_up), max_down)

        # Daca volatilitatea este foarte mica, limitam oricum intre -2% si +2%
        if vol20 is not None and not np.isnan(vol20) and vol20 < 0.02:
            cap_up = last_price * 1.02
            cap_down = last_price * 0.98
            base_next = max(min(base_next, cap_up), cap_down)

        change_pct = (base_next / last_price - 1.0) * 100.0
        return float(base_next), float(change_pct)
    except Exception:
        return None, None


def bet_arima_forecast(df, steps=1):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception:
        return None
    try:
        series = df["BET_Close"].astype(float).dropna()
        if len(series) < 50:
            return None
        model = ARIMA(series, order=(1, 1, 1))
        res = model.fit()
        forecast = res.forecast(steps=steps)
        if len(forecast) == 0:
            return None
        return float(forecast.iloc[-1])
    except Exception:
        return None


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
        if r["symbol"] not in BET_CONSTITUENTS:
            continue
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
            rec = "🔍 Insuficiente date pentru a face analiza"
        if r["symbol"] in USER_PORTFOLIO and rec == "Cumpara":
            rec = "Mentine"
        rec_map[r["symbol"]] = rec
    return rec_map

def compute_bet_alert(indicators):
    rsi = indicators.get("rsi14_last")
    macd = indicators.get("macd_last")
    sig = indicators.get("signal_last")
    sma50 = indicators.get("sma50_last")
    sma200 = indicators.get("sma200_last")
    dd60 = indicators.get("drawdown_60")

    values = [rsi, macd, sig, sma50, sma200, dd60]
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values):
        return None, "Nu exista suficiente date pentru a calcula semnalul tehnic pe BET."

    red = (
        dd60 <= -10.0 and
        rsi > 70 and
        macd < sig and
        sma50 < sma200
    )

    yellow = (
        -10.0 < dd60 <= -5.0 and
        60 <= rsi <= 70 and
        abs(macd - sig) < 0.1 and
        abs(sma50 - sma200) / max(abs(sma200), 1e-9) < 0.02
    )

    green = (
        dd60 > -5.0 and
        40 <= rsi < 70 and
        macd > sig and
        sma50 > sma200
    )

    if red:
        msg = "🔴 Corectie in derulare pe BET. Scadere de cel putin 10% fata de maximul pe 60 de zile, RSI peste 70, MACD sub semnal si SMA50 sub SMA200."
        return "red", msg

    if yellow:
        msg = "🟡 Posibil inceput de corectie pe BET. Scadere intre 5% si 10% fata de maximul pe 60 de zile si semnale tehnice mixte."
        return "yellow", msg

    if green:
        msg = "🟢 Trend pozitiv puternic pe BET. Scadere mica fata de maximul recent si indicatori tehnici favorabili."
        return "green", msg

    if dd60 is not None and sma50 is not None and sma200 is not None:
        if dd60 < -2.0 and sma50 <= sma200:
            msg = "🔵 Trend negativ moderat pe BET. Scaderi mici fata de maximul recent si trend tehnic usor descendent."
            return "neutral", msg
        if dd60 > 2.0 and sma50 >= sma200:
            msg = "🔵 Trend pozitiv moderat pe BET. Crestere usoara fata de maximul recent si trend tehnic ascendent."
            return "neutral", msg

    msg = "ℹ️ Niciun semnal tehnic puternic pe BET in acest moment. Piata este intr-o zona neutra, fara trend clar."
    return "neutral", msg

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_romania_gdp_latest():
    """Intoarce (PIB_RON, perioada_label) pentru Romania folosind Eurostat, PIB trimestrial nominal in moneda nationala.

    Sursa: Eurostat, dataset namq_10_gdp, indicator B1GQ, unit CP_MNAC (mil. moneda nationala, preturi curente).
    """
    try:
        import requests
        url = (
            "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
            "namq_10_gdp?geo=RO&na_item=B1GQ&unit=CP_MNAC&lastTimePeriod=1"
        )
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None, None
        data = r.json()
        vals = data.get("value", {})
        if not vals:
            return None, None
        # luam singura observatie intoarsa (ultimul trimestru disponibil)
        idx_str, val_mn = next(iter(vals.items()))
        try:
            idx_int = int(idx_str)
        except Exception:
            idx_int = None
        dim_time = data.get("dimension", {}).get("time", {}).get("category", {})
        labels = dim_time.get("label", {})
        index_map = dim_time.get("index", {})
        period_label = None
        if idx_int is not None and index_map:
            for lab, pos in index_map.items():
                if pos == idx_int:
                    period_label = lab
                    break
        if period_label is None and labels:
            # fallback, luam primul label
            period_label = list(labels.keys())[0]
        # valorile sunt in milioane moneda nationala
        try:
            gdp_ron = float(val_mn) * 1_000_000.0
        except Exception:
            return None, None
        return gdp_ron, period_label
    except Exception:
        return None, None


@st.cache_data(ttl=600, show_spinner=False)
def compute_buffett_indicator(symbols):
    """Calculeaza indicatorul Buffett pentru lista de simboluri BVB."""
    gdp_ron, gdp_period = fetch_romania_gdp_latest()
    if gdp_ron is None or gdp_ron <= 0:
        return None, None, None, None

    total_mcap = 0.0
    used = []
    skipped = []
    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            mcap = None
            fi = getattr(tk, "fast_info", None)
            if fi is not None:
                try:
                    if isinstance(fi, dict):
                        mcap = fi.get("market_cap")
                    else:
                        mcap = getattr(fi, "market_cap", None)
                except Exception:
                    mcap = None
            if not mcap:
                info = getattr(tk, "info", None) or {}
                mcap = info.get("marketCap")
            if mcap and mcap > 0:
                total_mcap += float(mcap)
                used.append(sym)
            else:
                skipped.append(sym)
        except Exception:
            skipped.append(sym)

    if total_mcap <= 0:
        return None, None, None, None

    buffett = total_mcap / float(gdp_ron) * 100.0
    meta = {"used": used, "skipped": skipped}
    return buffett, gdp_ron, gdp_period, meta

st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title("BVB Recommender")



with st.sidebar:
    st.header("Setari")
    history_days = st.number_input("Zile istoric", value=DEFAULT_SETTINGS["history_days"], step=10)
    momentum_lb = st.number_input("Lookback momentum", value=DEFAULT_SETTINGS["momentum_lookback"], step=5)

# lista completa de simboluri: BET + ETF-uri + AeRO standard
ALL_TICKERS = BET_CONSTITUENTS + ETF_TICKERS + AERO_TICKERS

# date principale
rows = fetch_all(ALL_TICKERS, int(history_days), int(momentum_lb))
for r in rows:
    r["score"] = score_row(r)

# fortam denumirile pentru companiile AeRO cunoscute
for r in rows:
    if r["symbol"] in AERO_CONSTITUENTS:
        r["name"] = AERO_CONSTITUENTS[r["symbol"]]

rows_sorted = sorted(rows, key=lambda x: (np.nan_to_num(x["score"], nan=-1e9)), reverse=True)

# impartim pe universuri
rows_bet = [r for r in rows_sorted if r["symbol"] in BET_CONSTITUENTS]
rows_aero = [r for r in rows_sorted if r["symbol"] in AERO_TICKERS]
rows_etf = [r for r in rows_sorted if r["symbol"] in ETF_TICKERS]

rec_bet = compute_recommendations([r for r in rows_bet if not r.get("no_data")])
rec_aero = compute_recommendations([r for r in rows_aero if not r.get("no_data")]) if rows_aero else {}
rec_etf = compute_recommendations([r for r in rows_etf if not r.get("no_data")]) if rows_etf else {}

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

tab_bet, tab_aero, tab_etf, tab_rezumat = st.tabs(["BET", "AeRO", "ETF-uri BVB", "Rezumat zi BVB"])

with tab_bet:
    st.subheader("Recomandari BET")
    if rows_bet:
        df_bet = pd.DataFrame([{
            "Nr": i+1,
            "Simbol": r["symbol"],
            "Denumire": r["name"],
            "Pret": round(r["price"],2) if r["price"] is not None else np.nan,
            "Delta zi %": round(r["day_change"],2) if not r.get("no_data") else np.nan,
            "Dividend net %": (round(r["last_dividend_net_pct"],2) if r.get("last_dividend_net_pct") is not None else np.nan),
            "Ex date": r.get("last_div_date") if r.get("last_div_date") else "",
            "Scor": round(r["score"],2) if r["score"]==r["score"] else np.nan,
            "Predictie 1 zi %": round(r.get("pred_next_change_pct", np.nan),2) if r.get("pred_next_change_pct") is not None else np.nan,
            "Recomandare": rec_bet.get(r["symbol"], "🔍 Insuficiente date pentru a face analiza") if not r.get("no_data") else "🔍 Insuficiente date pentru a face analiza",
            "Motiv": build_reason(r)
        } for i, r in enumerate(rows_bet)])
        st.dataframe(df_bet, use_container_width=True, hide_index=True)
    else:
        st.write("Nu exista date pentru companiile din BET.")

    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("Indice BET")
        choice = st.selectbox("Perioada", list(BET_PERIODS.keys()), index=5, key="bet_period")
        period, interval = BET_PERIODS.get(choice, ("1y","1d"))
        bet_yahoo, _ = bet_history(period, interval)
        simulare = compute_bet_simulare(rows_bet, choice) if rows_bet else None

        # logica de calibrare:
        # 1. daca avem date BET directe de la Yahoo, le folosim
        # 2. daca nu avem, folosim simularea pe componente si
        #    o calibram astfel incat ultimul punct sa fie aliniat
        #    cu ultimul BET disponibil pe o perioada mai lunga
        data = None
        if bet_yahoo is not None and not bet_yahoo.empty:
            data = bet_yahoo
        elif simulare is not None and not simulare.empty:
            ref_yahoo, _ = bet_history("6mo", "1d")
            if ref_yahoo is not None and not ref_yahoo.empty:
                try:
                    ref_last = float(ref_yahoo['BET_Close'].iloc[-1])
                    sim_last = float(simulare['BET_Close'].iloc[-1])
                    if sim_last > 0:
                        k = ref_last / sim_last
                        simulare = simulare.copy()
                        simulare['BET_Close'] = simulare['BET_Close'] * k
                except Exception:
                    pass
            data = simulare

        if data is not None and not data.empty:
            val_raw = float(data['BET_Close'].iloc[-1])
            prev_raw = float(data['BET_Close'].iloc[-2]) if len(data) >= 2 else val_raw

            # calibrare dinamica a scalei BET pe baza valorii curente de pe BVB
            bet_bvb = fetch_bet_last_from_bvb()
            scale = BET_SCALE
            if bet_bvb is not None and val_raw > 0:
                scale = bet_bvb / val_raw

            val = val_raw * scale
            prev = prev_raw * BET_SCALE
            var = val - prev
            varpct = (var / prev * 100.0) if prev else 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric("Valoare", f"{val:,.2f}".replace(","," ").replace(".",","))
            m2.metric("Var", f"{var:+.2f}".replace(".",","))
            m3.metric("Var%", f"{varpct:+.2f}%".replace(".",","))
            arima_raw = bet_arima_forecast(data)
            if arima_raw is not None:
                arima_val = arima_raw * scale
                arima_var = arima_val - val
                arima_varpct = (arima_var / val * 100.0) if val else 0.0
                st.metric("Predictie BET ARIMA 1 zi", f"{arima_val:,.2f}".replace(","," ").replace(".",","), f"{arima_varpct:+.2f}%".replace(".",","))
            buffett, gdp_ron, gdp_period, meta_buffett = compute_buffett_indicator(BET_CONSTITUENTS)
            if buffett is not None:
                if buffett < 70:
                    zona_text = "Piata pare ieftina. Evaluari atractive pentru acumulare."
                elif buffett <= 100:
                    zona_text = "Zona neutra. Evaluari echilibrate."
                else:
                    zona_text = "Piata pare scumpa. Risc mai mare de corectie."
                c1, c2 = st.columns([1, 2])
                label = "Buffett Romania (BET)"
                if gdp_period:
                    label = f"Buffett Romania (BET, PIB {gdp_period})"
                c1.metric(label, f"{buffett:.0f}%")
                c2.write(zona_text)
            else:
                st.info("Indicatorul Buffett nu poate fi calculat acum. Date PIB indisponibile.")
            if data is not None and not data.empty:
                df_bet = data.reset_index().rename(columns={data.index.name or 'index': 'Date'})
                scale_chart = BET_SCALE
                try:
                    # incercam sa calibram si pentru grafic
                    last_bet_bvb = fetch_bet_last_from_bvb()
                    if last_bet_bvb is not None and float(df_bet['BET_Close'].iloc[-1]) > 0:
                        scale_chart = last_bet_bvb / float(df_bet['BET_Close'].iloc[-1])
                except Exception:
                    pass
                df_bet['BET_Display'] = df_bet['BET_Close'] * scale_chart
                y_min = float(df_bet['BET_Display'].min()) * 0.97
                y_max = float(df_bet['BET_Display'].max()) * 1.03

                area = alt.Chart(df_bet).mark_area(
                    opacity=0.4
                ).encode(
                    x=alt.X('Date:T', axis=alt.Axis(format='%d/%m/%Y')),
                    y=alt.Y('BET_Display:Q', scale=alt.Scale(domain=[y_min, y_max])),
                )

                line = alt.Chart(df_bet).mark_line().encode(
                    x=alt.X('Date:T'),
                    y='BET_Display:Q',
                )

                chart = (area + line).interactive()
                st.altair_chart(chart, use_container_width=True)
            # Alerte BET pe baza indicatorilor tehnici
            try:
                bet_ind_df = data.copy()
                bet_ind_df = bet_ind_df.rename(columns={"BET_Close": "Close"})
                bet_ind = compute_indicators(bet_ind_df)
                alert_type, alert_msg = compute_bet_alert(bet_ind)
                if alert_type == "red":
                    st.error(alert_msg)
                elif alert_type == "yellow":
                    st.warning(alert_msg)
                elif alert_type == "green":
                    st.success(alert_msg)
                else:
                    st.info(alert_msg)
            except Exception:
                st.info("Nu se poate calcula alerta tehnica pentru BET in acest moment.")
        else:
            st.write("Nu exista suficiente date pentru a afisa graficul BET.")

    with col2:
        st.subheader("Detalii actiune BET")
        if rows_bet:
            symbols_bet = [r["symbol"] for r in rows_bet]
            sel = st.selectbox("Alege simbol", symbols_bet)
            perioada_act = st.selectbox("Perioada actiune", list(BET_PERIODS.keys()), index=5, key="bet_stock_period")
            impact_stiri = st.selectbox("Impact stiri", ["Neutru", "Pozitiv", "Negativ"], index=0, key="bet_news_impact")
            row = next(r for r in rows_bet if r["symbol"] == sel)
            h = row["history"].copy()
            if row.get("no_data"):
                st.write("nu ai date de a genera raportul")
            else:
                if "Date" not in h.columns:
                    h = h.rename(columns={h.columns[0]: "Date"})
                h["Close"] = h["Close"].astype(float)

                period_rows_map = {
                    "1 zi": 1,
                    "5 zile": 5,
                    "1 luna": 22,
                    "3 luni": 66,
                    "6 luni": 130,
                    "1 an": 260,
                    "5 ani": len(h)
                }
                n_rows = period_rows_map.get(perioada_act, len(h))
                if len(h) > n_rows:
                    h_plot = h.tail(n_rows).copy()
                else:
                    h_plot = h.copy()

                # baza de predictie din model
                base_pred_pct = row.get("pred_next_change_pct")
                # ajustare din stiri
                news_adj = 0.0
                if impact_stiri == "Pozitiv":
                    news_adj = 0.5
                elif impact_stiri == "Negativ":
                    news_adj = -0.5

                if base_pred_pct is not None:
                    adj_pred_pct = base_pred_pct + news_adj
                else:
                    adj_pred_pct = None

                st.metric("Pret curent RON", value=f"{row['price']:.2f}" if row['price'] is not None else "-")
                st.metric("Delta zi %", value=f"{row['day_change']:+.2f}%")
                st.metric("Momentum 30z %", value=f"{row['momentum']:+.2f}%")
                st.metric("Volatilitate %", value=f"{row['volatility']:.1f}%")
                st.metric("Volum mediu 30z", value=int(row['avg_volume']))
                st.metric("Ultimul dividend net %", value=(f"{row.get('last_dividend_net_pct'):.2f}%" if row.get('last_dividend_net_pct') is not None else "-"))
                st.metric("Ex date", value=(row.get('last_div_date') or "-"))
                st.metric("Predictie 1 zi %", value=(f"{adj_pred_pct:+.2f}%" if adj_pred_pct is not None else "-"))

                import pandas as _pd
                import altair as _alt

                df_real = _pd.DataFrame({
                    "Date": _pd.to_datetime(h_plot["Date"]),
                    "Price": h_plot["Close"].astype(float),
                    "Tip": "Istoric"
                })

                # calculam punctul de predictie ajustat
                last_close = float(h_plot["Close"].iloc[-1])
                if adj_pred_pct is not None:
                    pred_price = last_close * (1.0 + adj_pred_pct / 100.0)
                else:
                    pred_price = None

                if pred_price is not None and len(h_plot) > 0:
                    last_date = _pd.to_datetime(h_plot["Date"].iloc[-1])
                    next_date = last_date + _pd.Timedelta(days=1)
                    df_pred = _pd.DataFrame({
                        "Date": [last_date, next_date],
                        "Price": [last_close, float(pred_price)],
                        "Tip": "Predictie"
                    })
                    df_plot = _pd.concat([df_real, df_pred], ignore_index=True)
                else:
                    df_plot = df_real

                chart = _alt.Chart(df_plot).mark_line().encode(
                    x=_alt.X('Date:T', axis=_alt.Axis(format='%d/%m/%Y')),
                    y=_alt.Y('Price:Q'),
                    color=_alt.Color('Tip:N'),
                    strokeDash=_alt.condition(
                        _alt.datum.Tip == "Predictie",
                        _alt.value([4, 4]),
                        _alt.value([0, 0])
                    )
                ).interactive()

                st.altair_chart(chart, use_container_width=True)
        else:
            st.write("Nu exista companii BET in lista curenta.")


with tab_aero:
    st.subheader("Recomandari AeRO (BETAeRO)")
    if rows_aero:
        df_aero = pd.DataFrame([{
            "Nr": i+1,
            "Simbol": r["symbol"],
            "Denumire": r["name"],
            "Pret": round(r["price"],2) if r["price"] is not None else np.nan,
            "Delta zi %": round(r["day_change"],2) if not r.get("no_data") else np.nan,
            "Dividend net %": (round(r["last_dividend_net_pct"],2) if r.get("last_dividend_net_pct") is not None else np.nan),
            "Ex date": r.get("last_div_date") if r.get("last_div_date") else "",
            "Scor": round(r["score"],2) if r["score"]==r["score"] else np.nan,
            "Predictie 1 zi %": round(r.get("pred_next_change_pct", np.nan),2) if r.get("pred_next_change_pct") is not None else np.nan,
            "Recomandare": rec_aero.get(r["symbol"], "🔍 Insuficiente date pentru a face analiza") if not r.get("no_data") else "🔍 Insuficiente date pentru a face analiza",
            "Motiv": build_reason(r)
        } for i, r in enumerate(rows_aero)])
        st.dataframe(df_aero, use_container_width=True, hide_index=True)
    else:
        st.write("Nu exista companii AeRO de afisat in configuratia curenta.")

with tab_etf:
    st.subheader("Recomandari ETF-uri BVB")
    if rows_etf:
        df_etf = pd.DataFrame([{
            "Nr": i+1,
            "Simbol": r["symbol"],
            "Denumire": r["name"],
            "Pret": round(r["price"],2) if r["price"] is not None else np.nan,
            "Delta zi %": round(r["day_change"],2) if not r.get("no_data") else np.nan,
            "Dividend net %": (round(r["last_dividend_net_pct"],2) if r.get("last_dividend_net_pct") is not None else np.nan),
            "Ex date": r.get("last_div_date") if r.get("last_div_date") else "",
            "Scor": round(r["score"],2) if r["score"]==r["score"] else np.nan,
            "Predictie 1 zi %": round(r.get("pred_next_change_pct", np.nan),2) if r.get("pred_next_change_pct") is not None else np.nan,
            "Recomandare": rec_etf.get(r["symbol"], "🔍 Insuficiente date pentru a face analiza") if not r.get("no_data") else "🔍 Insuficiente date pentru a face analiza",
            "Motiv": build_reason(r)
        } for i, r in enumerate(rows_etf)])
        st.dataframe(df_etf, use_container_width=True, hide_index=True)
    else:
        st.write("Nu exista ETF-uri BVB de afisat in configuratia curenta.")
with tab_rezumat:
    st.subheader("Rezumat zi BVB")
    from datetime import datetime as _dt
    now = _dt.now()
    is_trading_day = now.weekday() < 5
    st.write("Program orientativ BVB: luni - vineri, aproximativ 10:00 - 18:00.")
    if not is_trading_day:
        st.write("Astazi nu este zi obisnuita de tranzactionare. Rezumatul este calculat pe ultimele date disponibile.")

    # Rezumat BET
    bet_data, _ = bet_history("1mo", "1d")
    if bet_data is not None and not bet_data.empty:
        bet_last = float(bet_data["BET_Close"].iloc[-1])
        bet_prev = float(bet_data["BET_Close"].iloc[-2]) if len(bet_data) >= 2 else bet_last
        bet_var = bet_last - bet_prev
        bet_varpct = (bet_var / bet_prev * 100.0) if bet_prev else 0.0
        st.markdown("Indice BET")
        st.write(f"Ultima variatie: {bet_var:+.2f} puncte, {bet_varpct:+.2f}%")
    else:
        st.write("Nu exista suficiente date pentru rezumatul BET.")

    # Rezumat portofoliu utilizator
    st.markdown("Rezumat portofoliu (simbolurile din USER_PORTFOLIO)")
    port_rows = [r for r in rows_bet + rows_aero + rows_etf if r["symbol"] in USER_PORTFOLIO]
    if port_rows:
        df_port = pd.DataFrame([{
            "Simbol": r["symbol"],
            "Denumire": r["name"],
            "Pret": round(r["price"], 2) if r["price"] is not None else None,
            "Delta zi %": round(r["day_change"], 2) if not r.get("no_data") else None,
            "Predictie 1 zi %": round(r.get("pred_next_change_pct", float("nan")), 2) if r.get("pred_next_change_pct") is not None else None,
            "Recomandare": (rec_bet.get(r["symbol"]) or rec_aero.get(r["symbol"]) or rec_etf.get(r["symbol"]) or "N/A")
        } for r in port_rows])
        st.dataframe(df_port, use_container_width=True, hide_index=True)
    else:
        st.write("Nu exista simboluri din portofoliul utilizatorului in universul curent.")

        st.write("Nu exista ETF-uri BVB de afisat in configuratia curenta.")
