
import os
import re
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

APP_TITLE = "BVB Recommender Web v2 (BVB.ro first)"

TICKERS = [
    "TLV","H2O","SNP","SNG","BRD",
    "TGN","EL","BVB","DIGI","FP",
    "SNN","TEL","ONE","COTE","WINE",
    "ALR","PE","ATB","SMTL","BIO"
]

BET_PROFILE_URL = "https://www.bvb.ro/TradingAndStatistics/Indices/IndicesProfiles.aspx?i=BET"

HEADERS = {"User-Agent": "Mozilla/5.0"}

def _clean(txt):
    return re.sub(r"\s+"," ",txt or "").strip()

def fetch_bvb_overview(symbol):
    url = f"https://www.bvb.ro/FinancialInstruments/Details/FinancialInstrumentsDetails.aspx?s={symbol}"
    out = {"symbol": symbol, "source": url}
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, "html.parser")

        price = None
        price_el = soup.find(id=re.compile("lbLast", re.I))
        if price_el:
            price_txt = price_el.get_text(strip=True).replace(",", ".")
            try:
                price = float(re.sub(r"[^0-9.\-]","", price_txt))
            except Exception:
                price = None
        out["price"] = price

        pe = None
        div_yield = None
        volume = None

        tables = soup.find_all("table")
        for t in tables:
            for row in t.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in row.find_all(["th","td"])]
                if len(cells) == 2:
                    k = _clean(cells[0]).lower()
                    v = _clean(cells[1])
                    if "p/e" in k or "price/earnings" in k:
                        try: pe = float(v.replace(",", ".").split()[0])
                        except: pass
                    if "dividend" in k and ("yield" in k or "%" in v):
                        try: div_yield = float(v.replace(",", ".").replace("%","").split()[0])
                        except: pass
                    if ("volum" in k or "volume" in k) and any(ch.isdigit() for ch in v):
                        try: volume = int(re.sub(r"[^0-9]", "", v))
                        except: pass

        out["pe"] = pe
        out["dividend_yield"] = div_yield
        out["volume"] = volume

        last_update = None
        lu_el = soup.find(string=re.compile(r"Last update|Ultima actualizare", re.I))
        if lu_el:
            lu_txt = _clean(lu_el if isinstance(lu_el, str) else lu_el.get_text(" ", strip=True))
            m = re.search(r"(\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2}).*?(\d{2}:\d{2})?", lu_txt)
            if m: last_update = m.group(0)
        if last_update is None:
            last_update = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        out["last_update"] = last_update

        return out
    except Exception as e:
        out["error"] = str(e)
        return out

def fetch_bvb_dividends(symbol):
    url = f"https://www.bvb.ro/FinancialInstruments/Details/FinancialInstrumentsDetails.aspx?s={symbol}"
    out = {
        "last_dividend_value": None,
        "last_ex_date": None,
        "last_payment_date": None,
        "next_ex_date": None,
        "next_payment_date": None,
        "source_div": url
    }
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        text = soup.get_text(" ", strip=True)

        m = re.search(r"Dividend(?:\s+brut)?[^0-9]*([0-9]+[.,][0-9]+)\s*RON", text, re.I)
        if m: out["last_dividend_value"] = float(m.group(1).replace(",", "."))

        m = re.search(r"Ex[-\s]?date[:\s]+(\d{2}\.\d{2}\.\d{4})", text, re.I)
        if m: out["last_ex_date"] = m.group(1)

        m = re.search(r"Payment\s*date[:\s]+(\d{2}\.\d{2}\.\d{4})", text, re.I)
        if m: out["last_payment_date"] = m.group(1)

        m = re.search(r"Next\s*Ex[-\s]?date[:\s]+(\d{2}\.\d{2}\.\d{4})", text, re.I)
        if m: out["next_ex_date"] = m.group(1)

        m = re.search(r"Next\s*Payment\s*date[:\s]+(\d{2}\.\d{2}\.\d{4})", text, re.I)
        if m: out["next_payment_date"] = m.group(1)

        return out
    except Exception:
        return out

def fetch_bet_snapshot():
    try:
        r = requests.get(BET_PROFILE_URL, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        idx_val = None
        m = re.search(r"\bBET\b.*?(\d{4,6}[.,]\d{2})", text)
        if m: idx_val = float(m.group(1).replace(",", "."))
        last_update = None
        m2 = re.search(r"(Last update|Ultima actualizare)\s*[:\-]?\s*([0-9.: ]{8,20})", text, re.I)
        if m2: last_update = m2.group(2)
        return {"bet_value": idx_val, "last_update": last_update, "source": BET_PROFILE_URL}
    except Exception as e:
        return {"error": str(e), "source": BET_PROFILE_URL}

def fallback_yahoo(symbol):
    if yf is None: return {}
    try:
        tk = yf.Ticker(symbol + ".RO")
        info = {}
        try: info = tk.info
        except: info = {}
        hist = tk.history(period="200d", interval="1d")
        price = info.get("regularMarketPrice") or (hist["Close"].iloc[-1] if len(hist) else None)
        pe = info.get("trailingPE")
        dy = info.get("dividendYield")
        if dy is not None: dy = float(dy) * 100.0
        vol = float(hist["Volume"].tail(30).mean()) if "Volume" in hist.columns and len(hist) else None
        last_update = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return {"price": price, "pe": pe, "dividend_yield": dy, "volume": vol, "last_update": last_update, "source":"Yahoo"}
    except Exception:
        return {}

def compute_momentum(symbol):
    if yf is None: return None
    try:
        tk = yf.Ticker(symbol + ".RO")
        h = tk.history(period="60d", interval="1d")
        if len(h) < 31: return None
        p_now = float(h["Close"].iloc[-1])
        p_30 = float(h["Close"].iloc[-31])
        return (p_now / p_30 - 1.0) * 100.0
    except Exception:
        return None

def verdict_from_metrics(momentum, pe, div_yield, volume):
    good = 0; bad = 0; reasons = []
    if momentum is not None:
        if momentum > 0: reasons.append(f"momentum +{momentum:.1f}%"); good += 1
        else: reasons.append(f"momentum {momentum:.1f}%"); bad += 1
    if pe is not None:
        if pe < 15: reasons.append(f"PE {pe:.1f} ok"); good += 1
        elif pe > 25: reasons.append(f"PE {pe:.1f} mare"); bad += 1
    if div_yield is not None and div_yield >= 3.0:
        reasons.append(f"div {div_yield:.1f}%"); good += 1
    if volume is not None and volume >= 100000:
        reasons.append("lichiditate ok"); good += 1
    v = "De cumparat" if good >= bad else "De evitat"
    return v, "; ".join(reasons) if reasons else "-"

def score_row(momentum, pe, div_yield, volume):
    s = 0.0
    if momentum is not None: s += 0.4 * momentum
    if pe is not None: s += 0.2 * max(0.0, 25 - min(pe, 40))
    if div_yield is not None: s += 0.25 * div_yield
    if volume is not None: s += 0.15 * min(volume/200000.0, 1.0) * 10.0
    return float(s)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("BVB Recommender")

with st.sidebar:
    st.caption("Sursa primara: BVB.ro. Fallback: Yahoo Finance.")
    st.caption("Top 20 pe baza momentum, PE, dividend, volum.")
    show_debug = st.toggle("Afiseaza surse/diagnoza", value=False)

rows = []
for sym in TICKERS:
    b = fetch_bvb_overview(sym)
    d = fetch_bvb_dividends(sym)
    if (b.get("price") is None or b.get("pe") is None or b.get("dividend_yield") is None or b.get("volume") is None) and yf is not None:
        fy = fallback_yahoo(sym)
        for k in ["price","pe","dividend_yield","volume","last_update","source"]:
            if b.get(k) is None and fy.get(k) is not None:
                b[k] = fy[k]
    mom = compute_momentum(sym)
    vrd, why = verdict_from_metrics(mom, b.get("pe"), b.get("dividend_yield"), b.get("volume"))
    score = score_row(mom, b.get("pe"), b.get("dividend_yield"), b.get("volume"))
    rows.append({
        "Simbol": sym,
        "Pret": round(b.get("price"), 4) if b.get("price") is not None else None,
        "Dividend ultima plata": d.get("last_dividend_value"),
        "Ex-date anterior": d.get("last_ex_date"),
        "Urmator ex-date": d.get("next_ex_date"),
        "Data plata": d.get("last_payment_date"),
        "PE": round(b.get("pe"), 2) if b.get("pe") is not None else None,
        "Randament div %": round(b.get("dividend_yield"), 2) if b.get("dividend_yield") is not None else None,
        "Volum mediu": int(b.get("volume")) if b.get("volume") is not None else None,
        "Momentum 30z %": round(mom, 2) if mom is not None else None,
        "Ultima actualizare": b.get("last_update"),
        "Verdict": vrd,
        "Motiv": why,
        "_score": score,
        "_src": b.get("source"),
        "_divsrc": d.get("source_div")
    })

df = pd.DataFrame(rows).sort_values(by="_score", ascending=False).reset_index(drop=True)
df.index = df.index + 1
show_cols = ["Simbol","Pret","Dividend ultima plata","Ex-date anterior","Urmator ex-date","Data plata",
             "PE","Randament div %","Volum mediu","Momentum 30z %","Ultima actualizare","Verdict","Motiv"]
st.subheader("Top 20 actiuni BVB")
st.dataframe(df[show_cols], use_container_width=True)

st.divider()
st.subheader("Indice BET")
bet_info = fetch_bet_snapshot()
c1, c2 = st.columns(2)
with c1:
    st.metric("BET", f"{bet_info.get('bet_value') or '-'}")
with c2:
    st.write(f"Ultima actualizare: {bet_info.get('last_update') or '-'}")

if st.checkbox("Arata surse per simbol") or show_debug:
    st.write(df[["Simbol","_src","_divsrc","_score"]])
