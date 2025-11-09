
import re
from datetime import datetime, timezone
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st

import yfinance as yf  # folosim Yahoo ca baza stabila, BVB doar pentru dividende daca raspunde

APP_TITLE = "BVB Recommender Web v2 SAFE"

TICKERS = ["TLV","H2O","SNP","SNG","BRD","TGN","EL","BVB","DIGI","FP",
           "SNN","TEL","ONE","COTE","WINE","ALR","PE","ATB","SMTL","BIO"]

HEADERS = {"User-Agent": "Mozilla/5.0"}
BVB_TIMEOUT = 4  # timp mic ca sa nu blocheze UI

def fetch_dividends_bvb(symbol):
    url = f"https://www.bvb.ro/FinancialInstruments/Details/FinancialInstrumentsDetails.aspx?s={symbol}"
    out = {"last_dividend_value": None,"last_ex_date": None,"last_payment_date": None,"next_ex_date": None,"next_payment_date": None,"_divsrc": url}
    try:
        r = requests.get(url, headers=HEADERS, timeout=BVB_TIMEOUT)
        r.raise_for_status()
        text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
        m = re.search(r"Dividend(?:\\s+brut)?[^0-9]*([0-9]+[.,][0-9]+)\\s*RON", text, re.I)
        if m: out["last_dividend_value"] = float(m.group(1).replace(",", "."))
        m = re.search(r"Ex[-\\s]?date[:\\s]+(\\d{2}\\.\\d{2}\\.\\d{4})", text, re.I)
        if m: out["last_ex_date"] = m.group(1)
        m = re.search(r"Payment\\s*date[:\\s]+(\\d{2}\\.\\d{2}\\.\\d{4})", text, re.I)
        if m: out["last_payment_date"] = m.group(1)
        m = re.search(r"Next\\s*Ex[-\\s]?date[:\\s]+(\\d{2}\\.\\d{2}\\.\\d{4})", text, re.I)
        if m: out["next_ex_date"] = m.group(1)
        m = re.search(r"Next\\s*Payment\\s*date[:\\s]+(\\d{2}\\.\\d{2}\\.\\d{4})", text, re.I)
        if m: out["next_payment_date"] = m.group(1)
    except Exception:
        pass
    return out

def metrics_yahoo(symbol):
    tk = yf.Ticker(symbol + ".RO")
    info = {}
    try: info = tk.info
    except Exception: info = {}
    hist = tk.history(period="60d", interval="1d")
    price = info.get("regularMarketPrice") or (hist["Close"].iloc[-1] if len(hist) else None)
    pe = info.get("trailingPE")
    dy = info.get("dividendYield")
    if dy is not None:
        dy = float(dy) * 100.0
    vol = int(hist["Volume"].tail(30).mean()) if "Volume" in hist.columns and len(hist) else None
    mom = None
    if len(hist) >= 31:
        p_now = float(hist["Close"].iloc[-1]); p_30 = float(hist["Close"].iloc[-31])
        mom = (p_now / p_30 - 1.0) * 100.0
    last_update = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return {"price": price, "pe": pe, "dividend_yield": dy, "volume": vol, "momentum": mom, "last_update": last_update}

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

st.caption("Mod sigur: pret, PE, volum, momentum din Yahoo. Dividende de pe BVB.ro cand raspunde.")

rows = []
bvb_ok = True
for sym in TICKERS:
    y = metrics_yahoo(sym)
    d = fetch_dividends_bvb(sym)
    if d["last_dividend_value"] is None and d["last_ex_date"] is None:
        bvb_ok = False
    verdict, why = verdict_from_metrics(y["momentum"], y["pe"], y["dividend_yield"], y["volume"])
    score = score_row(y["momentum"], y["pe"], y["dividend_yield"], y["volume"])
    rows.append({
        "Simbol": sym,
        "Pret": round(y["price"], 4) if y["price"] is not None else None,
        "Dividend ultima plata": d.get("last_dividend_value"),
        "Ex-date anterior": d.get("last_ex_date"),
        "Urmator ex-date": d.get("next_ex_date"),
        "Data plata": d.get("last_payment_date"),
        "PE": round(y["pe"], 2) if y["pe"] is not None else None,
        "Randament div %": round(y["dividend_yield"], 2) if y["dividend_yield"] is not None else None,
        "Volum mediu": y["volume"],
        "Momentum 30z %": round(y["momentum"], 2) if y["momentum"] is not None else None,
        "Ultima actualizare": y["last_update"],
        "Verdict": verdict,
        "Motiv": why,
        "_score": score
    })

df = pd.DataFrame(rows).sort_values(by="_score", ascending=False).reset_index(drop=True)
df.index = df.index + 1

st.subheader("Top 20 actiuni BVB")
st.dataframe(df[["Simbol","Pret","Dividend ultima plata","Ex-date anterior","Urmator ex-date","Data plata",
                 "PE","Randament div %","Volum mediu","Momentum 30z %","Ultima actualizare","Verdict","Motiv"]],
             use_container_width=True)

if not bvb_ok:
    st.info("Dividendele pot lipsi temporar daca BVB.ro nu raspunde. Tabelul ramane functional.")
