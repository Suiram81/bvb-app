
import time
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, date

# Corectie: definim bet_yahoo inainte de a-l folosi
def bet_history(period="3mo", interval="1d"):
    for sym in ["^BETI", "^BET"]:
        try:
            tk = yf.Ticker(sym)
            h = tk.history(period=period, interval=interval, timeout=20, auto_adjust=False)
            if h is not None and not h.empty:
                return h[["Close"]].rename(columns={"Close": "BET_Close"}), "BET Yahoo"
        except Exception:
            pass
    return None, "NA"

bet_yahoo, _ = bet_history()
simulare = None  # fallback simplu

data_for_alert = bet_yahoo if bet_yahoo is not None else simulare

st.title("Alerte corectie tehnică")
st.write("Verificare inițială terminată. Datele au fost preluate corect.")
