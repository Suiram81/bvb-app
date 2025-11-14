
import time
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, date

# ... [restul codului aplicației rămâne neschimbat] ...

# Funcție nouă pentru alertă de corecție pe BET
def check_correction_alert(data, threshold_pct=2.5):
    alert_level = ""
    message = ""
    if data is not None and len(data) > 2:
        last = float(data.iloc[-1])
        prev = float(data.iloc[-2])
        pct_drop = ((prev - last) / prev) * 100
        if pct_drop >= threshold_pct:
            alert_level = "high"
            message = f"⚠️ ALERTĂ: Posibilă corecție detectată pe BET (-{pct_drop:.2f}%)"
        elif pct_drop >= threshold_pct / 2:
            alert_level = "medium"
            message = f"ℹ️ Semnal de atenție: BET a scăzut cu -{pct_drop:.2f}%"
        else:
            alert_level = "none"
    return alert_level, message

# Apel și afișare alertă în aplicație (de introdus înainte de st.dataframe(df))
data_for_alert = bet_yahoo if bet_yahoo is not None else simulare
alert_level, alert_msg = check_correction_alert(data_for_alert["BET_Close"]) if data_for_alert is not None else ("", "")
if alert_msg:
    st.subheader("Alerte corecție tehnică")
    if alert_level == "high":
        st.error(alert_msg)
    elif alert_level == "medium":
        st.warning(alert_msg)

# ... [restul codului aplicației rămâne neschimbat] ...
