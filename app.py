import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Stock Forecast App", layout="wide")

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

WINDOW_SIZE = 60

# -----------------------------
# FUNCTIONS
# -----------------------------
def load_stock_data(ticker, start="2018-01-01"):
    df = yf.download(ticker, start=start, progress=False)
    df = df[['Close']]
    df = df.dropna()
    df = df[df['Close'] > 0]
    return df

def rolling_forecast(model, last_window, scaler, horizon=7):
    forecasts = []
    current_window = last_window.copy()

    for _ in range(horizon):
        X_input = current_window.reshape(1, WINDOW_SIZE, 1)
        next_pred = model.predict(X_input, verbose=0)[0][0]
        forecasts.append(next_pred)
        current_window = np.vstack([current_window[1:], [[next_pred]]])

    forecasts = scaler.inverse_transform(
        np.array(forecasts).reshape(-1, 1)
    )
    return forecasts

# -----------------------------
# UI
# -----------------------------
st.title("📈 Stock Forecast & Decision Support System")

ticker = st.selectbox(
    "Select Stock",
    ["BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK"]
)

horizon = st.slider("Forecast Horizon (days)", 7, 30, 7)

# -----------------------------
# MAIN LOGIC
# -----------------------------
df = load_stock_data(ticker)

scaled_data = scaler.transform(df[['Close']].values)
last_window = scaled_data[-WINDOW_SIZE:]

forecast = rolling_forecast(
    model=model,
    last_window=last_window,
    scaler=scaler,
    horizon=horizon
)

last_price = df['Close'].iloc[-1]
expected_return = (forecast[-1][0] - last_price) / last_price

# -----------------------------
# SIGNAL
# -----------------------------
if expected_return > 0.02:
    signal = "BUY"
elif expected_return < -0.02:
    signal = "SELL"
else:
    signal = "HOLD"

# -----------------------------
# DISPLAY
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Last Price", f"Rp {last_price:,.0f}")
col2.metric("Expected Return", f"{expected_return*100:.2f}%")
col3.metric("Signal", signal)

# -----------------------------
# PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df['Close'].iloc[-60:], label="Historical")
future_dates = pd.date_range(
    start=df.index[-1],
    periods=len(forecast)+1,
    freq='B'
)[1:]
ax.plot(future_dates, forecast, linestyle='--', marker='o', label="Forecast")
ax.legend()
st.pyplot(fig)

st.caption("⚠️ This app is for educational purposes only, not financial advice.")
