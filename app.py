import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

st.title("📊 Smart Stock Prediction System")

stock = st.text_input("Enter Stock Symbol", "BOSCHLTD.NS")

if st.button("Analyze"):

    try:
        data = yf.download(stock, period="1y")

        if data.empty:
            st.error("Invalid stock symbol")
        else:
            # FIX HERE 👇
            close_prices = data['Close'].squeeze().dropna().tolist()

            y = np.array(close_prices, dtype=float)
            X = np.arange(len(y)).reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, y)

            predictions = model.predict(X)
            mae = mean_absolute_error(y, predictions)

            next_day = np.array([[len(y)]])
            predicted_price = model.predict(next_day)[0]

            last_price = y[-1]

            # Moving averages
            ma50 = np.mean(y[-50:]) if len(y) >= 50 else np.mean(y)
            ma100 = np.mean(y[-100:]) if len(y) >= 100 else np.mean(y)

            if ma50 > ma100:
                trend = "Bullish 📈"
                signal = "BUY ✅"
            else:
                trend = "Bearish 📉"
                signal = "SELL ❌"

            st.subheader("📌 Results")
            st.write("Last Price:", round(last_price, 2))
            st.write("Predicted Price:", round(predicted_price, 2))
            st.write("Trend:", trend)
            st.write("Signal:", signal)
            st.write("Model Error (MAE):", round(mae, 2))

            fig, ax = plt.subplots()
            ax.plot(y, label="Close Price")
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error occurred: {e}")