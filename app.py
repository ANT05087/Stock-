"""
Swing Backtester - Streamlit app (works well in a phone browser).

Strategy (long only):
  BUY  when the close crosses above its N-day average AND volume is well
       above its recent average.
  SELL when the close drops back below the average, or a stop-loss is hit.

Trades happen at the NEXT day's open, with slippage charged on every fill.
This is a learning tool, not financial advice.
"""

import datetime as dt

import pandas as pd
import streamlit as st


# ----------------------------- BACKTEST CORE ----------------------------
def run_backtest(df, sma_days, vol_days, vol_mult, stop_pct, slippage, cash0):
    df = df.copy()
    df["sma"] = df["Close"].rolling(sma_days).mean()
    df["vol_avg"] = df["Volume"].rolling(vol_days).mean()
    df = df.dropna()

    cash, shares = float(cash0), 0.0
    entry_price, entry_date, stop_price = 0.0, None, 0.0
    pending_entry = pending_exit = False
    trades, equity = [], []
    prev_close = prev_sma = None

    def sell(date, price, reason):
        nonlocal cash, shares
        fill = price * (1 - slippage)
        cash += shares * fill
        trades.append({
            "Buy date": entry_date.date(), "Sell date": date.date(),
            "Buy price": round(entry_price, 2), "Sell price": round(fill, 2),
            "Return %": round((fill / entry_price - 1) * 100, 2),
            "Reason": reason,
        })
        shares = 0.0

    for date, row in df.iterrows():
        if pending_exit and shares > 0:
            sell(date, row["Open"], "below average")
        if pending_entry and shares == 0:
            entry_price = row["Open"] * (1 + slippage)
            shares = cash / entry_price
            cash = 0.0
            entry_date = date
            stop_price = entry_price * (1 - stop_pct)
        pending_entry = pending_exit = False

        if shares > 0 and row["Low"] <= stop_price:
            sell(date, min(row["Open"], stop_price), "stop-loss")

        if prev_close is not None:
            if shares == 0:
                crossed_up = prev_close <= prev_sma and row["Close"] > row["sma"]
                pending_entry = bool(crossed_up and row["Volume"] > vol_mult * row["vol_avg"])
            elif row["Close"] < row["sma"]:
                pending_exit = True
        prev_close, prev_sma = row["Close"], row["sma"]
        equity.append(cash + shares * row["Close"])

    if shares > 0:
        sell(df.index[-1], df["Close"].iloc[-1], "end of test")
        equity[-1] = cash

    equity = pd.Series(equity, index=df.index)
    buy_hold = cash0 * df["Close"] / df["Close"].iloc[0]
    return equity, buy_hold, pd.DataFrame(trades)


def max_drawdown(series):
    return (series / series.cummax() - 1).min()


def clean_prices(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda c: str(c).strip().title())
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    df = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(ticker, start, end):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    return clean_prices(df)


# --------------------------------- UI -----------------------------------
st.set_page_config(page_title="Swing Backtester", page_icon="📈")
st.title("📈 Swing Backtester")
st.caption("Buy on a breakout above the trend line with strong volume. "
           "Sell below the line or on a stop-loss.")

ticker = st.text_input("Ticker", value="SPOT").strip().upper()
c1, c2 = st.columns(2)
start = c1.date_input("Start", value=dt.date(2019, 1, 1))
end = c2.date_input("End", value=dt.date.today())

with st.expander("Strategy settings"):
    sma_days = st.number_input("Trend average (days)", 5, 400, 50)
    vol_days = st.number_input("Volume average (days)", 2, 200, 20)
    vol_mult = st.number_input("Volume multiple", 0.5, 5.0, 1.2, step=0.1)
    stop_pct = st.number_input("Stop-loss (%)", 1.0, 50.0, 8.0, step=0.5)
    slippage = st.number_input("Slippage per trade (%)", 0.0, 2.0, 0.05, step=0.01)
    cash0 = st.number_input("Starting cash ($)", 100, 10_000_000, 10_000, step=1000)

uploaded = st.file_uploader("Optional: upload your own daily CSV instead",
                            type=["csv"],
                            help="Needs Date, Open, High, Low, Close, Volume columns.")

if st.button("Run backtest", type="primary", use_container_width=True):
    try:
        if uploaded is not None:
            raw = pd.read_csv(uploaded, index_col=0, parse_dates=True)
            prices = clean_prices(raw)
            name = uploaded.name
        else:
            if start >= end:
                st.error("Start date must be before the end date.")
                st.stop()
            with st.spinner(f"Downloading {ticker} from Yahoo Finance..."):
                prices = download_prices(ticker, str(start), str(end))
            name = ticker
        if len(prices) < max(sma_days, vol_days) + 30:
            st.error("Not enough data for those settings. Try an earlier start date.")
            st.stop()
    except Exception as e:
        st.error(
            "Couldn't get price data. Yahoo sometimes blocks or rate-limits "
            "requests from cloud servers. Check the ticker, try again in a "
            f"minute, or upload a CSV above.\n\nDetails: {e}"
        )
        st.stop()

    equity, buy_hold, trades = run_backtest(
        prices, int(sma_days), int(vol_days), vol_mult,
        stop_pct / 100, slippage / 100, cash0)

    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / cash0 - 1
    bh_total = buy_hold.iloc[-1] / cash0 - 1
    cagr = (equity.iloc[-1] / cash0) ** (1 / years) - 1 if years > 0 else 0
    win_rate = (trades["Return %"] > 0).mean() * 100 if len(trades) else 0

    st.subheader(f"Results for {name}")
    m1, m2 = st.columns(2)
    m1.metric("Strategy total", f"{total:.1%}")
    m2.metric("Buy & hold total", f"{bh_total:.1%}")
    m3, m4 = st.columns(2)
    m3.metric("Strategy per year", f"{cagr:.1%}")
    m4.metric("Worst drop (strategy)", f"{max_drawdown(equity):.1%}")
    m5, m6 = st.columns(2)
    m5.metric("Trades", len(trades))
    m6.metric("Win rate", f"{win_rate:.0f}%")
    st.caption(f"Buy & hold worst drop: {max_drawdown(buy_hold):.1%}")

    st.subheader("Growth of your money")
    st.line_chart(pd.DataFrame({"Strategy": equity, "Buy & hold": buy_hold}))

    st.subheader("Trades")
    if len(trades):
        st.dataframe(trades, use_container_width=True, hide_index=True)
        st.download_button("Download trades (CSV)", trades.to_csv(index=False),
                           file_name=f"{name}_trades.csv", mime="text/csv")
    else:
        st.write("No trades were triggered with these settings.")

    st.info("Past results don't predict future returns. Changing settings "
            "until the result looks good is how people fool themselves.")
