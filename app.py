import pandas as pd
from newsapi import NewsApiClient
from transformers import pipeline
import streamlit as st
import datetime
import yfinance as yf

# Set page config
st.set_page_config(page_title="JAY Market Insights", layout="wide")

# Title and description
st.title("📊 JAY Market Insights")
st.markdown("Real-time sentiment analysis of market news for trading decisions")

# Initialize sentiment model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_model = load_model()
price_data = {
    "XAUUSD": {
        datetime(2026, 7, 20): {"close": 4000, "low": 3995, "high": 4010},
        datetime(2026, 7, 21): {"close": 4008, "low": 4000, "high": 4015}
    },
    "USDJPY": {
        datetime(2026, 7, 20): {"close": 162.30, "low": 162.00, "high": 162.50},
        datetime(2026, 7, 21): {"close": 162.80, "low": 162.20, "high": 163.00}
    }
}

# Function to check if market is open
def market_open_now():
    now = datetime.datetime.utcnow()
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour
    # Forex closes Friday 22:00 UTC, opens Sunday 22:00 UTC
    return not (weekday == 5 and hour >= 22 or weekday == 6 and hour < 22)

# Function to map sentiment to trading pairs
def map_to_pairs(headline, result):
    label = result[0]['label'].upper()
    pairs_map = {
        "POSITIVE": ["EURUSD", "GBPUSD", "XAUUSD"],
        "NEGATIVE": ["USDJPY", "USDCHF", "DXY"]
    }
    return pairs_map.get(label, [])

# Example logic to add Buy/Sell column
def get_trade_signal(sentiment, pairs):
    if sentiment == "POSITIVE":
        return "🟢 BUY " + ", ".join(pairs)
    elif sentiment == "NEGATIVE":
        return "🔴 SELL " + ", ".join(pairs)
    else:
        return "⏸️ HOLD"

# Get trade levels (stop loss and take profit) with better rounding
def get_trade_levels(signal, current_price):
    if isinstance(signal, str) and signal.startswith("🟢"):  # BUY signal
        stop_loss = round(current_price * 0.99, 5)   # 1% below
        take_profit = round(current_price * 1.02, 5) # 2% above
    elif isinstance(signal, str) and signal.startswith("🔴"):  # SELL signal
        stop_loss = round(current_price * 1.01, 5)   # 1% above
        take_profit = round(current_price * 0.98, 5) # 2% below
    else:
        stop_loss, take_profit = None, None
    return stop_loss, take_profit

# Example prices for trading pairs
example_prices = {
    "EURUSD": 1.09500,
    "GBPUSD": 1.27500,
    "XAUUSD": 2050.00000,
    "USDJPY": 145.50000,
    "USDCHF": 0.88500,
    "DXY": 103.50000
}

def get_current_price(pair):
    """Get current price for a trading pair"""
    return example_prices.get(pair, 0)

# Pip calculation helper
def pip_difference(pair, entry, exit_price):
    """Calculate pip difference based on pair type"""
    try:
        if "JPY" in pair:
            return round((exit_price - entry) * 100, 2)   # 1 pip = 0.01
        elif "XAU" in pair or "GC" in pair:
            return round(exit_price - entry, 2)           # treat 1 unit = 1 pip
        else:
            return round((exit_price - entry) * 10000, 2) # 1 pip = 0.0001
    except:
        return 0

# Backtesting function with timeframe selector
@st.cache_data
def backtest_signals(timeframe="1d"):
    """Download historical data and backtest trading signals with pip calculation"""
    try:
        pairs = {
            "EURUSD": "EURUSD=X",
            "USDJPY": "USDJPY=X",
            "GBPUSD": "GBPUSD=X",
            "XAUUSD": "GC=F"   # Gold futures as proxy for XAUUSD
        }
        
        data = {}
        for pair, ticker in pairs.items():
            try:
                df = yf.download(ticker, start="2026-01-01", end="2026-12-31", interval=timeframe, progress=False)
                if not df.empty:
                    df.index = pd.to_datetime(df.index)
                    data[pair] = df
            except:
                pass
        
        # Example signals for backtesting
        signals = pd.DataFrame([
            {"date":"2026-01-05","signal":"BUY","pair":"EURUSD","stop_loss":1.05,"take_profit":1.08},
            {"date":"2026-01-10","signal":"SELL","pair":"EURUSD","stop_loss":1.09,"take_profit":1.06},
            {"date":"2026-03-15","signal":"BUY","pair":"USDJPY","stop_loss":132.5,"take_profit":135.0},
            {"date":"2026-04-05","signal":"SELL","pair":"GBPUSD","stop_loss":1.28,"take_profit":1.25},
            {"date":"2026-04-10","signal":"BUY","pair":"XAUUSD","stop_loss":1980,"take_profit":2020},
        ])
        signals['date'] = pd.to_datetime(signals['date'])
        
        # Evaluate trades
        lookahead = 5
        results = []
        days_to_result = []
        pip_results = []
        
        for _, row in signals.iterrows():
            pair_data = data.get(row['pair'])
            
            if pair_data is not None and not pair_data.empty:
                try:
                    # Check if date exists in the data
                    matching_dates = pair_data.index[pair_data.index.date == row['date'].date()]
                    
                    if len(matching_dates) > 0:
                        start_idx = pair_data.index.get_loc(matching_dates[0])
                        end_idx = min(start_idx + lookahead, len(pair_data)-1)
                        window = pair_data.iloc[start_idx:end_idx+1]
                        
                        outcome = "HOLD"
                        days_taken = None
                        pip_value = 0
                        
                        entry_price = pair_data.iloc[start_idx]['Open']
                        
                        for i, (idx, day) in enumerate(window.iterrows()):
                            high = day['High']
                            low = day['Low']
                            
                            if row['signal'] == "BUY":
                                if high >= row['take_profit']:
                                    outcome = "WIN"
                                    days_taken = i
                                    pip_value = pip_difference(row['pair'], entry_price, row['take_profit'])
                                    break
                                elif low <= row['stop_loss']:
                                    outcome = "LOSS"
                                    days_taken = i
                                    pip_value = pip_difference(row['pair'], entry_price, row['stop_loss'])
                                    break
                            elif row['signal'] == "SELL":
                                if low <= row['take_profit']:
                                    outcome = "WIN"
                                    days_taken = i
                                    pip_value = pip_difference(row['pair'], entry_price, row['take_profit'])
                                    break
                                elif high >= row['stop_loss']:
                                    outcome = "LOSS"
                                    days_taken = i
                                    pip_value = pip_difference(row['pair'], entry_price, row['stop_loss'])
                                    break
                        
                        results.append(outcome)
                        days_to_result.append(days_taken)
                        pip_results.append(pip_value)
                    else:
                        results.append("NO DATA")
                        days_to_result.append(None)
                        pip_results.append(0)
                except Exception as e:
                    results.append("NO DATA")
                    days_to_result.append(None)
                    pip_results.append(0)
            else:
                results.append("NO DATA")
                days_to_result.append(None)
                pip_results.append(0)
        
        signals['result'] = results
        signals['days_to_result'] = days_to_result
        signals['pips'] = pip_results
        
        return signals
    except Exception as e:
        st.error(f"Backtest Error: {e}")
        return pd.DataFrame()

# Display market status
market_status = market_open_now()
if market_status:
    st.success("🟢 **FOREX MARKET IS OPEN**")
else:
    st.warning("🔴 **FOREX MARKET IS CLOSED**")

# Connect with your API key

    newsapi = NewsApiClient(api_key="f8665eb595e943a7bbbe1e05ecf32730")

    # Pull latest financial headlines
    articles = newsapi.get_everything(
        q="Federal Reserve OR inflation OR gold OR ECB OR Bank of England OR geopolitics OR oil OR unemployment OR China OR recession OR jobs OR USD OR forex OR currency",
        language="en",
        sort_by="publishedAt",
        page_size=10
    )

    # Collect results
    results = []
    for a in articles['articles'][:10]:  # show top 10 headlines
        headline = a['title']
        result = sentiment_model(headline)
        suggestion = map_to_pairs(headline, result)
        results.append({
            "Headline": headline,
            "Sentiment": result[0]['label'],
            "Confidence": round(result[0]['score'], 2),
            "Suggested Pairs": ", ".join(suggestion) if suggestion else "N/A",
            "Source": a['source']['name']
        })

    # Create dataframe
    df = pd.DataFrame(results)
    
    # Add trading signal column
    df["Signal"] = df.apply(lambda row: get_trade_signal(row["Sentiment"], row["Suggested Pairs"].split(", ") if row["Suggested Pairs"] != "N/A" else []), axis=1)
    
    # Add current price based on first suggested pair
    def get_first_pair_price(pairs_str):
        if pairs_str == "N/A" or not pairs_str:
            return 0
        pairs = pairs_str.split(", ")
        if pairs:
            first_pair = pairs[0]
            return get_current_price(first_pair)
        return 0

df["Current Price"] = df["Suggested Pairs"].apply(get_first_pair_price)
    
    # Calculate stop loss and take profit with improved rounding
df["Stop Loss"], df["Take Profit"] = zip(*df.apply(
    lambda row: get_trade_levels(row),
    axis=1
))
  
    
    # Format numeric columns for display
    df["Current Price"] = df["Current Price"].apply(lambda x: f"{x:.5f}" if x else "N/A")
    df["Stop Loss"] = df["Stop Loss"].apply(lambda x: f"{x:.5f}" if x else "N/A")
    df["Take Profit"] = df["Take Profit"].apply(lambda x: f"{x:.5f}" if x else "N/A")
# Display as beautiful table
st.subheader("📰 Latest Market")
st.dataframe(df, use_container_width=True)

# Backtest button
if st.button("Run Backtest", key="run_backtest_main"):
    stats = run_backtest(st.session_state.signals, price_data)
    st.write("Backtest Statistics") 
    st.write(stats)

# Show statistics
st.subheader("📈 Sentiment Summary")
col1, col2, col3 = st.columns(3)

with col1:
    positive = len(df[df['Sentiment'] == 'positive'])
    st.metric("🟢 Buy Signals", positive)

with col2:
    negative = len(df[df['Sentiment'] == 'negative'])
    st.metric("🔴 Sell Signals", negative)

with col3:
    neutral = len(df[df['Sentiment'] == 'neutral'])
    st.metric("⚪ Neutral Signals", neutral)

    
    with col2:
        negative = len(df[df['Sentiment'] == 'NEGATIVE'])
        st.metric("🔴 Sell Signals", negative)
    
    with col3:
        avg_conf = df['Confidence'].mean()
        st.metric("Average Confidence", f"{avg_conf:.1%}")

    # Backtesting section
    st.subheader("📊 Strategy Backtest Results")
    
    # Timeframe selector
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Select Timeframe for Backtest:**")
    with col2:
        timeframe = st.selectbox("Timeframe", ["1d", "15m", "5m"], label_visibility="collapsed")
    
    if st.button("Run Backtest"):
        with st.spinner(f"Running backtest on {timeframe} data..."):
            backtest_df = backtest_signals(timeframe=timeframe)
            
            if not backtest_df.empty:
                st.dataframe(backtest_df, use_container_width=True)
                
                # Show backtest statistics
                st.subheader("📈 Backtest Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    win_count = int((backtest_df['result'] == "WIN").sum())
                    total_trades = int(len(backtest_df))
                    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                with col2:
                    loss_count = int((backtest_df['result'] == "LOSS").sum())
                    loss_rate = (loss_count / total_trades * 100) if total_trades > 0 else 0
                    st.metric("Loss Rate", f"{loss_rate:.1f}%")
                
                with col3:
                    hold_count = int((backtest_df['result'] == "HOLD").sum())
                    hold_rate = (hold_count / total_trades * 100) if total_trades > 0 else 0
                    st.metric("Hold Rate", f"{hold_rate:.1f}%")
                
                with col4:
                    st.metric("Total Trades", total_trades)
                
                # Pip statistics
                st.subheader("💰 Pip Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_win_pips = backtest_df.loc[backtest_df['result']=="WIN", 'pips'].mean()
                    if pd.isna(avg_win_pips):
                        avg_win_pips = 0
                    st.metric("Avg Pips (WIN)", f"{avg_win_pips:.2f}")
                
                with col2:
                    avg_loss_pips = backtest_df.loc[backtest_df['result']=="LOSS", 'pips'].mean()
                    if pd.isna(avg_loss_pips):
                        avg_loss_pips = 0
                    st.metric("Avg Pips (LOSS)", f"{avg_loss_pips:.2f}")
                
                with col3:
                    net_pips = backtest_df['pips'].sum()
                    st.metric("Total Net Pips", f"{net_pips:.2f}")
                
                with col4:
                    avg_pips = backtest_df['pips'].mean()
                    st.metric("Avg Pips/Trade", f"{avg_pips:.2f}")
            else:
                st.error("No backtest data available")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Please check your API key is correct")
import streamlit as st
import yfinance as yf
import pandas as pd

# --- Helper: pip calculation ---
def pip_difference(pair, entry, exit):
    if "JPY" in pair:
        return (exit - entry) * 100   # 1 pip = 0.01
    elif "XAU" in pair:
        return exit - entry           # treat 1 unit = 1 pip
    else:
        return (exit - entry) * 10000 # 1 pip = 0.0001

# --- Streamlit UI ---
st.title("Strategy Backtest Results")

# Timeframe selector (default daily, but 15m/5m available)
timeframe = st.selectbox("Select timeframe", ["1d", "15m", "5m"], index=0)

# Manual signal input form
with st.form("signal_form"):
    date = st.date_input("Signal Date")
    signal = st.selectbox("Signal", ["BUY", "SELL"])
    pair = st.selectbox("Pair", ["EURUSD", "USDJPY", "GBPUSD", "XAUUSD"])
    stop_loss = st.number_input("Stop Loss", value=0.0)
    take_profit = st.number_input("Take Profit", value=0.0)
    submitted = st.form_submit_button("Add Signal")

# Store signals
if "signals" not in st.session_state:
    st.session_state.signals = pd.DataFrame(columns=["date","signal","pair","stop_loss","take_profit"])

if submitted:
    new_signal = pd.DataFrame([{
        "date": pd.to_datetime(date),
        "signal": signal,
        "pair": pair,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }])
    st.session_state.signals = pd.concat([st.session_state.signals, new_signal], ignore_index=True)

st.write("Signals Table", st.session_state.signals)

# --- Run Backtest ---
if st.button("Run Backtest", key="run_backtest") and not st.session_state.signals.empty:
    # backtest code goes here
    # Download data for pairs
    pairs = {
        "EURUSD": "EURUSD=X",
        "USDJPY": "USDJPY=X",
        "GBPUSD": "GBPUSD=X",
        "XAUUSD": "GC=F"
    }
    data = {}
    for pair, ticker in pairs.items():
        df = yf.download(ticker, start="2026-01-01", end="2026-12-31", interval=timeframe)
        df.index = pd.to_datetime(df.index)
        data[pair] = df

    lookahead = 5
    results, days_to_result, pip_results = [], [], []

    for _, row in st.session_state.signals.iterrows():
        pair_data = data.get(row['pair'])
        if row['date'] in pair_data.index:
            start_idx = pair_data.index.get_loc(row['date'])
            end_idx = min(start_idx + lookahead, len(pair_data)-1)
            window = pair_data.iloc[start_idx:end_idx+1]

            outcome, days_taken, pip_value = "HOLD", None, 0
            entry_price = pair_data.loc[row['date'], 'Open']

            for i, (idx, day) in enumerate(window.iterrows()):
                high, low = day['High'], day['Low']
                if row['signal'] == "BUY":
                    if high >= row['take_profit']:
                        outcome, days_taken = "WIN", i
                        pip_value = pip_difference(row['pair'], entry_price, row['take_profit'])
                        break
                    elif low <= row['stop_loss']:
                        outcome, days_taken = "LOSS", i
                        pip_value = pip_difference(row['pair'], entry_price, row['stop_loss'])
                        break
                elif row['signal'] == "SELL":
                    if low <= row['take_profit']:
                        outcome, days_taken = "WIN", i
                        pip_value = pip_difference(row['pair'], entry_price, row['take_profit'])
                        break
                    elif high >= row['stop_loss']:
                        outcome, days_taken = "LOSS", i
                        pip_value = pip_difference(row['pair'], entry_price, row['stop_loss'])
                        break

            results.append(outcome)
            days_to_result.append(days_taken)
            pip_results.append(pip_value)
        else:
            results.append("NO DATA")
            days_to_result.append(None)
            pip_results.append(0)

    st.session_state.signals['result'] = results
    st.session_state.signals['days_to_result'] = days_to_result
    st.session_state.signals['pips'] = pip_results

    st.write("Backtest Results", st.session_state.signals)

    # Stats
    st.subheader("Backtest Statistics")
    st.write("Win rate:", (st.session_state.signals['result'] == "WIN").mean())
    st.write("Loss rate:", (st.session_state.signals['result'] == "LOSS").mean())
    st.write("Hold rate:", (st.session_state.signals['result'] == "HOLD").mean())
    st.write("Average pips per WIN:", st.session_state.signals.loc[st.session_state.signals['result']=="WIN", 'pips'].mean())
    st.write("Average pips per LOSS:", st.session_state.signals.loc[st.session_state.signals['result']=="LOSS", 'pips'].mean())
    st.write("Net average pips per trade:", st.session_state.signals['pips'].mean())
# --- Manual signal input form ---
# ---- Manual signal input form ----
with st.form("signal_form_manuel"):
    date = st.date_input("Signal Date")
    signal = st.selectbox("Signal", ["BUY", "SELL"])
    pair = st.selectbox("Pair", ["EURUSD", "USDJPY", "GBPUSD", "XAUUSD"])
    stop_loss = st.number_input("Stop Loss", value=0.0)
    take_profit = st.number_input("Take Profit", value=0.0)
    submitted = st.form_submit_button("Add Signal", key="add_signal")

# ---- Run Backtest button ----
# ---- Run Backtest button ----
if st.button("Run Backtest", key="run_backtest_main") and not st.session_state.signals.empty:
    # backtest code goes here
    # (this is the same logic you already have for WIN/LOSS/HOLD checks)

    # Temporary placeholder so Python is happy
    st.write("Backtest running...")

from datetime import timedelta

def calculate_pips(entry, exit, pair, direction):
    if "JPY" in pair:
        pip_size = 0.01
    elif "XAU" in pair or "GOLD" in pair:
        pip_size = 0.1
    else:
        pip_size = 0.0001

    if direction == "BUY":
        diff = exit - entry
    else:  # SELL
        diff = entry - exit

    return diff / pip_size


def run_backtest(signals, price_data):
    wins, losses, holds = [], [], []

    for sig in signals:
        date = sig["date"]
        pair = sig["pair"]
        signal = sig["signal"]
        stop_loss = sig["stop_loss"]

        if date not in price_data[pair]:
            continue

        entry_price = price_data[pair][date]["close"]
        next_day = price_data[pair].get(date + timedelta(days=1))
        if not next_day:
            continue

        if signal == "BUY":
            if next_day["low"] <= stop_loss:
                pips = calculate_pips(entry_price, stop_loss, pair, "BUY")
                losses.append(abs(pips))
            else:
                pips = calculate_pips(entry_price, next_day["close"], pair, "BUY")
                wins.append(pips)

        elif signal == "SELL":
            if next_day["high"] >= stop_loss:
                pips = calculate_pips(entry_price, stop_loss, pair, "SELL")
                losses.append(abs(pips))
            else:
                pips = calculate_pips(entry_price, next_day["close"], pair, "SELL")
                wins.append(pips)
        else:
            holds.append(0)

    total_trades = len(wins) + len(losses) + len(holds)

    stats = {
        "win_rate": len(wins)/total_trades if total_trades else 0,
        "loss_rate": len(losses)/total_trades if total_trades else 0,
        "hold_rate": len(holds)/total_trades if total_trades else 0,
        "avg_win_pips": sum(wins)/len(wins) if wins else 0,
        "avg_loss_pips": sum(losses)/len(losses) if losses else 0,
        "net_avg_pips": (sum(wins) - sum(losses))/total_trades if total_trades else 0
    }

    return stats
