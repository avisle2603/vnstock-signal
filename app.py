import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="VNStock Signal Engine",
    page_icon="📈",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .signal-card {
        background: #1a1d27;
        border: 1px solid #2d3142;
        border-radius: 12px;
        padding: 20px;
        font-family: monospace;
        white-space: pre-wrap;
        color: #e2e8f0;
        font-size: 14px;
        line-height: 1.7;
    }
    .metric-box {
        background: #1a1d27;
        border: 1px solid #2d3142;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .buy-signal { border-left: 4px solid #22c55e; }
    .sell-signal { border-left: 4px solid #ef4444; }
    .hold-signal { border-left: 4px solid #f59e0b; }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────

def fetch_stock_data(ticker, days=120):
    try:
        from vnstock3 import Vnstock
        stock = Vnstock().stock(symbol=ticker.upper(), source='VCI')
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        df = stock.quote.history(start=start_date, end=end_date, interval='1D')
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Không thể tải dữ liệu cho {ticker}: {e}")
        return None


def compute_indicators(df):
    df = df.copy()
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    bb = ta.bbands(df['close'], length=20)
    df['bb_upper'] = bb['BBU_20_2.0']
    df['bb_lower'] = bb['BBL_20_2.0']
    df['bb_mid']   = bb['BBM_20_2.0']
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['atr']   = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    return df


def format_snapshot(ticker, df):
    latest = df.iloc[-1]
    vol_ratio = latest['volume'] / latest['vol_ma20'] if latest['vol_ma20'] > 0 else 1
    change_3w = ((latest['close'] - df.iloc[-15]['close']) / df.iloc[-15]['close']) * 100
    rsi_prev  = df.iloc[-4]['rsi']

    snapshot = f"""STOCK SNAPSHOT — {ticker.upper()}
Date: {datetime.now().strftime('%Y-%m-%d')}
Timeframe: Daily chart

PRICE DATA:
Current price: {latest['close']:,.0f} VND
Today's open:  {latest['open']:,.0f} VND
Today's high:  {latest['high']:,.0f} VND
Today's low:   {latest['low']:,.0f} VND
Volume today:  {latest['volume']:,.0f} shares
Volume 20-day average: {latest['vol_ma20']:,.0f} shares

INDICATOR READINGS:
RSI (14): {latest['rsi']:.1f} — {"rising" if latest['rsi'] > rsi_prev else "falling"} from {rsi_prev:.1f} three days ago
MACD: Histogram at {latest['macd_hist']:.2f} and {"positive/turning up" if latest['macd_hist'] > 0 else "negative/turning down"}
Bollinger Bands: Price {"touching/near lower band" if latest['close'] <= latest['bb_lower']*1.01 else "touching/near upper band" if latest['close'] >= latest['bb_upper']*0.99 else "inside bands"}. Upper: {latest['bb_upper']:,.0f}, Lower: {latest['bb_lower']:,.0f}
EMA 20: {latest['ema20']:,.0f} — price currently {"above" if latest['close'] > latest['ema20'] else "below"} it
EMA 50: {latest['ema50']:,.0f} — price currently {"above" if latest['close'] > latest['ema50'] else "below"} it
ATR (14): {latest['atr']:,.0f} VND per day

RECENT PRICE ACTION:
Stock has moved {change_3w:.1f}% over the past 3 weeks.
Volume today is {vol_ratio:.1f}x the 20-day average — {"above average, suggests conviction" if vol_ratio > 1.2 else "below average, suggests weak move"}
Recent 20-day low: {df['low'].tail(20).min():,.0f} VND
Recent 20-day high: {df['high'].tail(20).max():,.0f} VND

SECTOR CONTEXT:
Vietnamese stock — HOSE listed. Please analyze in context of Vietnamese market conditions."""
    return snapshot


def call_flowise(snapshot, flowise_url):
    try:
        response = requests.post(
            flowise_url,
            json={"question": snapshot},
            headers={"Content-Type": "application/json"},
            timeout=90
        )
        if response.status_code == 200:
            data = response.json()
            return data.get('text', data.get('answer', str(data)))
        else:
            return f"API Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Connection error: {e}"


def plot_chart(df, ticker):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f'{ticker.upper()} — Daily', 'Volume', 'RSI (14)']
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color='#22c55e',
        decreasing_line_color='#ef4444',
        name='Price'
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'],
        line=dict(color='rgba(148,163,184,0.4)', width=1), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'],
        line=dict(color='rgba(148,163,184,0.4)', width=1),
        fill='tonexty', fillcolor='rgba(148,163,184,0.05)', name='BB Lower'), row=1, col=1)

    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['ema20'],
        line=dict(color='#f59e0b', width=1.5), name='EMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema50'],
        line=dict(color='#60a5fa', width=1.5), name='EMA 50'), row=1, col=1)

    # Volume bars
    colors = ['#ef4444' if row['close'] < row['open'] else '#22c55e'
              for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'],
        marker_color=colors, name='Volume', opacity=0.7), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'],
        line=dict(color='#a78bfa', width=1.5), name='RSI'), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(239,68,68,0.08)',
        line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor='rgba(34,197,94,0.08)',
        line_width=0, row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444",
        line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e",
        line_width=1, row=3, col=1)

    fig.update_layout(
        height=680,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font_color='#94a3b8',
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=11)
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.update_xaxes(gridcolor='#1e2535', showgrid=True)
    fig.update_yaxes(gridcolor='#1e2535', showgrid=True)

    return fig

# ── UI ────────────────────────────────────────────────────────

st.title("📈 VNStock Signal Engine")
st.caption("Powered by AI • Vietnamese Market Analysis")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    flowise_url = st.text_input(
        "Flowise API URL",
        placeholder="https://cloud.flowiseai.com/api/v1/prediction/xxx",
        help="Get this from the </> icon in your Flowise chatflow"
    )
    st.divider()
    st.markdown("**Popular tickers**")
    quick_tickers = ["VNM", "VIC", "HPG", "VHM", "MSN", "FPT", "VCB", "TCB"]
    cols = st.columns(2)
    for i, t in enumerate(quick_tickers):
        if cols[i % 2].button(t, use_container_width=True):
            st.session_state['ticker'] = t

# Main input
col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input(
        "Enter Vietnamese stock ticker",
        value=st.session_state.get('ticker', ''),
        placeholder="e.g. VNM, VIC, HPG, FPT",
        label_visibility="collapsed"
    ).upper().strip()
with col2:
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

if analyze_btn and ticker_input:
    if not flowise_url:
        st.warning("⚠️ Please enter your Flowise API URL in the sidebar first.")
        st.stop()

    # Fetch data
    with st.spinner(f"Fetching live data for {ticker_input}..."):
        df = fetch_stock_data(ticker_input)

    if df is not None and len(df) > 50:
        df = compute_indicators(df)
        latest = df.iloc[-1]

        # Key metrics row
        vol_ratio = latest['volume'] / latest['vol_ma20']
        change_1d = ((latest['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close']) * 100

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Price", f"{latest['close']:,.0f}", f"{change_1d:+.2f}%")
        m2.metric("RSI (14)", f"{latest['rsi']:.1f}")
        m3.metric("ATR", f"{latest['atr']:,.0f}")
        m4.metric("Volume ratio", f"{vol_ratio:.1f}x")
        m5.metric("EMA 20", f"{latest['ema20']:,.0f}")

        # Chart
        st.plotly_chart(plot_chart(df, ticker_input), use_container_width=True)

        # Signal analysis
        st.subheader("🤖 AI Signal Analysis")
        with st.spinner("Running 4-stage signal analysis... (15–30 seconds)"):
            snapshot = format_snapshot(ticker_input, df)
            signal_output = call_flowise(snapshot, flowise_url)

        # Determine signal type for styling
        signal_class = "hold-signal"
        if any(x in signal_output.upper() for x in ["STRONG BUY", "BUY"]):
            signal_class = "buy-signal"
        elif any(x in signal_output.upper() for x in ["STRONG SELL", "SELL"]):
            signal_class = "sell-signal"

        st.markdown(
            f'<div class="signal-card {signal_class}">{signal_output}</div>',
            unsafe_allow_html=True
        )

        # Show raw snapshot in expander
        with st.expander("📋 View raw data snapshot sent to AI"):
            st.code(snapshot)

    else:
        st.error(f"Could not fetch enough data for {ticker_input}. Check the ticker symbol.")

elif not analyze_btn:
    st.info("👆 Enter a Vietnamese stock ticker above and click Analyze to get a live signal.")
