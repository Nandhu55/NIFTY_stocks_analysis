import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# ===== PAGE CONFIG =====
st.set_page_config(page_title="NIFTY AI Dashboard", layout="wide")
st.title("📈 NIFTY Stocks AI Dashboard")

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv")

    # 🔥 CLEAN COLUMN NAMES (IMPORTANT)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    df.columns = df.columns.str.replace('%', 'pct')

    return df

df = load_data()

# ===== SIDEBAR =====
st.sidebar.header("🔍 Filter")
company = st.sidebar.selectbox("Select Company", df["Company"].unique())
data = df[df["Company"] == company].copy()

# ===== CONVERT NUMERIC =====
numeric_cols = ['Open', 'High', 'Low', 'Price', 'Volume_lacs']
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# ===== KPI =====
st.subheader(f"📊 {company} Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Price", f"{data['Price'].iloc[-1]:.2f}")
col2.metric("High", f"{data['High'].iloc[-1]:.2f}")
col3.metric("Low", f"{data['Low'].iloc[-1]:.2f}")

# ===== CHARTS (ALL STOCKS) =====
st.subheader("📊 Market Overview (All Stocks)")

chart_option = st.radio(
    "Choose Chart",
    ["Yearly Growth", "Top Gainers", "Top Losers", "Correlation"],
    horizontal=True
)

# ---- YEARLY GROWTH ----
if chart_option == "Yearly Growth":

    df_sorted = df.sort_values("365_d_pct_chng")

    fig = px.bar(
        df_sorted,
        x="Company",
        y="365_d_pct_chng",
        color="365_d_pct_chng",
        title="📈 Yearly Growth (Sorted)"
    )

    fig.update_layout(xaxis_tickangle=-90)
    st.plotly_chart(fig, use_container_width=True)

# ---- TOP GAINERS ----
elif chart_option == "Top Gainers":

    top_gainers = df.sort_values("365_d_pct_chng", ascending=False).head(10)

    fig = px.bar(
        top_gainers,
        x="Company",
        y="365_d_pct_chng",
        color="365_d_pct_chng",
        title="🚀 Top 10 Gainers"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---- TOP LOSERS ----
elif chart_option == "Top Losers":

    top_losers = df.sort_values("365_d_pct_chng", ascending=True).head(10)

    fig = px.bar(
        top_losers,
        x="Company",
        y="365_d_pct_chng",
        color="365_d_pct_chng",
        title="📉 Top 10 Losers"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---- CORRELATION ----
elif chart_option == "Correlation":

    numeric_df = df.select_dtypes(include='number')

    fig = px.imshow(
        numeric_df.corr(),
        text_auto=True,
        title="🔥 Market Correlation"
    )

    st.plotly_chart(fig, use_container_width=True)

# ===== MODEL TRAINING =====
st.subheader("🤖 AI Prediction")

features = ['Open', 'High', 'Low', 'Volume_lacs']
target = 'Price'

model_data = data.dropna()

X = model_data[features]
y = model_data[target]

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# ===== MANUAL INPUT =====
st.subheader("✍️ Manual Prediction")

col1, col2, col3, col4 = st.columns(4)

open_val = col1.number_input("Open", value=float(data['Open'].iloc[-1]))
high_val = col2.number_input("High", value=float(data['High'].iloc[-1]))
low_val = col3.number_input("Low", value=float(data['Low'].iloc[-1]))
vol_val = col4.number_input("Volume", value=float(data['Volume_lacs'].iloc[-1]))

if st.button("Predict Custom Price"):
    input_data = np.array([[open_val, high_val, low_val, vol_val]])
    pred = model.predict(input_data)[0]

    st.success(f"🎯 Predicted Price: {pred:.2f}")

# ===== DATA TABLE =====
with st.expander("📄 View Data"):
    st.dataframe(data.tail(50), use_container_width=True)

# ===== FOOTER =====
st.markdown("---")
st.markdown("🚀 Built with Streamlit | ML Dashboard")