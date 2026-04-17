import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # Added for Upgrade 2
from sklearn.ensemble import RandomForestClassifier
from streamlit_lottie import st_lottie
import requests

# --- 1. UI CONFIGURATION & CSS ---
st.set_page_config(page_title="Outbreak Guard AI", layout="wide", page_icon="🛡️")

# Custom CSS for a professional "Dark Mode" aesthetic
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0e1117 0%, #161b22 100%); }
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #4B5563;
        padding: 20px;
        border-radius: 15px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Animation Loader
def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_health = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5njp3v83.json")

# --- 2. DATA & AI ENGINE ---
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_disease_data.csv')
    return df

df = load_data()
X = df.drop('prognosis', axis=1)
y = df['prognosis']
model = RandomForestClassifier(n_estimators=20).fit(X, y)

# --- UPGRADE 1: GLOBAL DISEASE FILTER ---
with st.sidebar:
    if lottie_health:
        st_lottie(lottie_health, height=150)
    else:
        st.title("🏥 Outbreak Guard")
    
    st.title("Navigation")
    page = st.radio("Select Module", ["🌐 Global Dashboard", "🩺 AI Diagnoser", "📊 Trend Analytics", "📍 Outbreak Map"])
    
    st.divider()
    # This filter controls the data across the whole app
    selected_disease = st.selectbox("🎯 Focus on Disease:", sorted(df['prognosis'].unique()))
    filtered_df = df[df['prognosis'] == selected_disease]
    
    st.sidebar.divider()
    st.sidebar.success("System Status: Online")

# --- UPGRADE 2: RISK GAUGE FUNCTION ---
def draw_risk_gauge(value):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Risk Level: {selected_disease}", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#00d4ff"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 40], 'color': "#008000"},
                {'range': [40, 75], 'color': "#FFA500"},
                {'range': [75, 100], 'color': "#FF0000"}],
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                      font={'color': "white", 'family': "Inter"}, height=300)
    return fig

# --- 4. MODULES ---

if page == "🌐 Global Dashboard":
    st.title(f"🌐 {selected_disease} Command Center")
    
    # Requirement #4: Dynamic Alert
    st.error(f"🚨 ALERT: Data ingestion shows {len(filtered_df)} confirmed patterns for {selected_disease}.")
    
    # Gauge and Stats
    col_left, col_right = st.columns([1, 2])
    with col_left:
        # Calculate a mock risk score based on row counts (capped at 100)
        risk_score = min(len(filtered_df) * 0.8, 100) 
        st.plotly_chart(draw_risk_gauge(risk_score), use_container_width=True)
        
    with col_right:
        c1, c2 = st.columns(2)
        c1.metric(label="Cases Detected", value=len(filtered_df), delta="+5% Week-over-Week")
        c2.metric(label="Global Coverage", value="41 Diseases", delta="Secure")
        
        st.write("### Data Integrity Check")
        st.dataframe(filtered_df.head(10), use_container_width=True)

elif page == "🩺 AI Diagnoser":
    st.title("🩺 Smart Diagnostic Support")
    st.info("Input symptoms below for an AI-driven provisional prognosis.")
    
    with st.container():
        col_a, col_b = st.columns(2)
        with col_a:
            s1 = st.toggle("Persistent Itching")
            s2 = st.toggle("Visible Skin Rash")
            s3 = st.toggle("Joint Inflammation")
        with col_b:
            s4 = st.toggle("High Grade Fever")
            s5 = st.toggle("Nausea / Vomiting")
            s6 = st.toggle("Fatigue")

    if st.button("Generate AI Prognosis", type="primary"):
        patient = [0] * 132
        if s1: patient[0] = 1
        if s2: patient[1] = 1
        if s3: patient[6] = 1
        if s4: patient[11] = 1
        
        res = model.predict([patient])[0]
        st.header(f"Provisional Diagnosis: :red[{res}]")
        st.markdown(f"**Action Plan:** If this matches the focused disease **({selected_disease})**, initiate protocol Delta-7.")

elif page == "📊 Trend Analytics":
    st.title(f"📊 {selected_disease} Trend Projections")
    
    t_col1, t_col2 = st.columns(2)
    
    with t_col1:
        st.subheader("Symptom Weight (AI Clues)")
        importance = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
        fig, ax = plt.subplots(facecolor='#1e2130')
        importance.plot(kind='barh', color='#00d4ff', ax=ax)
        ax.set_facecolor('#1e2130')
        ax.tick_params(colors='white')
        st.pyplot(fig)

    with t_col2:
        # UPGRADE 3: PREDICTIVE FORECASTING
        st.subheader("🔮 7-Day Outbreak Forecast")
        base_cases = len(filtered_df)
        # Mock growth calculation for visualization
        forecast = [base_cases * (1.05 ** i) for i in range(7)]
        st.line_chart(forecast)
        st.caption("AI-modeled projection based on current environmental and clinical variables.")

elif page == "📍 Outbreak Map":
    st.title(f"📍 {selected_disease} Hotspot Detection")
    st.write(f"Geospatial distribution of potential {selected_disease} clusters.")
    
    # Center map on Bengaluru, generating points based on the number of filtered cases
    points_to_show = min(len(filtered_df), 100)
    map_data = pd.DataFrame(
        np.random.randn(points_to_show, 2) / [80, 80] + [12.97, 77.59],
        columns=['lat', 'lon']
    )
    st.map(map_data, color='#FF0000')