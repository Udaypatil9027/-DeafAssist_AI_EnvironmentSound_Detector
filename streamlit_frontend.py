import streamlit as st
import socket
import json
import time
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from collections import deque
from datetime import datetime
import base64
from streamlit.components.v1 import html

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="DeafAssist AI | Professional Sound Detection System",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CUSTOM CSS - PROFESSIONAL DESIGN
# =====================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1433 100%);
    }
    
    /* Main Container */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* Header Section */
    .header-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .logo-icon {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 20px;
        padding: 0.5rem;
        width: 70px;
        height: 70px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    
    .title-section h1 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, #fff, #a8b5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .title-section p {
        color: rgba(255,255,255,0.7);
        margin: 0;
        font-size: 0.9rem;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-connected {
        background: rgba(81, 207, 102, 0.2);
        border: 1px solid #51cf66;
        color: #51cf66;
    }
    
    .status-disconnected {
        background: rgba(255, 107, 107, 0.2);
        border: 1px solid #ff6b6b;
        color: #ff6b6b;
    }
    
    /* Detection Card */
    .detection-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .detection-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        border-color: rgba(102,126,234,0.5);
    }
    
    .hazard-card {
        background: linear-gradient(135deg, rgba(245,87,108,0.2), rgba(240,147,251,0.2));
        border: 1px solid rgba(245,87,108,0.5);
        animation: pulseGlow 1.5s infinite;
    }
    
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(245,87,108,0.2); }
        50% { box-shadow: 0 0 40px rgba(245,87,108,0.5); }
    }
    
    .detection-label {
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    .detection-sound {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #fff, #a8b5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .confidence-container {
        background: rgba(0,0,0,0.3);
        border-radius: 15px;
        padding: 0.3rem;
        margin-top: 1rem;
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #51cf66, #ffd93d);
        border-radius: 12px;
        padding: 0.6rem;
        text-align: center;
        color: white;
        font-weight: 600;
        transition: width 0.3s ease;
    }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        background: rgba(255,255,255,0.08);
        border-color: rgba(102,126,234,0.5);
    }
    
    .metric-icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(255,255,255,0.03);
        border-radius: 20px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(10,14,39,0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        color: white;
        font-weight: 500;
    }
    
    /* Info Box */
    .stInfo {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 16px;
        color: rgba(255,255,255,0.9);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255,255,255,0.4);
        font-size: 0.8rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# CONSTANTS
# =====================================================================
CLASSES = [
    "Air Conditioner", "Car Horn", "Children Playing", 
    "Dog Bark", "Drilling", "Engine Idling", 
    "Gun Shot", "Jackhammer", "Siren", "Street Music"
]

HAZARDOUS_SOUNDS = ["Car Horn", "Siren", "Gun Shot", "Dog Bark"]
HOST = '127.0.0.1'
PORT = 9999

# Session State
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = "None"
if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = 0.0
if 'all_predictions' not in st.session_state:
    st.session_state.all_predictions = [0.0] * len(CLASSES)
if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=100)
if 'confidence_history' not in st.session_state:
    st.session_state.confidence_history = deque(maxlen=100)
if 'timestamp_history' not in st.session_state:
    st.session_state.timestamp_history = deque(maxlen=100)
if 'sock' not in st.session_state:
    st.session_state.sock = None

# =====================================================================
# FUNCTIONS
# =====================================================================
@st.cache_resource
def connect_to_server():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect((HOST, PORT))
        sock.setblocking(False)
        return sock
    except Exception:
        return None

def receive_data(sock):
    try:
        data = sock.recv(8192).decode()
        if data:
            lines = data.strip().split('\n')
            for line in lines:
                if line:
                    return json.loads(line)
    except:
        pass
    return None

def create_gauge_chart(confidence, title="Confidence"):
    """Create a gauge chart for confidence display"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        title = {'text': title, 'font': {'color': 'white', 'size': 14}},
        gauge = {
            'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickwidth': 2},
            'bar': {'color': "#51cf66", 'thickness': 0.3},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.3)",
            'steps': [
                {'range': [0, 30], 'color': "rgba(255,107,107,0.3)"},
                {'range': [30, 70], 'color': "rgba(255,221,61,0.3)"},
                {'range': [70, 100], 'color': "rgba(81,207,102,0.3)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100
            }
        }
    ))
    fig.update_layout(
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Inter'},
        margin=dict(l=30, r=30, t=50, b=30)
    )
    return fig

def create_top_predictions_chart(predictions, selected_hazards, top_n=8):
    """Create horizontal bar chart for top predictions"""
    sorted_indices = np.argsort(predictions)[::-1][:top_n]
    top_sounds = [CLASSES[i] for i in sorted_indices]
    top_probs = [predictions[i] * 100 for i in sorted_indices]
    
    colors = ['#ff6b6b' if sound in selected_hazards else '#51cf66' for sound in top_sounds]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_probs,
            y=top_sounds,
            orientation='h',
            marker_color=colors,
            text=[f"{prob:.1f}%" for prob in top_probs],
            textposition='outside',
            textfont=dict(color='white', size=11),
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>'
        )
    ])
    fig.update_layout(
        title={
            'text': '📊 Real-time Predictions',
            'font': {'color': 'white', 'size': 18, 'family': 'Inter'},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=400,
        plot_bgcolor='rgba(0,0,0,0.2)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Inter'},
        xaxis={
            'title': 'Confidence (%)',
            'title_font': {'color': 'white'},
            'gridcolor': 'rgba(255,255,255,0.1)',
            'range': [0, 100]
        },
        yaxis={
            'title': 'Sound Class',
            'title_font': {'color': 'white'},
            'gridcolor': 'rgba(255,255,255,0.1)'
        },
        showlegend=False,
        margin=dict(l=150, r=50, t=60, b=30)
    )
    return fig

# =====================================================================
# MAIN UI
# =====================================================================
def main():
    # Header Section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="header-container">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div class="logo-icon">
                        🎧
                    </div>
                    <div class="title-section">
                        <h1>DeafAssist AI</h1>
                        <p>Professional Environmental Sound Detection System</p>
                    </div>
                </div>
                <div>
                    <div class="status-badge {}">
                        {}
                    </div>
                </div>
            </div>
        </div>
        """.format(
            "status-connected" if st.session_state.connected else "status-disconnected",
            "● CONNECTED" if st.session_state.connected else "○ DISCONNECTED"
        ), unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎮 Control Center")
        st.markdown("---")
        
        # Connection
        if not st.session_state.connected:
            if st.button("🔌 Connect to Backend", use_container_width=True):
                sock = connect_to_server()
                if sock:
                    st.session_state.sock = sock
                    st.session_state.connected = True
                    st.rerun()
                else:
                    st.error("❌ Backend not running")
        
        st.markdown("---")
        
        # Settings
        st.markdown("#### ⚙️ Detection Parameters")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.3,
            max_value=0.95,
            value=0.55,
            step=0.05,
            help="Higher values = more accurate but fewer detections"
        )
        
        detection_hold = st.slider(
            "Display Hold Time",
            min_value=0.3,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="How long to show detection after sound stops"
        )
        
        st.markdown("---")
        
        # Hazard Configuration
        st.markdown("#### ⚠️ Safety Alerts")
        enable_hazards = st.checkbox("Enable Hazard Alerts", value=True)
        
        selected_hazards = st.multiselect(
            "Hazardous Sound Classes",
            options=CLASSES,
            default=HAZARDOUS_SOUNDS,
            help="These sounds will trigger visual warnings"
        )
        
        st.markdown("---")
        
        # Statistics
        st.markdown("#### 📊 Detection Stats")
        if len(st.session_state.history) > 0:
            df = pd.DataFrame(list(st.session_state.history), columns=['Sound'])
            counts = df['Sound'].value_counts()
            for sound, count in list(counts.items())[:5]:
                st.markdown(f"• **{sound}**: {count} times")
        else:
            st.markdown("*No detections yet*")
        
        st.markdown("---")
        
        # Actions
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history.clear()
            st.session_state.confidence_history.clear()
            st.session_state.timestamp_history.clear()
            st.success("History cleared!")
    
    # Main Content
    if st.session_state.connected and st.session_state.sock:
        # Receive data
        data = receive_data(st.session_state.sock)
        if data:
            st.session_state.current_prediction = data['prediction']
            st.session_state.current_confidence = data['confidence']
            st.session_state.all_predictions = data['all_predictions']
            
            if data['confidence'] >= confidence_threshold:
                st.session_state.history.append(data['prediction'])
                st.session_state.confidence_history.append(data['confidence'])
                st.session_state.timestamp_history.append(datetime.now())
        
        current_conf = st.session_state.current_confidence
        current_pred = st.session_state.current_prediction
        is_hazard = enable_hazards and current_pred in selected_hazards
        
        # Detection Card
        if current_conf >= confidence_threshold and current_pred != "None":
            if is_hazard:
                st.markdown(f"""
                <div class="detection-card hazard-card">
                    <div class="detection-label">⚠️ HAZARD ALERT</div>
                    <div class="detection-sound">{current_pred.upper()}</div>
                    <div class="confidence-container">
                        <div class="confidence-bar" style="width: {current_conf*100}%;">
                            Confidence: {current_conf*100:.1f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="detection-card">
                    <div class="detection-label">🎯 CURRENT DETECTION</div>
                    <div class="detection-sound">{current_pred}</div>
                    <div class="confidence-container">
                        <div class="confidence-bar" style="width: {current_conf*100}%;">
                            Confidence: {current_conf*100:.1f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🎤 **Monitoring Environment** — System ready for sound detection")
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        metrics_data = [
            ("🎯", "Current Sound", current_pred),
            ("📊", "Confidence", f"{current_conf*100:.1f}%"),
            ("📈", "Total Detections", len(st.session_state.history)),
            ("⚡", "Status", "Active" if current_conf > 0.3 else "Idle")
        ]
        
        for idx, (icon, label, value) in enumerate(metrics_data):
            with [col1, col2, col3, col4][idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Charts Row
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            gauge_chart = create_gauge_chart(current_conf)
            st.plotly_chart(gauge_chart, use_container_width=True, key="gauge")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            predictions = np.array(st.session_state.all_predictions)
            chart = create_top_predictions_chart(predictions, selected_hazards, top_n=8)
            st.plotly_chart(chart, use_container_width=True, key="predictions")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # All Classes Distribution
        with st.expander("📊 Complete Probability Distribution", expanded=False):
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            predictions = np.array(st.session_state.all_predictions)
            sorted_indices = np.argsort(predictions)[::-1]
            
            # Create columns for better display
            cols = st.columns(2)
            for idx, col in enumerate(cols):
                start_idx = idx * 5
                end_idx = start_idx + 5
                with col:
                    for i in range(start_idx, min(end_idx, len(sorted_indices))):
                        if i < len(sorted_indices):
                            sound = CLASSES[sorted_indices[i]]
                            prob = predictions[sorted_indices[i]]
                            is_haz = sound in selected_hazards
                            color = "#ff6b6b" if is_haz and prob > 0.3 else "#51cf66"
                            
                            st.markdown(f"""
                            <div style="margin-bottom: 1rem;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                                    <span style="color: white; font-weight: 500;">{i+1}. {sound}</span>
                                    <span style="color: rgba(255,255,255,0.7);">{prob*100:.1f}%</span>
                                </div>
                                <div style="background: rgba(255,255,255,0.1); border-radius: 10px; overflow: hidden;">
                                    <div style="width: {prob*100}%; background: {color}; height: 8px; border-radius: 10px;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detection History
        if len(st.session_state.history) > 0:
            with st.expander("📜 Detection History", expanded=False):
                df = pd.DataFrame({
                    'Time': [t.strftime("%H:%M:%S") for t in list(st.session_state.timestamp_history)],
                    'Sound': list(st.session_state.history),
                    'Confidence': [f"{c*100:.1f}%" for c in list(st.session_state.confidence_history)]
                })
                
                st.dataframe(df.tail(20), use_container_width=True, height=300)
                
                # Export functionality
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Export History (CSV)",
                    data=csv,
                    file_name=f"deafassist_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Auto-refresh
        time.sleep(0.05)
        st.rerun()
    
    else:
        # Welcome Screen
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <div style="background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border-radius: 24px; padding: 3rem; border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-size: 5rem; margin-bottom: 1rem;">🎧</div>
                    <h2 style="color: white; margin-bottom: 1rem;">Welcome to DeafAssist AI</h2>
                    <p style="color: rgba(255,255,255,0.7); margin-bottom: 2rem;">
                        Professional Real-time Environmental Sound Detection System
                    </p>
                    
                    <div style="background: rgba(0,0,0,0.3); border-radius: 16px; padding: 1.5rem; margin: 1rem 0;">
                        <h3 style="color: white; margin-bottom: 1rem;">🚀 Quick Start</h3>
                        <p style="color: rgba(255,255,255,0.8); font-family: monospace; background: rgba(0,0,0,0.5); padding: 0.5rem; border-radius: 8px;">
                            Terminal 1: python audio_backend_accurate.py
                        </p>
                        <p style="color: #51cf66; margin-top: 1rem;">⬆️ First start the backend server</p>
                        <p style="color: rgba(255,255,255,0.8); margin-top: 1rem;">
                            Then click <strong style="color: #51cf66;">Connect to Backend</strong> in the sidebar
                        </p>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 2rem;">
                        <div>
                            <div style="font-size: 2rem;">🎯</div>
                            <div style="color: white; font-weight: 600;">10 Classes</div>
                            <div style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">UrbanSound8K</div>
                        </div>
                        <div>
                            <div style="font-size: 2rem;">⚡</div>
                            <div style="color: white; font-weight: 600;">Real-time</div>
                            <div style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">Low Latency</div>
                        </div>
                        <div>
                            <div style="font-size: 2rem;">⚠️</div>
                            <div style="color: white; font-weight: 600;">Hazard Alerts</div>
                            <div style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">Visual Warnings</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>DeafAssist AI — Professional Sound Detection System | Powered by Deep Learning</p>
        <p style="font-size: 0.7rem;">© 2024 | Real-time Environmental Sound Classification</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()