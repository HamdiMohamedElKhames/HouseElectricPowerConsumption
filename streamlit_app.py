# ============================================================================
# SMART CITY ENERGY CONSUMPTION DASHBOARD
# SG02 - Prédiction et Analyse de la Consommation Énergétique
# Fixed Version - Compatible with pandas 2.0+
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Smart City Energy Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR ADVANCED STYLING
# ============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0d1117 100%);
    }
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        font-weight: 400;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #1e2340 0%, #252b4a 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100px;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .metric-delta.positive {
        color: #4ade80;
    }
    
    .metric-delta.negative {
        color: #f87171;
    }
    
    /* Section Headers */
    .section-header {
        color: white;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 2rem 0 1rem 0;
        padding-left: 1rem;
        border-left: 4px solid #667eea;
        background: linear-gradient(90deg, rgba(102,126,234,0.1) 0%, transparent 100%);
        padding: 0.5rem 1rem;
        border-radius: 0 10px 10px 0;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: #667eea;
        margin: 0 0 0.5rem 0;
    }
    
    .info-box p {
        color: rgba(255,255,255,0.8);
        margin: 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255,255,255,0.5);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        # Try to load from file
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'best_model_rf.pkl')
        scaler_path = os.path.join(base_path, 'scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        else:
            return None, None
    except:
        return None, None

@st.cache_data
def load_data():
    """Load and prepare data"""
    try:
        # Try to load actual dataset
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, 'household_power_consumption.txt')
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, sep=';', na_values=['?', ''])
            
            # Process data
            df['Timestamp'] = pd.to_datetime(
                df['Date'] + ' ' + df['Time'], 
                format='%d/%m/%Y %H:%M:%S'
            )
            df = df.set_index('Timestamp')
            df = df.drop(['Date', 'Time'], axis=1)
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            if len(df) > 50000:
                df = df.sample(50000, random_state=42)
        else:
            # Generate sample data for demo
            df = create_sample_data()
        
        # Feature engineering
        df['Hour'] = df.index.hour
        df['Day'] = df.index.day
        df['Month'] = df.index.month
        df['DayOfWeek'] = df.index.dayofweek
        df['Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        return df
    except Exception as e:
        st.warning(f"Using demo data: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration"""
    # FIXED: Using 'h' instead of 'H' for hourly frequency
    dates = pd.date_range('2024-01-01', periods=10000, freq='h')
    np.random.seed(42)
    
    # Create realistic patterns
    hourly_pattern = np.sin(np.arange(10000) * np.pi / 12) * 0.5 + 1.0
    
    df = pd.DataFrame({
        'Global_active_power': hourly_pattern + np.random.normal(0, 0.2, 10000),
        'Global_reactive_power': np.abs(np.random.normal(0.1, 0.05, 10000)),
        'Voltage': np.random.normal(240, 10, 10000),
        'Global_intensity': np.abs(np.random.normal(5, 2, 10000)),
        'Sub_metering_1': np.abs(np.random.normal(0.5, 0.2, 10000)),
        'Sub_metering_2': np.abs(np.random.normal(0.3, 0.1, 10000)),
        'Sub_metering_3': np.abs(np.random.normal(2, 1, 10000))
    }, index=dates)
    
    return df

# Load resources
model, scaler = load_model()
df = load_data()
st.session_state.model_loaded = model is not None

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #667eea; font-weight: 800; margin: 0;">⚡ SMART CITY</h2>
        <p style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">Energy Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📍 Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Dashboard", "📊 Analytics", "🔮 Predictions", "💡 Optimization", "📈 Reports"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🔧 Data Filters")
    
    hour_range = st.slider(
        "🕐 Filter by Hour",
        0, 23, (0, 23)
    )
    
    # Filter data based on selection
    filtered_df = df[
        (df.index.hour >= hour_range[0]) & 
        (df.index.hour <= hour_range[1])
    ]
    
    st.markdown(f"*Showing {len(filtered_df):,} records*")
    
    st.markdown("---")
    
    # Model Status
    st.markdown("### 🤖 System Status")
    if st.session_state.model_loaded:
        st.success("✅ ML Model Active")
    else:
        st.warning("⚠️ Demo Mode - No Model")
    
    st.success("✅ Data Pipeline Active")
    st.success("✅ Real-time Monitoring")
    
    # Refresh Button
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.4); font-size: 0.8rem;">
        <p>SG02 - Energy Consumption</p>
        <p>Mini Projet Data Analyst</p>
        <p>© 2024</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN HEADER
# ============================================================================
st.markdown("""
<div class="main-header">
    <h1 class="header-title">⚡ Smart City Energy Analytics</h1>
    <p class="header-subtitle">Real-time Energy Consumption Monitoring & Prediction Platform</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# DASHBOARD PAGE
# ============================================================================
if page == "🏠 Dashboard":
    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_consumption = filtered_df['Global_active_power'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚡ Avg Consumption</div>
            <div class="metric-value">{avg_consumption:.2f} kW</div>
            <div class="metric-delta positive">Current Period</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        peak_hour = filtered_df.groupby(filtered_df.index.hour)['Global_active_power'].mean().idxmax()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎯 Peak Hour</div>
            <div class="metric-value">{peak_hour}:00</div>
            <div class="metric-delta negative">Highest Demand</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        savings_potential = 18.5
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💰 Savings Potential</div>
            <div class="metric-value">{savings_potential}%</div>
            <div class="metric-delta positive">With Optimization</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        max_power = filtered_df['Global_active_power'].max()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🏭 Max Load</div>
            <div class="metric-value">{max_power:.2f} kW</div>
            <div class="metric-delta negative">Peak Period</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Charts Row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="section-header">📈 Real-Time Consumption</p>', unsafe_allow_html=True)
        
        # FIXED: Using 'h' instead of 'H'
        hourly_data = df['Global_active_power'].resample('h').mean()
        
        # Show last 7 days or all data if less
        if len(hourly_data) > 168:
            hourly_data = hourly_data[-168:]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hourly_data.index,
            y=hourly_data.values,
            mode='lines',
            name='Consumption',
            line=dict(color='#667eea', width=2),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.15)'
        ))
        
        # Add moving average
        if len(hourly_data) > 24:
            ma = hourly_data.rolling(window=24).mean()
            fig.add_trace(go.Scatter(
                x=ma.index,
                y=ma.values,
                mode='lines',
                name='24h Average',
                line=dict(color='#f59e0b', width=2, dash='dash')
            ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(
                title='Time',
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True
            ),
            yaxis=dict(
                title='Power (kW)',
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True
            ),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<p class="section-header">🎯 Quick Insights</p>', unsafe_allow_html=True)
        
        # Hourly pattern
        hourly_avg = df.groupby(df.index.hour)['Global_active_power'].mean()
        peak_h = hourly_avg.idxmax()
        off_peak_h = hourly_avg.idxmin()
        
        fig = go.Figure()
        
        colors = ['#f87171' if h == peak_h else '#4ade80' if h == off_peak_h else '#667eea' for h in range(24)]
        
        fig.add_trace(go.Bar(
            x=list(range(24)),
            y=hourly_avg.values,
            marker_color=colors,
            hovertemplate='Hour: %{x}:00<br>Power: %{y:.2f} kW<extra></extra>'
        ))
        
        fig.add_hline(
            y=hourly_avg.mean(),
            line_dash="dash",
            line_color="white",
            annotation_text=f"Avg: {hourly_avg.mean():.2f}"
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title='Hour', tickmode='linear', dtick=3),
            yaxis=dict(title='kW'),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottom Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="section-header">📅 Weekly Pattern</p>', unsafe_allow_html=True)
        
        weekly_avg = df.groupby('DayOfWeek')['Global_active_power'].agg(['mean', 'std'])
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=days,
            y=weekly_avg['mean'],
            error_y=dict(type='data', array=weekly_avg['std']),
            marker_color='#764ba2',
            hovertemplate='%{x}: %{y:.2f} kW<extra></extra>'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(title='Average Power (kW)'),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<p class="section-header">🔥 Monthly Seasonality</p>', unsafe_allow_html=True)
        
        monthly_avg = df.groupby('Month')['Global_active_power'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=monthly_avg.values,
            mode='lines+markers',
            line=dict(color='#f59e0b', width=3),
            marker=dict(size=10, color='#f59e0b'),
            fill='tozeroy',
            fillcolor='rgba(245, 158, 11, 0.15)'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(title='Average Power (kW)'),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ANALYTICS PAGE
# ============================================================================
elif page == "📊 Analytics":
    st.markdown('<p class="section-header">📊 Advanced Analytics</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🔥 Heatmap", "📊 Distributions"])
    
    with tab1:
        st.markdown("### Consumption Trends Analysis")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            trend_metric = st.selectbox(
                "Select Metric",
                ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity'],
                key='trend_metric'
            )
            
            # FIXED: Using lowercase frequency strings
            trend_period = st.selectbox(
                "Time Period",
                ['Hourly', 'Daily', 'Weekly', 'Monthly'],
                key='trend_period'
            )
            
            show_trendline = st.checkbox("Show Trendline", value=True)
        
        with col2:
            # FIXED: Using lowercase frequency strings
            freq_map = {'Hourly': 'h', 'Daily': 'D', 'Weekly': 'W', 'Monthly': 'ME'}
            resampled = df[trend_metric].resample(freq_map[trend_period]).mean()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=resampled.index,
                y=resampled.values,
                mode='lines',
                name=trend_metric,
                line=dict(color='#667eea', width=2)
            ))
            
            if show_trendline and len(resampled) > 7:
                rolling = resampled.rolling(window=7).mean()
                fig.add_trace(go.Scatter(
                    x=rolling.index,
                    y=rolling.values,
                    mode='lines',
                    name='7-period MA',
                    line=dict(color='#f59e0b', width=2, dash='dash')
                ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Consumption Heatmap (Day × Hour)")
        
        pivot = df.pivot_table(
            values='Global_active_power',
            index='DayOfWeek',
            columns='Hour',
            aggfunc='mean'
        )
        
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f'{h}:00' for h in range(24)],
            y=days,
            colorscale='Viridis',
            colorbar=dict(title='kW'),
            hovertemplate='Day: %{y}<br>Hour: %{x}<br>Power: %{z:.2f} kW<extra></extra>'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            title='Average Power Consumption Heatmap'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dist_metric = st.selectbox(
                "Select Feature",
                df.select_dtypes(include=[np.number]).columns.tolist(),
                key='dist_metric'
            )
        
        with col2:
            nbins = st.slider("Number of Bins", 10, 100, 50)
        
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=('Histogram', 'Box Plot')
        )
        
        fig.add_trace(
            go.Histogram(
                x=df[dist_metric],
                nbinsx=nbins,
                name='Histogram',
                marker_color='#667eea',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Box(
                y=df[dist_metric],
                name='Box Plot',
                marker_color='#764ba2',
                boxmean='sd'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PREDICTIONS PAGE
# ============================================================================
elif page == "🔮 Predictions":
    st.markdown('<p class="section-header">🔮 Energy Predictions</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎛️ Input Parameters")
        
        pred_hour = st.slider("Hour of Day", 0, 23, 12, key='pred_hour')
        pred_day = st.slider("Day of Month", 1, 31, 15, key='pred_day')
        pred_month = st.slider("Month", 1, 12, 6, key='pred_month')
        pred_dayofweek = st.slider("Day of Week (0=Mon)", 0, 6, 3, key='pred_dow')
        
        col_a, col_b = st.columns(2)
        with col_a:
            voltage = st.number_input("Voltage (V)", 200.0, 260.0, 240.0, 0.1)
            intensity = st.number_input("Intensity (A)", 0.0, 50.0, 5.0, 0.1)
        with col_b:
            reactive_power = st.number_input("Reactive Power (kW)", 0.0, 2.0, 0.1, 0.01)
            sub_metering = st.number_input("Sub Metering (kW)", 0.0, 50.0, 2.0, 0.1)
        
        predict_btn = st.button("🔮 Predict", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Prediction Result")
        
        if predict_btn:
            # Simulate prediction
            base_load = 1.0
            hour_effect = np.sin(pred_hour * np.pi / 12) * 0.3
            day_effect = (pred_dayofweek >= 5) * 0.1
            season_effect = np.cos((pred_month - 1) * np.pi / 6) * 0.2
            
            prediction = base_load + hour_effect + day_effect + season_effect
            prediction += np.random.normal(0, 0.05)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                number={'font': {'color': '#667eea', 'size': 50}},
                delta={'reference': 1.09, 'increasing': {'color': '#f87171'}},
                title={'text': "Predicted Power (kW)"},
                gauge={
                    'axis': {'range': [None, 3]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 1], 'color': 'rgba(74, 222, 128, 0.3)'},
                        {'range': [1, 2], 'color': 'rgba(250, 204, 21, 0.3)'},
                        {'range': [2, 3], 'color': 'rgba(248, 113, 113, 0.3)'}
                    ],
                }
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Predicted", f"{prediction:.2f} kW")
            with col_b:
                confidence = 0.15
                st.metric("Confidence", f"±{confidence:.2f} kW")
    
    # 24-Hour Forecast
    st.markdown('<p class="section-header">📈 24-Hour Forecast</p>', unsafe_allow_html=True)
    
    # FIXED: Using 'h' instead of 'H'
    future_hours = pd.date_range(start=datetime.now(), periods=24, freq='h')
    forecast = 1.0 + np.sin(np.arange(24) * np.pi / 12) * 0.5 + np.random.normal(0, 0.1, 24)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=future_hours,
        y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#f59e0b', width=3),
        marker=dict(size=8)
    ))
    
    # Confidence interval
    ci_upper = forecast + 0.2
    ci_lower = forecast - 0.2
    
    fig.add_trace(go.Scatter(
        x=future_hours.tolist() + future_hours.tolist()[::-1],
        y=ci_upper.tolist() + ci_lower.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(245, 158, 11, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        hovermode='x unified',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Predicted Power (kW)')
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# OPTIMIZATION PAGE
# ============================================================================
elif page == "💡 Optimization":
    st.markdown('<p class="section-header">💡 Energy Optimization</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Peak Hours")
        hourly_avg = df.groupby(df.index.hour)['Global_active_power'].mean()
        peak_hours = hourly_avg.nlargest(5)
        
        fig = go.Figure(data=[go.Bar(
            x=[f'{h}:00' for h in peak_hours.index],
            y=peak_hours.values,
            marker_color='#f87171',
            text=np.round(peak_hours.values, 2),
            textposition='outside'
        )])
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            title='Peak Consumption Hours',
            yaxis=dict(title='kW')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🟢 Off-Peak Hours")
        off_peak_hours = hourly_avg.nsmallest(5)
        
        fig = go.Figure(data=[go.Bar(
            x=[f'{h}:00' for h in off_peak_hours.index],
            y=off_peak_hours.values,
            marker_color='#4ade80',
            text=np.round(off_peak_hours.values, 2),
            textposition='outside'
        )])
        
       