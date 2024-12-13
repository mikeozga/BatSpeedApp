import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

# Pitch type mapping
PITCH_TYPES = {
    "FA": "Fastball",
    "FF": "Four-seam FB",
    "FT": "Two-seam FB",
    "FC": "Cutter",
    "FS": "Splitter",
    "FO": "Forkball",
    "SI": "Sinker",
    "ST": "Sweeper",
    "SL": "Slider",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "SC": "Screwball",
    "GY": "Gyroball",
    "SV": "Slurve",
    "CS": "Slow Curve",
    "CH": "Changeup",
    "KN": "Knuckleball",
    "EP": "Eephus Pitch"
}

# Set page config
st.set_page_config(page_title="Bat Speed Analysis", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("heatmap.csv")
    df_cleaned = df.dropna(subset=['plate_x', 'plate_z', 'bat_speed', 'p_throws'])
    df_cleaned['pitch_type_desc'] = df_cleaned['pitch_type'].map(PITCH_TYPES)
    return df_cleaned

# Load data
df_cleaned = load_data()

# Calculate league average bat speed (top 90% of swings)
bat_speed_threshold = np.percentile(df_cleaned['bat_speed'], 10)
filtered_speeds = df_cleaned[df_cleaned['bat_speed'] > bat_speed_threshold]['bat_speed']
league_avg_speed = filtered_speeds.mean()

# Title
st.title('Bat Speed by Pitch Location')

# Create columns for filters
col1, col2, col3 = st.columns(3)

with col1:
    selected_batter = st.selectbox(
        'Select Batter:',
        ['All'] + sorted(df_cleaned['batter_name'].unique().tolist())
    )

with col2:
    selected_pitch_types = st.multiselect(
        'Select Pitch Type(s):',
        ['All'] + sorted(df_cleaned['pitch_type_desc'].dropna().unique().tolist()),
        default=['All']
    )

with col3:
    selected_hand = st.selectbox(
        'Pitcher Throws:',
        ['All'] + sorted(df_cleaned['p_throws'].unique().tolist())
    )

# Filter data
df_filtered = df_cleaned.copy()

if selected_batter != 'All':
    df_filtered = df_filtered[df_filtered['batter_name'] == selected_batter]

if 'All' not in selected_pitch_types:
    df_filtered = df_filtered[df_filtered['pitch_type_desc'].isin(selected_pitch_types)]

if selected_hand != 'All':
    df_filtered = df_filtered[df_filtered['p_throws'] == selected_hand]

# Create heatmap
x = np.linspace(-3, 3, 50)
y = np.linspace(-1, 5, 50)
xx, yy = np.meshgrid(x, y)

xy_points = np.vstack([df_filtered['plate_x'], df_filtered['plate_z']])
kde = gaussian_kde(xy_points)

z = np.zeros(xx.shape)
bat_speeds = np.zeros(xx.shape)

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        point = np.array([[xx[i,j]], [yy[i,j]]])
        weights = kde.evaluate(point)
        nearby_points = ((df_filtered['plate_x'] - xx[i,j])**2 + 
                       (df_filtered['plate_z'] - yy[i,j])**2 < 0.5)
        
        if any(nearby_points):
            bat_speeds[i,j] = np.average(df_filtered.loc[nearby_points, 'bat_speed']) - league_avg_speed
            z[i,j] = weights[0]
        else:
            bat_speeds[i,j] = np.nan
            z[i,j] = 0

z_norm = (z - z.min()) / (z.max() - z.min())

fig = go.Figure(data=go.Heatmap(
    x=x,
    y=y,
    z=bat_speeds * z_norm,
    colorscale='RdBu_r',
    zmid=0,
    zmin=-25,
    zmax=25,
    colorbar=dict(title='Bat Speed vs. League Avg (mph)'),
))

# Add strike zone rectangle
fig.add_shape(
    type="rect",
    x0=-0.83, y0=1.5, x1=0.83, y1=3.5,
    line=dict(color="Black", width=2, dash='dot'),
    fillcolor="rgba(0,0,0,0.1)"
)

# Update layout
subtitle = f"{' - ' + selected_batter if selected_batter != 'All' else ''}"
subtitle += f"{' vs ' + selected_hand if selected_hand != 'All' else ''}"
if 'All' not in selected_pitch_types:
    pitch_text = ', '.join(selected_pitch_types)
    subtitle += f" ({pitch_text})"

fig.update_layout(
    title=f'Bat Speed vs. League Average{subtitle}',
    xaxis_title='Horizontal Pitch Location (plate_x)',
    yaxis_title='Vertical Pitch Location (plate_z)',
    height=600,
    width=800,
    plot_bgcolor='rgba(240,240,240,0.95)'
)

# Display plot
st.plotly_chart(fig, use_container_width=True)

# Add some stats
st.subheader("Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Number of Swings", len(df_filtered))
    
with col2:
    avg_speed = df_filtered['bat_speed'].mean()
    st.metric("Average Bat Speed", f"{avg_speed:.1f} mph")
    
with col3:
    speed_vs_league = avg_speed - league_avg_speed
    st.metric("vs League Average", f"{speed_vs_league:+.1f} mph")