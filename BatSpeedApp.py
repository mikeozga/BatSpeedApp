import dash
from dash import dcc, html
from dash.dependencies import Input, Output
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

# Load and preprocess data
df = pd.read_csv("heatmap.csv")
df_cleaned = df.dropna(subset=['plate_x', 'plate_z', 'bat_speed', 'p_throws'])

# Map pitch codes to descriptions
df_cleaned['pitch_type_desc'] = df_cleaned['pitch_type'].map(PITCH_TYPES)

# Calculate league average bat speed (top 90% of swings)
bat_speed_threshold = np.percentile(df_cleaned['bat_speed'], 10)
filtered_speeds = df_cleaned[df_cleaned['bat_speed'] > bat_speed_threshold]['bat_speed']
league_avg_speed = filtered_speeds.mean()

app = dash.Dash(__name__)

# Prepare dropdown options
batters = ['All'] + sorted(df_cleaned['batter_name'].unique().tolist())
pitch_types = ['All'] + sorted(df_cleaned['pitch_type_desc'].dropna().unique().tolist())
pitcher_hands = ['All'] + sorted(df_cleaned['p_throws'].unique().tolist())

app.layout = html.Div([
    html.H1('Bat Speed by Pitch Location'),
    
    html.Div([
        html.Div([
            html.Label('Select Batter:'),
            dcc.Dropdown(
                id='batter-dropdown',
                options=[{'label': batter, 'value': batter} for batter in batters],
                value='All',
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '3%'}),
        
        html.Div([
            html.Label('Select Pitch Type(s):'),
            dcc.Dropdown(
                id='pitch-type-dropdown',
                options=[{'label': pitch, 'value': pitch} for pitch in pitch_types],
                value=['All'],
                multi=True,
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '3%'}),
        
        html.Div([
            html.Label('Pitcher Throws:'),
            dcc.Dropdown(
                id='pitcher-hand-dropdown',
                options=[{'label': hand, 'value': hand} for hand in pitcher_hands],
                value='All',
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block'})
    ]),
    
    dcc.Graph(id='bat-speed-heatmap')
])

@app.callback(
    Output('bat-speed-heatmap', 'figure'),
    [Input('batter-dropdown', 'value'),
     Input('pitch-type-dropdown', 'value'),
     Input('pitcher-hand-dropdown', 'value')]
)
def update_heatmap(selected_batter, selected_pitch_types, selected_hand):
    df_filtered = df_cleaned.copy()
    
    if selected_batter != 'All':
        df_filtered = df_filtered[df_filtered['batter_name'] == selected_batter]
    
    if 'All' not in selected_pitch_types:
        df_filtered = df_filtered[df_filtered['pitch_type_desc'].isin(selected_pitch_types)]
        
    if selected_hand != 'All':
        df_filtered = df_filtered[df_filtered['p_throws'] == selected_hand]
    
    x = np.linspace(-2, 2, 50)
    y = np.linspace(0, 5, 50)
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
        zmin=5,
        zmax=5,
        colorbar=dict(title='Bat Speed vs. League Avg (mph)'),
    ))
    
    fig.add_shape(
        type="rect",
        x0=-0.83, y0=1.5, x1=0.83, y1=3.5,
        line=dict(color="Black", width=2, dash='dot'),
        fillcolor="rgba(0,0,0,0.1)"
    )
    
    subtitle = f"{' - ' + selected_batter if selected_batter != 'All' else ''}"
    subtitle += f"{' vs ' + selected_hand if selected_hand != 'All' else ''}"
    if 'All' not in selected_pitch_types:
        pitch_text = ', '.join(selected_pitch_types)
        subtitle += f" ({pitch_text})"
    
    fig.update_layout(
        title=f'Bat Speed vs. League Average{subtitle}',
        xaxis_title='Horizontal Pitch Location',
        yaxis_title='Vertical Pitch Location',
        height=600,
        width=800,
        plot_bgcolor='rgba(240,240,240,0.95)'
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
