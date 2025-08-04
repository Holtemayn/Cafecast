import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ----------------------
# 1. Fetch 10-day weather forecast from Open-Meteo
# ----------------------
def fetch_weather(latitude=55.6761, longitude=12.5683):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max,sunshine_duration",
        "forecast_days": 10,
        "timezone": "Europe/Copenhagen"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data['daily'])
    df['time'] = pd.to_datetime(df['time'])
    df.rename(columns={
        'temperature_2m_max': 'temp_max',
        'precipitation_sum': 'precip_sum',
        'wind_speed_10m_max': 'wind_max',
        'sunshine_duration': 'sunshine_sec'
    }, inplace=True)
    
    # Convert sunshine to hours
    df['sunshine_hours'] = df['sunshine_sec'] / 3600
    df['month'] = df['time'].dt.month
    df['weekday'] = df['time'].dt.weekday + 1  # Monday=1
    df['year'] = df['time'].dt.year
    
    # Assign rain groups
    bins = [-0.1, 0.1, 1, 4, 10, 20, 1000]
    labels = ['0', '0.1-1', '1.1-4', '4.1-10', '10.1-20', '20+']
    df['rain_group'] = pd.cut(df['precip_sum'], bins=bins, labels=labels)
    
    return df

# ----------------------
# 2. Apply regression model manually
# ----------------------
def predict_revenue(df):
    # Coefficients from your model
    intercept = -6091.86
    
    revenue = intercept \
        + 1091.45 * df['temp_max'] \
        - 161.57 * df['wind_max'] \
        + 1050.98 * df['sunshine_hours']
    
    # Month adjustments
    month_coef = {
        2:0, 3:1921.83, 4:3218.33, 5:4395.99, 6:3272.20,
        7:-635.25, 8:1713.98, 9:-524.20, 10:-218.02, 11:5332.89, 12:7911.60
    }
    revenue += df['month'].map(month_coef).fillna(0)
    
    # Weekday adjustments
    weekday_coef = {
        1:0, 2:-675.05, 3:507.61, 4:3441.95, 5:3488.52, 6:3736.35, 7:2099.48
    }
    revenue += df['weekday'].map(weekday_coef).fillna(0)
    
    # Rain group adjustments
    rain_coef = {
        '0':0, '0.1-1':-2542.21, '1.1-4':-4584.56,
        '4.1-10':-5169.57, '10.1-20':-7182.38, '20+':-7340.26
    }
    revenue += df['rain_group'].astype(str).map(rain_coef).fillna(0)
    
    # Year adjustments
    year_coef = {2023:6727.87, 2024:10734.00, 2025:16610.22}
    revenue += df['year'].map(year_coef).fillna(0)
    
    df['predicted_revenue'] = revenue
    
    # Beregn predicted medarbejder timer
    df['predicted_medarbejder_timer'] = (df['predicted_revenue'] * 0.2) / 155
    
    return df[['time','temp_max','wind_max','precip_sum','sunshine_hours','predicted_revenue','predicted_medarbejder_timer']]

# ----------------------
# 3. Streamlit Dashboard
# ----------------------
st.title("☀️ Café Revenue Forecast Dashboard")

st.write("**Based on 10-day weather forecast and regression model**")

# Fetch and calculate
df_weather = fetch_weather()
df_forecast = predict_revenue(df_weather)

# Formatér time-kolonnen til kun dato
vis_df = df_forecast.copy()
vis_df['time'] = vis_df['time'].dt.strftime('%Y-%m-%d')

st.subheader("Forecast Table")
st.dataframe(
    vis_df.style.format({
        'temp_max': '{:.1f}',
        'wind_max': '{:.1f}',
        'precip_sum': '{:.1f}',
        'sunshine_hours': '{:.1f}',
        'predicted_revenue': '{:.0f} kr',
        'predicted_medarbejder_timer': '{:.1f} timer'
    }),
    hide_index=True
)

# Fjern eksisterende grafer og tilføj ét samlet søjlediagram
import plotly.graph_objects as go

# Normaliser værdier for at kunne sammenligne på samme akse
vis_df['revenue_norm'] = vis_df['predicted_revenue'] / vis_df['predicted_revenue'].max()
vis_df['sun_norm'] = vis_df['sunshine_hours'] / vis_df['sunshine_hours'].max()
vis_df['rain_norm'] = vis_df['precip_sum'] / vis_df['precip_sum'].max()

# Kombineret diagram med to y-akser
fig = go.Figure()

# Nedbør (søjle)
fig.add_trace(go.Bar(x=vis_df['time'], y=vis_df['precip_sum'], name='Nedbør (mm)', marker_color='royalblue', yaxis='y'))
# Solskinstimer (søjle)
fig.add_trace(go.Bar(x=vis_df['time'], y=vis_df['sunshine_hours'], name='Solskinstimer', marker_color='gold', yaxis='y'))
# Omsætning (linje)
fig.add_trace(go.Scatter(x=vis_df['time'], y=vis_df['predicted_revenue'], name='Omsætning', mode='lines+markers', marker_color='firebrick', yaxis='y2'))

fig.update_layout(
    barmode='group',
    title='Nedbør, Solskinstimer og Omsætning',
    xaxis_title='Dato',
    yaxis=dict(
        title='Nedbør (mm) / Solskinstimer',
        side='left',
        showgrid=False
    ),
    yaxis2=dict(
        title='Omsætning (kr)',
        overlaying='y',
        side='right',
        showgrid=False
    ),
    legend_title='Parameter',
    height=500
)
st.plotly_chart(fig, use_container_width=True)