import streamlit as st
st.set_page_config(layout="wide")

st.markdown("""
    <style>
        MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 0rem; /* Adjust this value as needed */
        }
        
        h1:hover a, h2:hover a, h3:hover a, h4:hover a, h5:hover a, h6:hover a {
        display: none !important;
        }
        
    </style>
""", unsafe_allow_html=True)


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import seaborn as sns
import pandas as pd
import requests
import json
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset    
from pmdarima import auto_arima
import pmdarima as pm
from pandas import json_normalize
import plotly.graph_objects as go
from datetime import datetime
import os
from components.header import show_header

# 2. Show the constant header
show_header()

st.markdown("""
<style>
.stTabs:nth-of-type(1) [data-baseweb="tab"] {
    margin-right: 7px !important;
    font-size: 20px !important;
    color: #7a2e2b;
    background: #f5f5f5 !important;
    border-radius: 6px 6px 0 0 !important;
    transition: background 0.2s, color 0.2s;
}
.stTabs:nth-of-type(1) [data-baseweb="tab"]:hover {
    background: #e0e0e0 !important;
    color: #b71c1c !important;
}
.stTabs:nth-of-type(1) [data-baseweb="tab"][aria-selected="true"] {
    background: #827e7e !important;
    color: #e53935 !important;
    font-weight: bold !important;
}

/* 2. CLI_tabs (second stTabs on the page) */
.stTabs:nth-of-type(2) [data-baseweb="tab"] {
    margin-right: 7px !important;
    padding: 8px 10px !important;
    border-radius: 20px !important;
    background: #ffffff !important;
    color: #1565c0;
    font-size: 20px !important;
    font-weight: 500 !important;
    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
    cursor: pointer;
    border: none !important;
    box-shadow: none !important;
    border-bottom: none !important;
}
.stTabs:nth-of-type(2) [data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 16px !important;
}
/* Active (selected) tab: no underline */
.stTabs:nth-of-type(2) [data-baseweb="tab"][aria-selected="true"] {
    background: #e6e8eb !important; 
    color: #fc0004 !important;
    font-weight: bold !important;
}

/* 3. SE_tabs (fourth stTabs on the page, same as CLI_tabs) */
.stTabs:nth-of-type(3) [data-baseweb="tab"] {
    margin-right: 7px !important;
    padding: 8px 10px !important;
    border-radius: 20px !important;
    background: #ffffff !important;
    color: #1565c0;
    font-size: 20px !important;
    font-weight: 500 !important;
    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
    cursor: pointer;
    border: none !important;
    box-shadow: none !important;
    border-bottom: none !important;
}
.stTabs:nth-of-type(3) [data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 16px !important;
}
/* Active (selected) tab: no underline */
.stTabs:nth-of-type(3) [data-baseweb="tab"][aria-selected="true"] {
    background: #e6e8eb !important; 
    color: #fc0004 !important;
    font-weight: bold !important;
}

/* 4. SE_tabs (fourth stTabs on the page, same as CLI_tabs) */
.stTabs:nth-of-type(4) [data-baseweb="tab"] {
    margin-right: 7px !important;
    padding: 8px 10px !important;
    border-radius: 20px !important;
    background: #ffffff !important;
    color: #1565c0;
    font-size: 20px !important;
    font-weight: 500 !important;
    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
    cursor: pointer;
    border: none !important;
    box-shadow: none !important;
    border-bottom: none !important;
}
.stTabs:nth-of-type(4) [data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 16px !important;
}
/* Active (selected) tab: no underline */
.stTabs:nth-of-type(4) [data-baseweb="tab"][aria-selected="true"] {
    background: #e6e8eb !important; 
    color: #fc0004 !important;
    font-weight: bold !important;
}

/* Weather forecast card styling */
.forecast-card {{
    background: #62a1c7;
    color: #fff;
    border-radius: 10px;
    padding: 5px 5px 5px 5px;
    margin: 2px;
    text-align: center;
    min-width: 180px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    font-family: 'Segoe UI', sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
}}
.forecast-card .day {{
    background: #3a4d5c;
    border-radius: 2px;
    display: inline-block;
    font-size: 11px;
    margin-bottom: 2px;
    font-weight: bold;
}}
.forecast-card .date {{
    font-size: 12px;
    color: #050505;
    margin-bottom: 10px;
}}
.forecast-card .icon {{
    font-size: 40px;
}}
.forecast-card .desc {{
    font-size: 12px;
    margin-bottom: 2px;
    color: #e0e0e0;
}}
.forecast-card .temp {{
    font-size: 15px;
    margin-top: 2px;
    font-weight: bold;
}}
.forecast-summary-header {{
    font-size: 17.5px !important;
    color: #0a0a0a; 
}}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; font-size: 2.5em; font-weight: bold;'>ARGENTINA</div>",
    unsafe_allow_html=True
)


tabs = st.tabs([
    "Country Profile",
    "Climate Indicators",
    "Socio-economic Indicators",
    "Vulnerability Indicators",
    "Resilience Indicators",
    "Humanitarian Indicators"
])

# --- Climate Indicators Tab ---
with tabs[0]:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Mapa_Argentina_Tipos_clima_IGN.jpg/500px-Mapa_Argentina_Tipos_clima_IGN.jpg",
        caption="Mapa de los tipos de clima en Argentina", use_container_width=120 )

with tabs[1]:
    st.header("Climate Indicators")
    
    CLI_tabs = st.tabs([
    "Temperature and Precipitation",
    "Sea Level",
    "Hurricanes, Droughts and Floods",
    "Wildfires"])
    with CLI_tabs[0]:
        
        city_list = ["Bariloche", "Buenos Aires", "Cordoba", "El Calafate", "Iguazu", "Mar del Plata", "Mendoza", "Salta", "Trelew", "Ushuaia"]
        selected_city = st.selectbox("Select a city to view climate data:", city_list)

        # --- Load city list and data ---
        df = pd.read_csv('full_city_list.txt', sep=';')
        df_arg = df[df['Country'] == 'Argentina']
        df_matched = df_arg[df_arg['City'].isin(city_list)].copy()

        coords = {
            'Bariloche': (-41.1335, -71.3103), 'Buenos Aires': (-34.6037, -58.3816),
            'Cordoba': (-31.4201, -64.1888), 'El Calafate': (-50.3379, -72.2648),
            'Iguazu': (-25.5972, -54.5512), 'Mar del Plata': (-38.0055, -57.5575),
            'Mendoza': (-32.8908, -68.8272), 'Salta': (-24.7829, -65.4232),
            'Trelew': (-43.2532, -65.3090), 'Ushuaia': (-54.8019, -68.3030)
        }

        df_matched['lat'] = df_matched['City'].map(lambda x: coords[x][0])
        df_matched['lon'] = df_matched['City'].map(lambda x: coords[x][1])

        @st.cache_data(ttl=600)
        def fetch_all_weather(df):
            weather_data = {}
            for _, row in df.iterrows():
                city = row['City']
                city_id = int(row['CityId'])
                url = f'https://worldweather.wmo.int/en/json/{city_id}_en.json'
                try:
                    res = requests.get(url)
                    res.raise_for_status()
                    data = res.json()
                    weather_data[city] = data
                except Exception as e:
                    weather_data[city] = {"error": str(e)}
            return weather_data

        weather_dict = fetch_all_weather(df_matched)

        with open("ar.json", "r", encoding="utf-8") as f:
            city_geojson = json.load(f)

        city_map = folium.Map(location=[-38.4161, -63.6167], zoom_start=3.4)
        for feature in city_geojson["features"]:
            city_name = feature["properties"]["name"]
            gj = folium.GeoJson(
                data=feature,
                name=city_name,
                style_function=lambda f: {
                    'fillColor': '#9d9bc9',
                    'color': 'white',
                    'weight': 1,
                    'dashArray': '5, 5',
                    'fillOpacity': 0.7
                },
                highlight_function=lambda f: {
                    'fillColor': '#003366',
                    'color': 'white',
                    'weight': 2,
                    'fillOpacity': 0.9
                },
                tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Ciudad:'])
            )
            gj.add_child(folium.features.GeoJsonPopup(fields=["name"]))
            gj.add_to(city_map)

        # --- Split layout for map + charts ---
        left_col, right_col = st.columns([1,2])

        with left_col:
            map_data = st_folium(city_map, width=450, height=480, use_container_width=True)

        def get_city_weather_df(city_name):
            entry = weather_dict.get(city_name, None)
            if entry and 'city' in entry:
                return pd.json_normalize(entry)
            else:
                return pd.DataFrame()

        df = get_city_weather_df(selected_city)

        # --- Graphs: Monthly Climate Overview & Forecast ---
        if not df.empty:
            climate_cols = [
                'city.climate.raintype', 'city.climate.raindef', 'city.climate.rainunit',
                'city.climate.datab', 'city.climate.datae', 'city.climate.tempb', 'city.climate.tempe',
                'city.climate.rdayb', 'city.climate.rdaye', 'city.climate.rainfallb', 'city.climate.rainfalle',
                'city.climate.climatefromclino'
            ]
            changes = ['city.climate.climateMonth']
            df_base = df[climate_cols].copy().reset_index(drop=True)
            exploded = df[changes[0]].explode().reset_index(drop=True)
            norm = json_normalize(exploded)
            norm.columns = [f"{changes[0]}.{sub}" for sub in norm.columns]
            city_climate = pd.concat([df_base, norm], axis=1)

            city_climate['city.climate.climateMonth.month'] = pd.to_datetime(city_climate['city.climate.climateMonth.month'], format='%m').dt.strftime('%b')
            city_climate['city.climate.climateMonth.minTemp'] = pd.to_numeric(city_climate['city.climate.climateMonth.minTemp'])
            city_climate['city.climate.climateMonth.maxTemp'] = pd.to_numeric(city_climate['city.climate.climateMonth.maxTemp'])
            city_climate['city.climate.climateMonth.rainfall'] = pd.to_numeric(city_climate['city.climate.climateMonth.rainfall'])

            with right_col:
                st.markdown('<p class="forecast-summary-header">📊 Monthly Climate Overview</p>', unsafe_allow_html=True)

                months = city_climate['city.climate.climateMonth.month']
                min_temp = np.round(city_climate['city.climate.climateMonth.minTemp'], 2)
                max_temp = np.round(city_climate['city.climate.climateMonth.maxTemp'], 2)
                rainfall = np.round(city_climate['city.climate.climateMonth.rainfall'], 2)

                fig = go.Figure()

                # Min Temp Line
                fig.add_trace(go.Scatter(
                    x=months, y=min_temp, name='Min Temp (°C)', mode='lines+markers',
                    marker=dict(symbol='circle', size=8), line=dict(color='blue')
                ))

                # Max Temp Line
                fig.add_trace(go.Scatter(
                    x=months, y=max_temp, name='Max Temp (°C)', mode='lines+markers',
                    marker=dict(symbol='circle', size=8), line=dict(color='orange')
                ))

                # Rainfall Bar (secondary y-axis)
                fig.add_trace(go.Bar(
                    x=months, y=rainfall, name='Rainfall (mm)', yaxis='y2',
                    marker=dict(color='rgba(30, 144, 255, 0.3)')
                ))
                
                fig.update_xaxes(
                    showspikes=False,                   # <--- Hides vertical spikelines
                    color='black',
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')
                )
                fig.update_yaxes(
                    showspikes=False,                   # <--- Hides horizontal spikelines
                    color='black',
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')
                )
                
                fig.update_layout(
                    xaxis=dict(title='Month'),
                    yaxis=dict(title='Temperature (°C)'),
                    yaxis2=dict(title='Rainfall (mm)', overlaying='y', side='right'),
                    legend=dict(x=1.09, y=1, bordercolor='Black', borderwidth=1),
                    bargap=0.2,
                    height=500,
                    margin=dict(l=60, r=60, t=40, b=40),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- Forecast (Next 3 Months) ---
                st.markdown('<p class="forecast-summary-header">📉 Forecast (Next 3 Months)</p>', unsafe_allow_html=True)

                city_climate.index = pd.period_range(start='2025-01', periods=12, freq='M')
                forecast_periods = 3
                future_periods = pd.period_range(start='2026-01', periods=forecast_periods, freq='M')

                def forecast_series(series, periods=3):
                    from pmdarima import auto_arima
                    model = auto_arima(series, seasonal=False)
                    forecast = model.predict(n_periods=periods)
                    return np.round(forecast, 2)  # Round forecast to 2 decimals

                minTemp_forecast = forecast_series(city_climate['city.climate.climateMonth.minTemp'])
                maxTemp_forecast = forecast_series(city_climate['city.climate.climateMonth.maxTemp'])
                rainfall_forecast = forecast_series(city_climate['city.climate.climateMonth.rainfall'])

                # Combine periods for x-axis
                all_periods = list(city_climate.index) + list(future_periods)
                all_month_labels = [p.strftime('%b %Y') for p in all_periods]

                min_temp_full = list(np.round(city_climate['city.climate.climateMonth.minTemp'], 2)) + list(minTemp_forecast)
                max_temp_full = list(np.round(city_climate['city.climate.climateMonth.maxTemp'], 2)) + list(maxTemp_forecast)
                rainfall_full  = list(np.round(city_climate['city.climate.climateMonth.rainfall'], 2)) + list(rainfall_forecast)

                fig2 = go.Figure()

                # Actual Min Temp
                fig2.add_trace(go.Scatter(
                    x=all_month_labels[:12], y=min_temp_full[:12], name='Min Temp (°C)',
                    mode='lines+markers', marker=dict(symbol='circle', size=8), line=dict(color='blue')
                ))
                # Actual Max Temp
                fig2.add_trace(go.Scatter(
                    x=all_month_labels[:12], y=max_temp_full[:12], name='Max Temp (°C)',
                    mode='lines+markers', marker=dict(symbol='circle', size=8), line=dict(color='orange')
                ))
                # Forecast Min Temp
                fig2.add_trace(go.Scatter(
                    x=all_month_labels[12:], y=min_temp_full[12:], name='Min Temp - Forecast',
                    mode='lines+markers', line=dict(dash='dash', color='green'), marker=dict(symbol='square', size=7)
                ))
                # Forecast Max Temp
                fig2.add_trace(go.Scatter(
                    x=all_month_labels[12:], y=max_temp_full[12:], name='Max Temp - Forecast',
                    mode='lines+markers', line=dict(dash='dash', color='red'), marker=dict(symbol='square', size=7)
                ))
                # Actual Rainfall
                fig2.add_trace(go.Bar(
                    x=all_month_labels[:12], y=rainfall_full[:12], name='Rainfall - Actual',
                    yaxis='y2', marker=dict(color='rgba(242, 5, 13, 0.3)')
                ))
                # Forecast Rainfall
                fig2.add_trace(go.Bar(
                    x=all_month_labels[12:], y=rainfall_full[12:], name='Rainfall - Forecast',
                    yaxis='y2', marker=dict(color='rgba(0, 128, 0, 0.3)')
                ))

                fig2.update_layout(
                    xaxis=dict(title='Month'),
                    yaxis=dict(title='Temperature (°C)'),
                    yaxis2=dict(title='Rainfall (mm)', overlaying='y', side='right'),
                    legend=dict(x=1.09, y=1, bordercolor='Black', borderwidth=1),
                    bargap=0.2,
                    height=500,
                    margin=dict(l=60, r=60, t=40, b=40),
                    hovermode='x unified'
                )
                fig.update_xaxes(
                    showspikes=False,                   # <--- Hides vertical spikelines
                    color='black',
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')
                )
                fig.update_yaxes(
                    showspikes=False,                   # <--- Hides horizontal spikelines
                    color='black',
                    title_font=dict(color='black'),
                    tickfont=dict(color='black')
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            with left_col:
                # --- Weather Forecast Table (now as 6-day card UI) ---
                forecast_cols = ['city.forecast.issueDate', 'city.forecast.timeZone']
                nested_col = 'city.forecast.forecastDay'

                df_base = df[forecast_cols].copy().reset_index(drop=True)
                exploded = df[nested_col].apply(pd.Series).stack().reset_index(level=1, drop=True).reset_index(drop=True)
                forecast_normalized = json_normalize(exploded)
                forecast_normalized['ForecastDay'] = [f'Today+{i}' for i in range(len(forecast_normalized))]

                df_repeated = pd.concat([df_base.loc[i].repeat(7) for i in df_base.index], ignore_index=True)
                city_forecast = pd.concat([df_repeated.reset_index(drop=True), forecast_normalized], axis=1)
                city_forecast = city_forecast[['ForecastDay', 'forecastDate', 'wxdesc', 'weather', 'minTemp', 'maxTemp', 'minTempF', 'maxTempF', 'weatherIcon']]
                city_forecast = city_forecast.head(6)
                city_forecast['minTemp'] = pd.to_numeric(city_forecast['minTemp'])
                city_forecast['maxTemp'] = pd.to_numeric(city_forecast['maxTemp'])
                city_forecast['forecastDate'] = pd.to_datetime(city_forecast['forecastDate'])

                # --- Icon map for weather types ---
                icon_map = {
                    'Showers': '🌧️', 'Partly Cloudy': '⛅', 'Mostly Cloudy': '☁️', 'Sunny': '☀️',
                    'Cloudy': '☁️', 'Rain': '🌧️', 'Thunderstorms': '⛈️', 'Thundershowers': '⛈️',
                    'Sleet': '🌧️❄️', 'Snow': '❄️', 'Light Rain': '🌦️', 'Sunny Periods': '🌥️',
                    'Fine': '🌞', 'Mist': '🌫️', 'Storm': '⛈️'
                }

                st.markdown(
                    '<div class="forecast-summary-header">Weather forecast of <b>{}</b> (capital city)</div>'.format(selected_city),
                    unsafe_allow_html=True
                )

                num_cards = len(city_forecast)
                for i in range(0, num_cards, 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < num_cards:
                            row = city_forecast.iloc[i + j]
                            # Prefer emoji icon if mapped, else fallback to image URL, else default emoji
                            emoji_icon = icon_map.get(row['weather'], None)
                            if emoji_icon:
                                icon_html = emoji_icon
                            elif pd.notna(row['weatherIcon']) and str(row['weatherIcon']).startswith('http'):
                                icon_html = f'<img src="{row["weatherIcon"]}" width="38">'
                            else:
                                icon_html = "*"
                            day_name = row['forecastDate'].strftime('%a').upper()
                            date_str = row['forecastDate'].strftime('%d %b')
                            with cols[j]:
                                st.markdown(f"""
                                <div class="forecast-card">
                                    <div class="day">{day_name}</div>
                                    <div class="date">{date_str}</div>
                                    <div class="icon">{icon_html}</div>
                                    <div class="desc">{row['wxdesc']}</div>
                                    <div class="temp">{int(row['minTemp'])}°C | {int(row['maxTemp'])}°C</div>
                                </div>
                                """, unsafe_allow_html=True)

                st.markdown(
                    '<div class="forecast-summary-subtext">Issued at {} (Local Time) {}</div>'.format(
                        city_forecast.iloc[0]['forecastDate'].strftime('%I:%M %p'),
                        city_forecast.iloc[0]['forecastDate'].strftime('%b %d, %Y')
                    ),
                    unsafe_allow_html=True
                )
        else:
            st.warning(f"Weather data for {selected_city} is unavailable.")
        
    with CLI_tabs[1]:
        st.header("Sea level Dashboard")
        
    with CLI_tabs[2]:
        st.header("Droughts, Hurricanes and Floods data")
    
    with CLI_tabs[3]:
        st.header("Wildfires data")
        col1, col2, col3 = st.columns([1,1,1])
        
        
        with col1:
            image_folder = r'images/Climate Indicators/wildfires/number_of_fires_month'
            image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.png')]
            pro_names = [os.path.splitext(f)[0] for f in image_files]
            province = st.selectbox("Select a month to view:", pro_names, index=1)
            st.markdown("<br>", unsafe_allow_html=True)
            selected_image_path = os.path.join(image_folder, f"{province}.png")
            st.image(selected_image_path, caption=province, use_container_width=True)
            
        with col3:
            image_folder = r'images/Climate Indicators/wildfires/hectares_fires_province'
            image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.png')]
            pro_names = [os.path.splitext(f)[0] for f in image_files]
            province = st.selectbox("Select a Province to view fire impact :", pro_names, index=3)
            st.markdown("<br>", unsafe_allow_html=True)
            selected_image_path = os.path.join(image_folder, f"{province}.png")
            st.image(selected_image_path, caption=province, use_container_width=True)
            
        with col2:
            image_folder = r'images/Climate Indicators/wildfires/number_of_fires_jurisdicción'
            image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.png')]
            province_names = [os.path.splitext(f)[0] for f in image_files]
            province = st.selectbox("Select a Jurisdiction to view:", province_names, index=2)
            st.markdown("<br>", unsafe_allow_html=True)
            selected_image_path = os.path.join(image_folder, f"{province}.png")
            st.image(selected_image_path, caption=province, use_container_width=True)


with tabs[2]:
    st.header("Socio-economic Indicators")
    st.write("Information on demographics, economics, and social factors.")
    
    SE_tabs = st.tabs([
    "Poverty and Inequality",
    "Health and Sanitation",
    "Employment and Economy",
    "Education"])
    
    with SE_tabs[0]:
        st.write('Poverty and Inequality dashboard')
        
    with SE_tabs[1]:
        st.write('Health and Sanitation Dashboard')
        
    with SE_tabs[2]:
        st.write('Employment and Economy Dashboard')
        
    with SE_tabs[3]:
        st.write('Education dashboard')
        

with tabs[3]:
    st.header("Vulnerability Indicators")
    st.write("Details on exposure, sensitivity, and susceptibility to risks.")
    
    SE_tabs = st.tabs([
    "Age",
    "Gender",
    "Health",
    "Housing conditions and habitability"])
    
    with SE_tabs[0]:
        st.write('Age dashboard')
        st.image('images/Vulnerability Indicators/Age/AgeF.png', use_container_width=True)
        
    with SE_tabs[1]:
        st.write('Gender Dashboard')
        st.image('images/Vulnerability Indicators/Gender/SexRatioByAge.png', use_container_width=True)
        
    with SE_tabs[2]:
        st.write('Health Dashboard')
        st.image('images/Vulnerability Indicators/Health/LifeExpectancy.png', use_container_width=True)       

with tabs[4]:
    st.header("Resilience Indicators")
    st.write("Data measuring ability to prepare for, respond to, and recover.")

with tabs[5]:
    st.header("Humanitarian Indicators")
    st.write("Critical needs and response capacities for humanitarian aid.")

st.markdown('</div>', unsafe_allow_html=True)
