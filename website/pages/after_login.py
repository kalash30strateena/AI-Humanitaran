# import streamlit as st
# st.set_page_config(layout="wide")
# from components.styles import apply_global_styles # type: ignore
# apply_global_styles()

import streamlit as st
from components.styles import apply_global_styles # type: ignore
apply_global_styles()


# Check login status
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.switch_page("pages/Login.py")
    st.stop()
# Check view indicators flag
if "from_view_indicators" not in st.session_state or not st.session_state["from_view_indicators"]:
    st.switch_page("pages/Login.py")
    st.stop()


import warnings
import plotly.express as px
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import seaborn as sns # type: ignore
import psycopg2
from pmdarima import auto_arima #type: ignore
from datetime import datetime
import pandas as pd
import itertools
import calendar
import requests
import re
import json
import numpy as np
import plotly.graph_objects as go
import folium # type: ignore
from streamlit_folium import st_folium # type: ignore
from sqlalchemy import create_engine # type: ignore
from pandas.tseries.offsets import DateOffset    
from pmdarima import auto_arima # type: ignore
import pmdarima as pm # type: ignore
from pandas import json_normalize
import altair as alt
import plotly.graph_objects as go
from datetime import datetime
import os
from docx import Document
from components.logged_header import logged_header # type: ignore
import urllib.parse
from sqlalchemy import create_engine # type: ignore
import pandas as pd
from sqlalchemy import text
from sqlalchemy import bindparam

# 2. Show the constant header
logged_header()

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


DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres.ajbcqqwgdmfscvmkbtqz',
    'password':'StrateenaAIML',
    'host': 'aws-0-ap-south-1.pooler.supabase.com',
    'port': 6543
}
def get_engine():
    password = urllib.parse.quote_plus(DB_CONFIG['password'])
    url = (
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{password}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    )
    return create_engine(url)

tabs = st.tabs([
    "Country Profile",
    "Climate Indicators",
    "Socio-economic Indicators",
    "Vulnerability Indicators",
    "Resilience Indicators",
    "Humanitarian Indicators"
])
engine=get_engine()

icon_map = {
                'Showers': '🌧️', 'Partly Cloudy': '⛅', 'Mostly Cloudy': '☁️', 'Sunny': '☀️',
                'Cloudy': '☁️', 'Rain': '🌧️', 'Thunderstorms': '⛈️', 'Thundershowers': '⛈️',
                'Sleet': '🌧️❄️', 'Snow': '❄️', 'Light Rain': '🌦️', 'Sunny Periods': '🌥️',
                'Fine': '🌞', 'Mist': '🌫️', 'Storm': '⛈️'
            }
# --- Climate Indicators Tab ---

with tabs[0]:
    left_col , right_col = st.columns([4,5])
    with left_col:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Mapa_Argentina_Tipos_clima_IGN.jpg/500px-Mapa_Argentina_Tipos_clima_IGN.jpg",
        caption="Mapa de los tipos de clima en Argentina", use_container_width=160 )
    with right_col:
        st.write(" ")
        docx_path = "docs/country_profile_data.docx"  # Your file path
        def docx_to_markdown(doc):
            md_lines = []
            for para in doc.paragraphs:
                md_para = ""
                for run in para.runs:
                    text = run.text.replace('\n', ' ')
                    if not text:
                        continue
                    # Apply formatting
                    if run.bold:
                        text = f"**{text}**"
                    if run.italic:
                        text = f"*{text}*"
                    if run.underline:
                        text = f"__{text}__"
                    md_para += text
                md_lines.append(md_para)
            return "\n".join(md_lines)

        if os.path.exists(docx_path):
            doc = Document(docx_path)
            markdown_content = docx_to_markdown(doc)
            st.markdown(markdown_content)
        else:
            st.error(f"File not found: {docx_path}")        
        
with tabs[1]:
    st.header("Climate Indicators")
    
    CLI_tabs = st.tabs([
    "Temperature",
    "Precipitation",
    "Sea Level",
    "Hurricanes, Droughts and Floods",
    "Wildfires"])
    
    with CLI_tabs[0]:

        left_col, right_col = st.columns([4,5])
        with left_col:
            def get_station_df():
                conn = psycopg2.connect(**DB_CONFIG)
                query = """
                    SELECT station_id, uid, station, province, latitude, longitude, altitude
                    FROM stations
                """
                df = pd.read_sql_query(query, conn)
                conn.close()
                return df

            station_df = get_station_df()

            # --- Province selection dropdown ---
            province_list = sorted(station_df["province"].dropna().unique())
            selected_province = st.selectbox("Filter stations by province:", ["All"] + province_list, key="province_select")

            # --- Filter DataFrame and set map center/zoom ---
            if selected_province == "All":
                filtered_df = station_df
                map_center = [-38.4161, -63.6167]  # Center of Argentina
                zoom_level = 4
            else:
                filtered_df = station_df[station_df["province"] == selected_province]
                # If there are stations, center on their mean location, else fallback to default
                if not filtered_df.empty:
                    map_center = [
                        filtered_df["latitude"].mean(),
                        filtered_df["longitude"].mean()
                    ]
                    zoom_level = 5  # More zoomed-in for province
                else:
                    map_center = [-38.4161, -63.6167]
                    zoom_level = 4

            # --- Create Folium map and add filtered station markers ---
            m = folium.Map(location=map_center, zoom_start=zoom_level)
            for _, row in filtered_df.iterrows():
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    tooltip=f"{row['station']} ({row['station_id']})",
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(m)

            st_map = st_folium(m, width=450, height=480, use_container_width=True)
            
        with right_col:
            st.write("")
            st.write("")
            # --- AUTHENTICATE AND FETCH STATION DATA ---
            auth_url = "https://api-test.smn.gob.ar/v1/api-token/auth"
            auth_data = {
                "username": "cruzroja",
                "password": "TCpqNb7b"
            }
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            auth_response = requests.post(auth_url, headers=headers, json=auth_data)

            # --- GET CLICKED STATION ID ---
            clicked_station_id = None
            if st_map and st_map.get("last_object_clicked"):
                lat = st_map["last_object_clicked"]["lat"]
                lng = st_map["last_object_clicked"]["lng"]
                tolerance = 1e-5
                match = station_df[
                    (abs(station_df["latitude"] - lat) < tolerance) &
                    (abs(station_df["longitude"] - lng) < tolerance)
                ]
                if not match.empty:
                    clicked_station_id = str(int(match.iloc[0]["station_id"])).strip()

            # --- SHOW STATION WEATHER CARD ---
            if auth_response.status_code == 200 and clicked_station_id:
                token = auth_response.json().get("token", "").strip()
                stations_url = f"https://api-test.smn.gob.ar/v1/weather/station/{clicked_station_id}"
                headers_with_auth = {
                    "Authorization": f"JWT {token}",
                    "Accept": "application/json"
                }
                stations_response = requests.get(stations_url, headers=headers_with_auth)
                if stations_response.status_code == 200:
                    stations_data = stations_response.json()
                    # Get the weather icon if available
                    weather_desc = stations_data['weather'].get('description', '')

                    # Format the date
                    from datetime import datetime
                    try:
                        date_obj = datetime.fromisoformat(stations_data['date'])
                        date_str = date_obj.strftime('%A, %d %b %Y, %I:%M %p')
                    except Exception:
                        date_str = stations_data['date']

                    # Card UI
                    st.markdown(
                        f"""
                        <div style="
                            background: #b9d9fa; 
                            border-radius: 16px; 
                            padding: 10px 10px; 
                            width: 100%; 
                            max-width: 100vw;
                            margin: 0 auto;
                            display: flex; 
                            flex-direction: column;
                            gap: 10px;
                        ">
                            <!-- First row: weather_desc and date_str -->
                            <div style="display: flex; flex-direction: row; gap: 22px; align-items: center;">
                                <div style="font-size: 1em; font-weight: bold;">{weather_desc}</div>
                                <div style="color: #000305; font-size: 1em;">{date_str}</div>
                            </div>
                            <!-- Second row: all other details -->
                            <div style="display: flex; flex-direction: row; gap: 22px; flex-wrap: wrap;">
                                <div><b>Temperature:</b> {stations_data['temperature']}°C</div>
                                <div><b>Humidity:</b> {stations_data['humidity']}%</div>
                                <div><b>Pressure:</b> {stations_data['pressure']} hPa</div>
                                <div><b>Visibility:</b> {stations_data['visibility']} km</div>
                                <div><b>Wind:</b> {stations_data['wind']['speed']} km/h {stations_data['wind']['direction']} ({stations_data['wind']['deg']}°)</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    st.info("Failed to fetch station weather data.")
            elif not clicked_station_id:
                st.info("Click on a station marker to view its climate data.")
            else:
                st.info("Failed to authenticate for station data.")

            # --- CLIMATE DATA PLOT ---
            def get_climate_data(station_id):
                conn = psycopg2.connect(**DB_CONFIG)
                query = """
                    SELECT month, max_temp, min_temp, avg_pressure
                    FROM climate_months
                    WHERE station = %s
                    ORDER BY id
                """
                df = pd.read_sql_query(query, conn, params=(station_id,))
                conn.close()
                return df

            if clicked_station_id:
                climate_df = get_climate_data(clicked_station_id)
                if not climate_df.empty:
                    fig = go.Figure()

                    # max_temp
                    fig.add_trace(go.Scatter(
                        x=climate_df["month"],
                        y=climate_df["max_temp"],
                        mode="lines+markers",
                        name="Max Temp",
                        line=dict(color="red"),
                        marker=dict(symbol="circle", size=8, color="red"),
                        yaxis="y1"
                    ))

                    # min_temp
                    fig.add_trace(go.Scatter(
                        x=climate_df["month"],
                        y=climate_df["min_temp"],
                        mode="lines+markers",
                        name="Min Temp",
                        line=dict(color="orange"),
                        marker=dict(symbol="circle", size=8, color="orange"),
                        yaxis="y1"
                    ))

                    # avg_pressure
                    fig.add_trace(go.Scatter(
                        x=climate_df["month"],
                        y=climate_df["avg_pressure"],
                        mode="lines+markers",
                        name="Avg Pressure",
                        line=dict(color="grey"),
                        marker=dict(symbol="circle", size=8, color="grey"),
                        yaxis="y2"
                    ))

                    # Layout
                    fig.update_layout(
                        title=f"Climate Data for Station {clicked_station_id}",
                        xaxis=dict(title="Month"),
                        yaxis=dict(title="Temperature (°C)"),
                        yaxis2=dict(
                            title="Pressure (hPa)",
                            overlaying="y",
                            side="right"
                        ),
                        legend=dict(title="Variable"),
                        width=700,
                        height=400,
                        margin=dict(l=40, r=40, t=60, b=40)
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No climate data available for this station.")


        with left_col:
            
            # --- City list ---
            city_list = ["Bariloche", "Buenos Aires", "Cordoba", "El Calafate", "Iguazu", "Mar del Plata", "Mendoza", "Salta", "Trelew", "Ushuaia"]

            # --- Load city geojson ---
            with open("complete_map.geojson", "r", encoding="utf-8") as f:
                city_geojson = json.load(f)

            # --- Dropdown at the top (controls everything) ---
            selected_city = st.selectbox(
                "Select a Province to view the Forecast data:",
                city_list,
                key="city_select"
            )
            
            # --- Weather Forecast Table (6-day card UI) ---
            query = """
                    SELECT *
                    FROM forecast_days
                    WHERE city_name = %s
                    ORDER BY "forecastdate"
                    LIMIT 6
                """

            df_forecast = pd.read_sql_query(query, engine, params=(selected_city,))

            # --- Clean up types ---
            if not df_forecast.empty:
                df_forecast['forecastdate'] = pd.to_datetime(df_forecast['forecastdate'], format='%Y-%m-%d')
                df_forecast['minTemp'] = pd.to_numeric(df_forecast['minTemp'])
                df_forecast['maxTemp'] = pd.to_numeric(df_forecast['maxTemp'])

            # --- Forecast card UI ---
            if not df_forecast.empty:
                st.markdown(
                    f'<div class="forecast-summary-header">Weather forecast of <b>{selected_city}</b> (capital city)</div>',
                    unsafe_allow_html=True
                )
                num_cards = len(df_forecast)
                for i in range(0, num_cards, 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < num_cards:
                            row = df_forecast.iloc[i + j]
                            emoji_icon = icon_map.get(row['weather'], None)
                            if emoji_icon:
                                icon_html = emoji_icon
                            elif pd.notna(row['weatherIcon']) and str(row['weatherIcon']).startswith('http'):
                                icon_html = f'<img src="{row["weatherIcon"]}" width="38">'
                            else:
                                icon_html = "*"
                            day_name = row['forecastdate'].strftime('%a').upper()
                            date_str = row['forecastdate'].strftime('%d %b')
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
                        df_forecast.iloc[0]['forecastdate'].strftime('%I:%M %p'),
                        df_forecast.iloc[0]['forecastdate'].strftime('%b %d, %Y')
                    ),
                    unsafe_allow_html=True
                )
            else:
                st.warning("No forecast data available for the selected city.")
    
    with CLI_tabs[1]:
        st.write('Precipitation dashboard')
        
    with CLI_tabs[2]:
        st.write('Sea Level data')
        
    with CLI_tabs[3]:
        st.header("Droughts, Hurricanes and Floods data")
        
        # HYDROLOGICAL PLOTTING CODE
        
        @st.cache_data
        def get_hydro_data():
            conn = psycopg2.connect(**DB_CONFIG)
            query = """
            SELECT hydrological_station_name, daily_date, value_index, historical_forecast
            FROM hydrological_droughts
            ORDER BY hydrological_station_name, daily_date
            """
            df = pd.read_sql(query, conn)
            conn.close()
            df['daily_date'] = pd.to_datetime(df['daily_date'], format='%d-%m-%Y')
            return df

        @st.cache_data
        def get_hydro_forecast(city, df, forecast_days=8):
            city_df = df[df['hydrological_station_name'] == city].sort_values('daily_date')
            ts = city_df.set_index('daily_date')['value_index'].astype(float)
            hist_forecast = city_df.set_index('daily_date')['historical_forecast'].astype(float)
            if len(ts) < 14:
                return None, None, None
            model_arima = auto_arima(
                ts,
                start_p=1, start_q=1,
                max_p=5, max_q=5,
                d=None,
                seasonal=False,
                trace=False,
                suppress_warnings=True,
                stepwise=True
            )
            forecast = model_arima.predict(n_periods=forecast_days)
            last_date = ts.index[-1]
            forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
            forecast_series = pd.Series(forecast, index=forecast_dates)
            return ts, hist_forecast, forecast_series

        # --- MAIN APP ---
        st.title("Hydrological Drought Forecast")

        hydro_df = get_hydro_data()
        hydro_cities = hydro_df['hydrological_station_name'].unique()

        hydro_city = st.selectbox("Select a station", hydro_cities, key="hydro_city")

        if hydro_city:
            ts, hist_forecast, forecast_series = get_hydro_forecast(hydro_city, hydro_df)
            if ts is None:
                st.warning(f"Not enough data for {hydro_city} (need at least 14 records).")
            else:
                fig = go.Figure()
                # Actual data (last year, ~360 days, or adjust as needed)
                fig.add_trace(go.Scatter(
                    x=ts.index[-360:], y=ts.values[-360:], mode='lines+markers',
                    name='Actual Data', line=dict(color='#62c0d1')
                ))
                # Historical forecasts (all available)
                fig.add_trace(go.Scatter(
                    x=hist_forecast.index,
                    y=hist_forecast.values,
                    mode='lines',
                    name='Historical Forecasts',
                    line=dict(color='#82070d', width=1, dash='dash')
                ))

                # 8-day forecast
                fig.add_trace(go.Scatter(
                    x=forecast_series.index, y=forecast_series, mode='lines+markers',
                    name='8-Day Forecast', line=dict(color='orange', dash='dash')
                ))
                fig.update_layout(
                    title=f'Forecast for {hydro_city}',
                    xaxis_title='Date',
                    yaxis_title='Drought Index Value',
                    legend=dict(x=0, y=1),
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # METEROLOGICAL PLOTTING CODE
    
        @st.cache_data
        def get_metero_data():
            conn = psycopg2.connect(**DB_CONFIG)
            query = """
            SELECT meterological_station_name, monthly_date, value_index, historical_forecast
            FROM meterological_droughts
            ORDER BY meterological_station_name, monthly_date
            """
            df = pd.read_sql(query, conn)
            conn.close()
            df['monthly_date'] = pd.to_datetime(df['monthly_date'], format='%d-%m-%Y')
            return df

        @st.cache_data
        def get_metero_forecast(city, df, forecast_days=2):
            city_df = df[df['meterological_station_name'] == city].sort_values('monthly_date')
            ts = city_df.set_index('monthly_date')['value_index'].astype(float)
            hist_forecast = city_df.set_index('monthly_date')['historical_forecast'].astype(float)
            if len(ts) < 14:
                return None, None, None
            model_arima = auto_arima(
                ts,
                start_p=1, start_q=1,
                max_p=5, max_q=5,
                d=None,
                seasonal=False,
                trace=False,
                suppress_warnings=True,
                stepwise=True
            )
            forecast = model_arima.predict(n_periods=forecast_days)
            forecast = np.array(forecast)
            last_date = ts.index[-1]
            forecast_dates = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_days + 1)]
            forecast_series = pd.Series(forecast, index=forecast_dates)
            return ts, hist_forecast, forecast_series

        # --- MAIN APP ---
        st.title("Meterological Drought Forecast")

        metero_df = get_metero_data()
        metero_cities = metero_df['meterological_station_name'].unique()

        metero_city = st.selectbox("Select a station", metero_cities, key="metero_city")

        if metero_city:
            ts, hist_forecast, forecast_series = get_metero_forecast(metero_city, metero_df)
            if ts is None:
                st.warning(f"Not enough data for {metero_city} (need at least 14 records).")
            else:
                fig = go.Figure()
                # Actual data (last year, ~360 days, or adjust as needed)
                fig.add_trace(go.Scatter(
                    x=ts.index[-36:], y=ts.values[-36:], mode='lines+markers',
                    name='Actual Data', line=dict(color='#62c0d1')
                ))
                # Historical forecasts (all available)
                fig.add_trace(go.Scatter(
                    x=hist_forecast.index,
                    y=hist_forecast.values,
                    mode='lines',
                    name='Historical Forecasts',
                    line=dict(color='#82070d', width=1, dash='dash')
                ))

                # 3-month forecast
                fig.add_trace(go.Scatter(
                    x=forecast_series.index, y=forecast_series, mode='lines+markers',
                    name='2-Month Forecast', line=dict(color='orange', dash='dash')
                ))
                fig.update_layout(
                    title=f'Forecast for {metero_city}',
                    xaxis_title='Date',
                    yaxis_title='Drought Index Value',
                    legend=dict(x=1, y=0.5),
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
        
    with CLI_tabs[4]:
        st.header("Wildfires data")
        col1, col2 = st.columns([1,1])
        
        with col1:            
            
            @st.cache_data
            def load_regions():
                conn = psycopg2.connect(**DB_CONFIG)
                df = pd.read_sql("SELECT region_id, region_name FROM regions ORDER BY region_name", conn)
                conn.close()
                return df

            @st.cache_data
            def load_fires_by_region(region_id):
                conn = psycopg2.connect(**DB_CONFIG)
                query = """
                    SELECT f.year, r.region_name, f.fire_count
                    FROM fires_by_region f
                    JOIN regions r ON f.region_id = r.region_id
                    WHERE f.region_id = %s
                    ORDER BY f.year
                """
                df = pd.read_sql(query, conn, params=(region_id,))
                conn.close()
                return df

            regions_df = load_regions()
            region_names = regions_df['region_name'].tolist()
            selected_region_name = st.selectbox("Select a region to view fire alerts per year:", region_names)
            selected_region_id = int(regions_df.loc[regions_df['region_name'] == selected_region_name, 'region_id'].iloc[0])  # Convert to native int
            df_selected = load_fires_by_region(selected_region_id)

            years = df_selected['year'].to_numpy()
            values = df_selected['fire_count'].to_numpy()

            if len(values) > 2:
                model = auto_arima(values, seasonal=False, suppress_warnings=True)
                forecast = model.predict(n_periods=1)
                forecast_years = [years[-1] + i for i in range(1, 4)]
            else:
                forecast = []
                forecast_years = []

            all_years = list(years) + list(forecast_years)
            all_values = list(values) + list(forecast)

            fig = go.Figure()

            # Actual values: solid red line
            fig.add_trace(go.Scatter(
                x=years, y=values,
                mode='lines+markers',
                name='Actual Fire Count',
                line=dict(color='red', width=2)
            ))

            # Forecast values: solid black line
            if forecast_years:
                fig.add_trace(go.Scatter(
                    x=forecast_years, y=forecast,
                    mode='lines+markers',
                    name='Forecasted Fire Count',
                    line=dict(color='black', width=2)
                ))
                # Connecting line between last actual and first forecast (dashed)
                fig.add_trace(go.Scatter(
                    x=[years[-1], forecast_years[0]],
                    y=[values[-1], forecast[0]],
                    mode='lines',
                    name='Forecast Transition',
                    line=dict(color='black', width=2, dash='dash'),
                    showlegend=False
                ))

            fig.update_layout(
                title=f"Fires by Month - {selected_region_name}",
                xaxis_title="Year",
                yaxis_title="Fire Count",
                legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
            )

            st.plotly_chart(fig, use_container_width=True)
            
        with col2:

            @st.cache_data
            def load_months():
                conn = psycopg2.connect(**DB_CONFIG)
                df = pd.read_sql("SELECT DISTINCT month FROM fires_by_month ORDER BY month", conn)
                conn.close()
                return df['month'].tolist()

            @st.cache_data
            def load_fires_by_month(selected_month):
                conn = psycopg2.connect(**DB_CONFIG)
                query = """
                    SELECT year, month, fire_count
                    FROM fires_by_month
                    WHERE month = %s
                    ORDER BY year
                """
                df = pd.read_sql(query, conn, params=(selected_month,))
                conn.close()
                return df
            
            months = load_months()  # This should return a list of integers
            month_options = [calendar.month_abbr[m] for m in months]
            selected_month_name = st.selectbox("Select a month to view yearly fire alerts:", month_options)
            selected_month = months[month_options.index(selected_month_name)]

            # Load data for selected month
            df_selected = load_fires_by_month(selected_month)
            years = df_selected['year'].to_numpy()
            values = df_selected['fire_count'].to_numpy()

            # Forecast next 3 years
            if len(values) > 2:
                model = auto_arima(values, seasonal=False, suppress_warnings=True)
                forecast = model.predict(n_periods=1)
                forecast_years = [years[-1] + i for i in range(1, 4)]
            else:
                forecast = []
                forecast_years = []

            fig = go.Figure()

            # Historical: royal blue line
            fig.add_trace(go.Scatter(
                x=years, y=values,
                mode='lines+markers',
                name='Actual Fire Count',
                line=dict(color='royalblue', width=2)
            ))

            # Forecast: black line
            if forecast_years:
                fig.add_trace(go.Scatter(
                    x=forecast_years, y=forecast,
                    mode='lines+markers',
                    name='Forecasted Fire Count',
                    line=dict(color='black', width=2)
                ))
                # Dashed connecting line
                fig.add_trace(go.Scatter(
                    x=[years[-1], forecast_years[0]],
                    y=[values[-1], forecast[0]],
                    mode='lines',
                    name='Forecast Transition',
                    line=dict(color='black', width=2, dash='dash'),
                    showlegend=False
                ))

            fig.update_layout(
                title=f"Fires by Month - {selected_month}",
                xaxis_title="Year",
                yaxis_title="Fire Count",
                legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
            )

            st.plotly_chart(fig, use_container_width=True)

        
        with col1:
            @st.cache_data
            def load_provinces():
                conn = psycopg2.connect(**DB_CONFIG)
                df = pd.read_sql("SELECT province_id, province_name FROM provinces ORDER BY province_name", conn)
                conn.close()
                return df

            @st.cache_data
            def load_fires_by_province(province_id):
                conn = psycopg2.connect(**DB_CONFIG)
                query = """
                    SELECT f.year, p.province_name, f.fires_count
                    FROM fires_by_province f
                    JOIN provinces p ON f.province_id = p.province_id
                    WHERE f.province_id = %s
                    ORDER BY f.year
                """
                df = pd.read_sql(query, conn, params=(province_id,))
                conn.close()
                return df

            # Province selection
            provinces_df = load_provinces()
            province_names = provinces_df['province_name'].tolist()
            selected_province_name = st.selectbox("Select a Province to view amount of fire alerts:", province_names)
            selected_province_id = int(provinces_df.loc[provinces_df['province_name'] == selected_province_name, 'province_id'].iloc[0])

            # Data for selected province
            df_selected = load_fires_by_province(selected_province_id)
            years = df_selected['year'].to_numpy()
            values = df_selected['fires_count'].to_numpy()

            # Forecast next 3 years
            if len(values) > 2:
                model = auto_arima(values, seasonal=False, suppress_warnings=True)
                forecast = model.predict(n_periods=1)
                forecast_years = [years[-1] + i for i in range(1, 4)]
            else:
                forecast = []
                forecast_years = []

            fig = go.Figure()

            # Historical: seagreen line
            fig.add_trace(go.Scatter(
                x=years, y=values,
                mode='lines+markers',
                name='Actual Fire Count',
                line=dict(color='seagreen', width=2)
            ))

            # Forecast: black line
            if forecast_years:
                fig.add_trace(go.Scatter(
                    x=forecast_years, y=forecast,
                    mode='lines+markers',
                    name='Forecasted Fire Count',
                    line=dict(color='black', width=2)
                ))
                # Dashed connecting line
                fig.add_trace(go.Scatter(
                    x=[years[-1], forecast_years[0]],
                    y=[values[-1], forecast[0]],
                    mode='lines',
                    name='Forecast Transition',
                    line=dict(color='black', width=2, dash='dash'),
                    showlegend=False
                ))

            fig.update_layout(
                title=f"Fires by Province - {selected_province_name}",
                xaxis_title="Year",
                yaxis_title="Fire Count",
                legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
            )

            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            @st.cache_data
            def load_provinces():
                conn = psycopg2.connect(**DB_CONFIG)
                df = pd.read_sql("SELECT province_id, province_name FROM provinces ORDER BY province_name", conn)
                conn.close()
                return df

            @st.cache_data
            def load_hectares_by_province(province_id):
                conn = psycopg2.connect(**DB_CONFIG)
                query = """
                    SELECT f.year, p.province_name, f.hectares_damage
                    FROM fires_by_province f
                    JOIN provinces p ON f.province_id = p.province_id
                    WHERE f.province_id = %s
                    ORDER BY f.year
                """
                df = pd.read_sql(query, conn, params=(province_id,))
                conn.close()
                return df

            # Province selection
            provinces_df = load_provinces()
            province_names = provinces_df['province_name'].tolist()
            selected_province_name = st.selectbox("Select a Province to view hectares damaged:", province_names)
            selected_province_id = int(provinces_df.loc[provinces_df['province_name'] == selected_province_name, 'province_id'].iloc[0])

            # Data for selected province
            df_selected = load_hectares_by_province(selected_province_id)
            years = df_selected['year'].to_numpy()
            values = df_selected['hectares_damage'].to_numpy()

            # Forecast next 3 years
            if len(values) > 2:
                model = auto_arima(values, seasonal=False, suppress_warnings=True)
                forecast = model.predict(n_periods=1)
                forecast_years = [years[-1] + i for i in range(1, 4)]
            else:
                forecast = []
                forecast_years = []

            fig = go.Figure()

            # Historical: seagreen line
            fig.add_trace(go.Scatter(
                x=years, y=values,
                mode='lines+markers',
                name='Actual Hectares Damaged',
                line=dict(color='purple', width=2)
            ))

            # Forecast: black line
            if forecast_years:
                fig.add_trace(go.Scatter(
                    x=forecast_years, y=forecast,
                    mode='lines+markers',
                    name='Forecasted Hectares Damaged',
                    line=dict(color='black', width=2)
                ))
                # Dashed connecting line
                fig.add_trace(go.Scatter(
                    x=[years[-1], forecast_years[0]],
                    y=[values[-1], forecast[0]],
                    mode='lines',
                    name='Forecast Transition',
                    line=dict(color='black', width=2, dash='dash'),
                    showlegend=False
                ))

            fig.update_layout(
                title=f"Hectares Damaged by Province (Yearly Total) - {selected_province_name}",
                xaxis_title="Year",
                yaxis_title="Hectares Damaged",
                legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
            )

            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header("Socio-economic Indicators")
    st.write("Information on demographics, economics, and social factors.")
    
    SE_tabs = st.tabs([
    "Poverty and Inequality",
    "Migration",
    "Employment and Economy",
    "Education"])
    
    with SE_tabs[0]:
        
        indicator_groups = {
            "Income Share Distribution": [
                "Income Share of First Quintile", "Income Share of Second Quintile",
                "Income Share of Third Quintile", "Income Share of Fourth Quintile",
                "Income Share of Fifth Quintile", "Income share held by second 20%",
                "Income share held by third 20%", "Income share held by fourth 20%",
                "Income share held by highest 20%", "Income share held by highest 10%"],
            "Labor-Linked Poverty": ["Labor Income Poverty Index"],
            "Middle Class & Vulnerable Groups": [
                "Middle Class ($10-50 a day) Headcount", "Vulnerable ($4-10 a day) Headcount"],
            "Extreme & Moderate Poverty": [
                "Poverty Gap ($1.90 a day)", "Poverty Gap ($2.50 a day)", "Poverty Gap ($4 a day)",
                "Poverty Headcount ($1.90 a day)", "Poverty Headcount ($2.50 a day)", "Poverty Headcount ($4 a day)",
                "Poverty Severity ($1.90 a day)", "Poverty Severity ($2.50 a day)", "Poverty Severity ($4 a day)",
                "Poverty headcount ratio at national poverty lines (% of population)",
                "Proportion of people living below 50 percent of median income (%)"],
            "Poverty Gap - PPP": [
                "Official Moderate Poverty Rate-Urban", "Poverty gap at $2.15 a day (2017 PPP) (%)",
                "Poverty gap at $3.65 a day (2017 PPP) (%)", "Poverty gap at $6.85 a day (2017 PPP) (%)"]
            }
        forecast_indicators = [
            "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)",
            "Poverty headcount ratio at $3.65 a day (2017 PPP) (% of population)",
            "Poverty headcount ratio at $6.85 a day (2017 PPP) (% of population)"]
        inequality_options = {
            "Income": [
                "Atkinson, A(.5)", "Atkinson, A(1)", "Atkinson, A(2)",
                "Generalized Entrophy, GE(-1)", "Generalized Entrophy, GE(2)", "Gini Coefficient",
                "Gini Coefficient (No Zero Income)", "Gini index", "Gini, Urban",
                "Mean Log Deviation, GE(0)", "Mean Log Deviation, GE(0), Urban",
                "Rate 75/25", "Rate 90/10", "Theil Index, GE(1)", "Theil Index, GE(1), Urban"],
            "Human Development" :['Coefficient of human inequality', 'Inequality in eduation', 'Inequality in income', 'Inequality in life expectancy',]
            }

        @st.cache_data
        def get_filtered_data(indicator_list, category):
            conn = psycopg2.connect(**DB_CONFIG)
            table_name = "poverty" if category != "Human Development" else "human_development"
            # Handle single and multiple indicators correctly
            if len(indicator_list) == 1:
                query = f"""
                SELECT year, indicator_name, value 
                FROM {table_name}
                WHERE indicator_name = '{indicator_list[0]}'
                ORDER BY year;
                """
            else:
                query = f"""
                SELECT year, indicator_name, value 
                FROM {table_name}
                WHERE indicator_name IN {tuple(indicator_list)}
                ORDER BY year;
                """
            df = pd.read_sql(query, conn)
            conn.close()
            
            df["year"] = pd.to_datetime(df["year"], format='%Y')
            df.set_index("year", inplace=True)
            df = df.pivot(columns="indicator_name", values="value").sort_index()
            #df.fillna(method="ffill", inplace=True)
            return df

        st.subheader("Poverty Data")
        selected_category = st.selectbox("Select a Poverty Category", list(indicator_groups.keys()))
        selected_indicators = indicator_groups[selected_category]
        df = get_filtered_data(selected_indicators, selected_category)

        def forecast_and_plot(df):
            forecast_df = pd.DataFrame()
            future_years = pd.date_range(start=str(df.index[-1].year + 1), periods=5, freq="YS")

            fig = go.Figure()
            colors = ["blue", "green", "orange"]  
            for i, indicator in enumerate(forecast_indicators):
                series = df[indicator].fillna(method="ffill")

                model = auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)
                forecast_values = model.predict(n_periods=5)

                forecast_df[indicator] = np.array(forecast_values)
                forecast_df.index = future_years

                clean_label = next(word for word in indicator.split() if "$" in word) + " a day" 

                # **Actual values**
                fig.add_trace(go.Scatter(
                    x=df.index, y=series, mode="lines+markers",
                    name=f"Actual: {clean_label}", line=dict(color=colors[i])
                ))
                # **Forecasted values**
                fig.add_trace(go.Scatter(
                    x=forecast_df.index, y=forecast_df[indicator], mode="lines",
                    name=f"Forecast: {clean_label}", line=dict(dash="dash", color=colors[i])
                ))
            fig.update_layout(
                title="Forecast for Poverty Headcount Ratios with 5 year forecast",
                xaxis_title="Year",
                yaxis_title="Population (%)",
                width=1000,
                height=500,
                legend=dict(x=1, y=0.5, bgcolor="rgba(255,255,255,0.5)", font=dict(size=10)),
                template="plotly_white"
            )
            st.plotly_chart(fig)
            
        if selected_category == "Extreme & Moderate Poverty":
            forecast_df = get_filtered_data(forecast_indicators)
            forecast_and_plot(forecast_df)
            
            rows = [st.columns(3) for _ in range((len(df.columns) + 2) // 3)]  # Create rows dynamically
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            
            for i, indicator in enumerate(df.columns):
                color = colors[i % len(colors)]
                min_year = df[indicator].dropna().index.min()  # Get first available year
                max_year = df[indicator].dropna().index.max()  # Get last available year
                filtered_df = df.loc[min_year:max_year]  # Filter dataframe within this range
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df[indicator], mode="lines+markers", name=indicator, line=dict(color=color)))
                fig.update_layout(title=indicator, xaxis_title="Year", yaxis_title="value", width=450, height=400)

                row = rows[i // 3]  
                row[i % 3].plotly_chart(fig)
        else:
            rows = [st.columns(3) for _ in range((len(df.columns) + 2) // 3)]  
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            
            for i, indicator in enumerate(df.columns):
                color = colors[i % len(colors)]
                min_year = df[indicator].dropna().index.min()  # Get first available year
                max_year = df[indicator].dropna().index.max()  # Get last available year
                filtered_df = df.loc[min_year:max_year]  # Filter dataframe within this range
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df[indicator], mode="lines+markers", name=indicator, line=dict(color=color)))
                fig.update_layout(title=indicator, xaxis_title="Year", yaxis_title="Population (%)", width=450, height=400, template="plotly_white", legend=dict(x=1, y=0.5, bgcolor="rgba(255,255,255,0.5)", font=dict(size=10)))

                row = rows[i // 3]  
                row[i % 3].plotly_chart(fig)
                
        # **Inequality Data Section**
        st.subheader("Inequality Data")
        selected_inequality_category = st.selectbox("Select an Inequality category", list(inequality_options.keys()))

        selected_indicators = inequality_options[selected_inequality_category]
        df = get_filtered_data(selected_indicators, selected_inequality_category)

        rows = [st.columns(3) for _ in range((len(df.columns) + 2) // 3)]  
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            
        for i, indicator in enumerate(df.columns):
            color = colors[i % len(colors)]
            min_year = df[indicator].dropna().index.min()  # Get first available year
            max_year = df[indicator].dropna().index.max()  # Get last available year
            filtered_df = df.loc[min_year:max_year]  # Filter dataframe within this range
                
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df[indicator], mode="lines+markers", name=indicator, line=dict(color=color)))
            fig.update_layout(title=indicator, xaxis_title="Year", yaxis_title="Population (%)", width=450, height=400, template="plotly_white", legend=dict(x=1, y=0.5, bgcolor="rgba(255,255,255,0.5)", font=dict(size=10)))

            row = rows[i // 3]  
            row[i % 3].plotly_chart(fig)
    
    with SE_tabs[1]:
        st.write('Migration Dashboard')
        
    with SE_tabs[2]:
        st.write('Employment and Economy Dashboard')
        
    with SE_tabs[3]:
        st.write('Education dashboard')
        

with tabs[3]:
    st.header("Vulnerability Indicators")    
    VI_tabs = st.tabs([
    "Age",
    "Gender",
    "Health",
    "Housing conditions and habitability",
    "Urban development"])
    
    with VI_tabs[0]:
        
        @st.cache_data
        def get_filtered_data(indicator_list):
            """Fetch specified indicators from the database and preprocess."""
            conn = psycopg2.connect(**DB_CONFIG)
            if len(indicator_list) == 1:
                query = f"""
                SELECT year, indicator_name, value 
                FROM health
                WHERE indicator_name = '{indicator_list[0]}'
                ORDER BY year;
                """
            else:
                query = f"""
                SELECT year, indicator_name, value 
                FROM health
                WHERE indicator_name IN {tuple(indicator_list)}
                ORDER BY year;
                """
            df = pd.read_sql(query, conn)
            conn.close()
            
            df["year"] = pd.to_datetime(df["year"], format='%Y')
            df.set_index("year", inplace=True)
            df = df.pivot(columns="indicator_name", values="value").sort_index()
            df.fillna(method="ffill", inplace=True)
            return df

        def forecast_next_5_years(df, indicator):
            """Train Auto ARIMA and forecast the next 5 years for the given indicator."""
            df_filtered = df[[indicator]].dropna()

            model = auto_arima(df_filtered, seasonal=False, suppress_warnings=True)

            forecast_years = pd.date_range(start=df.index[-1] + pd.DateOffset(years=1), periods=5, freq="YS")
            forecast_values = model.predict(n_periods=5)
            forecast_values_np=np.array(forecast_values)

            forecast_df = pd.DataFrame(forecast_values_np, index=forecast_years, columns=[indicator])
            return forecast_df

        def plot_all_indicators_with_forecast(df):
            for indicator in df.columns:
                forecast_df = forecast_next_5_years(df, indicator)
                
                last_actual_year = df.index[-1]
                first_forecast_year = forecast_df.index[0]
                last_actual_value = df[indicator].iloc[-1]
                first_forecast_value = forecast_df[indicator].iloc[0]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[indicator],
                    mode="lines+markers",
                    name="Actual Data",
                    line=dict(color="blue")
                ))
                # Bridging line (dashed)
                fig.add_trace(go.Scatter(
                    x=[last_actual_year, first_forecast_year],
                    y=[last_actual_value, first_forecast_value],
                    mode="lines",
                    showlegend=False,
                    line=dict(color="red", dash="dash")
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df[indicator],
                    mode="lines+markers",
                    name="Forecasted Data",
                    line=dict(color="red", dash="dash")
                ))
                fig.update_layout(
                    title=f"{indicator} with 5 year forecast",
                    xaxis_title="Year",
                    yaxis_title="Value",
                    template="plotly_white"
                )
                st.plotly_chart(fig, key=f"plot_{indicator}")
        categories = {
            "Population": [
                "Population ages 0-14, total",
                "Population ages 15-64, total",
                "Population ages 65 and above, total"
            ],
            "Survival": [
                "Survival to age 65, female (% of cohort)",
                "Survival to age 65, male (% of cohort)"
            ],
            "Dependency Ratio": [
                "Age dependency ratio (% of working-age population)"
            ]
        }

        st.title("Age")

        selected_category = st.selectbox("Select a category", list(categories.keys()))

        # Fetch and plot data based on selection
        selected_indicators = categories[selected_category]
        df = get_filtered_data(selected_indicators)
        plot_all_indicators_with_forecast(df)
        
    with VI_tabs[1]:
        
        @st.cache_data
        def get_filtered_data(indicator_list):
            """Fetch specified indicators from the database and preprocess."""
            conn = psycopg2.connect(**DB_CONFIG)
            if len(indicator_list) == 1:
                query = f"""
                SELECT year, indicator_name, value 
                FROM health
                WHERE indicator_name = '{indicator_list[0]}'
                ORDER BY year;
                """
            else:
                query = f"""
                SELECT year, indicator_name, value 
                FROM health
                WHERE indicator_name IN {tuple(indicator_list)}
                ORDER BY year;
                """
            df = pd.read_sql(query, conn)
            conn.close()
            
            df["year"] = pd.to_datetime(df["year"], format='%Y')
            df.set_index("year", inplace=True)
            df = df.pivot(columns="indicator_name", values="value").sort_index()
            df.fillna(method="ffill", inplace=True)
            return df

        def forecast_next_5_years(df, indicator):
            df_filtered = df[[indicator]].fillna(method="ffill")

            model = auto_arima(df_filtered, seasonal=False, suppress_warnings=True)

            forecast_years = pd.date_range(start=df.index[-1] + pd.DateOffset(years=1), periods=5, freq="YS")
            forecast_values = model.predict(n_periods=5)
            forecast_values_np=np.array(forecast_values)

            forecast_df = pd.DataFrame(forecast_values_np, index=forecast_years, columns=[indicator])
            return forecast_df

        def plot_combined_forecast(df, indicators):
            fig = go.Figure()

            color_palette = ['#9467bd', '#ff7f0e', '#2ca02c']
            color_map = {ind: color_palette[i % len(color_palette)] for i, ind in enumerate(indicators)}

            # Add actual data traces
            for indicator in indicators:
                if indicator in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[indicator],
                        mode="lines+markers",
                        name=f"{indicator} - Actual",
                        line=dict(color=color_map[indicator])  
                    ))

            for indicator in indicators:
                forecast_df = forecast_next_5_years(df, indicator)
                if forecast_df is not None:
                    fig.add_trace(go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df[indicator],
                        mode="lines",
                        name=f"{indicator} - Forecast",
                        line=dict(color=color_map[indicator], dash="dash")  
                    ))
            fig.update_layout(
                title="Population Forecast for Next 5 Years",
                xaxis_title="Year",
                yaxis_title="Population Count",
                template="plotly_white"
            )
            st.plotly_chart(fig)
                
        gender_indicators = ["Population, total", "Population, female", "Population, male"]
        df = get_filtered_data(gender_indicators)
        plot_combined_forecast(df, gender_indicators)

    with VI_tabs[2]:
        
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        color_cycle = itertools.cycle(color_palette)

        indicator_groups = {
            "Alcohol & Substance Use": [
            'Total alcohol consumption per capita, female (liters of pure alcohol, projected estimates, female 15+ years of age)',
            'Total alcohol consumption per capita, male (liters of pure alcohol, projected estimates, male 15+ years of age)',
            'Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)',
        ],
            "Disease Burden": [
                ["Number of infant deaths", "Number of infant deaths, female", "Number of infant deaths, male"],
                ["Number of under-five deaths", "Number of under-five deaths, female", "Number of under-five deaths, male"], 
                ["Number of neonatal deaths"],  
                ["Mortality rate, under-5 (per 1,000 live births)", "Mortality rate, under-5, female (per 1,000 live births)", "Mortality rate, under-5, male (per 1,000 live births)"], 
                ["Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70, female (%)"], 
                ["Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70, male (%)"],  
                ["Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70 (%)"],  
                ["Mortality rate, neonatal (per 1,000 live births)"], ["Number of maternal deaths"],
                ["Lifetime risk of maternal death (1 in: rate varies by country)"], ["Lifetime risk of maternal death (%)"],
                ["Maternal mortality ratio (national estimate, per 100,000 live births)"],
                ["Suicide mortality rate, female (per 100,000 female population)", "Suicide mortality rate, male (per 100,000 female population)", "Suicide mortality rate (per 100,000 population)"],  
                ["Mortality caused by road traffic injury (per 100,000 population)"], 
                ["Mortality rate, adult, female (per 1,000 female adults)", "Mortality rate, adult, male (per 1,000 male adults)"],  
                ["Death rate, crude (per 1,000 people)"], 
                ["Mortality rate, infant, female (per 1,000 live births)", "Mortality rate, infant (per 1,000 live births)", "Mortality rate, infant, male (per 1,000 live births)"]  
            ],
            "Health Services Access": [
            "Hospital beds (per 1,000 people)", "Nurses and midwives (per 1,000 people)", "Physicians (per 1,000 people)",
            "Current health expenditure (% of GDP)", "Current health expenditure per capita (current US$)", "Current health expenditure per capita, PPP (current international $)",
            "External health expenditure (% of current health expenditure)", "External health expenditure per capita (current US$)",
            "External health expenditure per capita, PPP (current international $)", "Domestic general government health expenditure (% of current health expenditure)",
            "Domestic general government health expenditure (% of GDP)", "Domestic general government health expenditure (% of general government expenditure)",
            "Domestic general government health expenditure per capita (current US$)", "Domestic private health expenditure (% of current health expenditure)",
            "Domestic private health expenditure per capita (current US$)", "Domestic private health expenditure per capita, PPP (current international $)"
        ],
            "Maternal & Child Health": [
                'Prevalence of anemia among children (% of children ages 6-59 months)',
                'Immunization, HepB3 (% of one-year-old children)', 'Immunization, DPT (% of children ages 12-23 months)',
                'Immunization, measles (% of children ages 12-23 months)', 'Immunization, measles second dose (% of children by the nationally recommended age)',
                'Births attended by skilled health staff (% of total)', 'Low-birthweight babies (% of births)',
                'Adolescent fertility rate (births per 1,000 women ages 15-19)', 'Birth rate, crude (per 1,000 people)',
                'Life expectancy at birth, female (years)', 'Life expectancy at birth, male (years)', 'Life expectancy at birth, total (years)',
                'Fertility rate, total (births per woman)', 'Sex ratio at birth (male births per female births)',
                'Number of stillbirths', 'Stillbirth rate (per 1,000 total births)'
            ],
            "Other": [ 
                "Prevalence of anemia among pregnant women (%)",
                "Prevalence of anemia among non-pregnant women (% of women ages 15-49)",
                "Prevalence of HIV, total (% of population ages 15-49)",
                "Incidence of HIV, all (per 1,000 uninfected population)",
                "Incidence of HIV, ages 15-49 (per 1,000 uninfected population ages 15-49)",
                "Incidence of malaria (per 1,000 population at risk)",
                "Risk of catastrophic expenditure for surgical care (% of people at risk)",
                "Risk of impoverishing expenditure for surgical care (% of people at risk)",
                "Tuberculosis treatment success rate (% of new cases)",
                "Tuberculosis case detection rate (%, all forms)",
                "Incidence of tuberculosis (per 100,000 people)",
                "Out-of-pocket expenditure per capita (current US$)",
                "Out-of-pocket expenditure per capita, PPP (current international $)",
                "Prevalence of undernourishment (% of population)",
                "Population growth (annual %)"
            ]
        }

        maternal_child_health_indicators = [
            "Prevalence of anemia among children (% of children ages 6-59 months)",
            "Immunization, HepB3 (% of one-year-old children)",
            "Immunization, DPT (% of children ages 12-23 months)",
            "Immunization, measles (% of children ages 12-23 months)",
            "Births attended by skilled health staff (% of total)",
            "Low-birthweight babies (% of births)",
            "Prevalence of stunting, height for age, female (% of children under 5)",
            "Prevalence of stunting, height for age, male (% of children under 5)",
            "Prevalence of wasting, weight for height, female (% of children under 5)",
            "Prevalence of wasting, weight for height, male (% of children under 5)",
            "Prevalence of wasting, weight for height (% of children under 5)",
            "Prevalence of severe wasting, weight for height, female (% of children under 5)",
            "Prevalence of severe wasting, weight for height, male (% of children under 5)",
            "Prevalence of severe wasting, weight for height (% of children under 5)",
            "Adolescent fertility rate (births per 1,000 women ages 15-19)",
            "Birth rate, crude (per 1,000 people)",
            "Fertility rate, total (births per woman)",
            "Sex ratio at birth (male births per female births)",
            "Completeness of birth registration, male (%)",
            "Completeness of birth registration (%)",
            "Number of stillbirths",
            "Stillbirth rate (per 1,000 total births)",
            "Immunization, measles second dose (% of children by the nationally recommended age)"
        ]

        @st.cache_data
        def get_filtered_data(indicator_list):
            conn = psycopg2.connect(**DB_CONFIG)
            # Handle single and multiple indicators correctly
            if len(indicator_list) == 1:
                query = f"""
                SELECT year, indicator_name, value 
                FROM health
                WHERE indicator_name = '{indicator_list[0]}'
                ORDER BY year;
                """
            else:
                query = f"""
                SELECT year, indicator_name, value 
                FROM health
                WHERE indicator_name IN {tuple(indicator_list)}
                ORDER BY year;
                """
            df = pd.read_sql(query, conn)
            conn.close()
            
            df["year"] = pd.to_datetime(df["year"], format='%Y')
            df.set_index("year", inplace=True)
            df = df.pivot(columns="indicator_name", values="value").sort_index()
            #df.fillna(method="ffill", inplace=True)
            return df

        def plot_data(df, indicators, title):
            if df.empty:
                return None  

            min_year, max_year = df.index.min(), df.index.max()  # Get available range
            
            fig = go.Figure()
            for indicator in indicators:
                if indicator in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.loc[min_year:max_year].index, 
                        y=df.loc[min_year:max_year, indicator],
                        mode="lines+markers",
                        name=simplify_indicator_name(indicator),
                        line=dict(color=next(color_cycle))
                    ))   
            fig.update_layout(
                title=title,
                xaxis_title="Year",
                yaxis_title="Value",
                template="plotly_white"
            )
            return fig

        def simplify_indicator_name(indicator):
            if "female" in indicator.lower():
                return "Female"
            elif "male" in indicator.lower():
                return "Male"
            else:
                return "Both"

        def plot_individual_indicators(df, indicator):
            if df.empty or indicator not in df.columns:
                return None
            min_year, max_year = df.index.min(), df.index.max()  # Determine available range
            
            fig = go.Figure()   
            fig.add_trace(go.Scatter(
                x=df.loc[min_year:max_year].index,  # Filter between min/max years
                y=df.loc[min_year:max_year, indicator],
                mode="lines+markers",
                name=indicator,
                line=dict(color=next(color_cycle))
            ))
            fig.update_layout(
                title=indicator,
                xaxis_title="Year",
                yaxis_title="Value",
                template="plotly_white"
            )
            return fig
            
        def plot_grouped_indicators(df, indicator_set, title): 
            fig = go.Figure()
            for indicator in indicator_set:
                simplified_label = simplify_indicator_name(indicator)  # Extract 'Female', 'Male', 'Both'
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[indicator],
                    mode="lines+markers",
                    name=simplified_label,
                    line=dict(color=next(color_cycle))
                ))
            fig.update_layout(
                title=title,
                xaxis_title="Year",
                yaxis_title="Value",
                template="plotly_white"
            )
            return fig

        def plot_alcohol_substance_use():
            plots = []
            grouped_plots = [
                (['Total alcohol consumption per capita, female (liters of pure alcohol, projected estimates, female 15+ years of age)',
                'Total alcohol consumption per capita, male (liters of pure alcohol, projected estimates, male 15+ years of age)',
                'Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)'],
                    "Total Alcohol Consumption per Capita by Gender")]
            
            for indicators, title in grouped_plots:
                df = get_filtered_data(indicators)
                fig = plot_data(df, indicators, title)
                if fig:
                    plots.append(fig)
            return plots
                
        def plot_disease_burden():
            plots = []
            # Grouped plots
            grouped_plots = [
                (["Number of infant deaths", "Number of infant deaths, female", "Number of infant deaths, male"], "Number of Infant Deaths"),
                (["Number of under-five deaths", "Number of under-five deaths, female", "Number of under-five deaths, male"], "Number of Under-Five Deaths"),
                (["Mortality rate, under-5 (per 1,000 live births)", "Mortality rate, under-5, female (per 1,000 live births)", "Mortality rate, under-5, male (per 1,000 live births)"], "Mortality Rate Under-5"),
                (['Suicide mortality rate, female (per 100,000 female population)','Suicide mortality rate, male (per 100,000 male population)','Suicide mortality rate (per 100,000 population)'], "Suicide Mortality Rate"),
                (["Mortality rate, adult, female (per 1,000 female adults)", "Mortality rate, adult, male (per 1,000 male adults)"], "Adult Mortality Rate"),
                (["Mortality rate, infant, female (per 1,000 live births)", "Mortality rate, infant (per 1,000 live births)", "Mortality rate, infant, male (per 1,000 live births)"], "Infant Mortality Rate")
            ]

            # Plot grouped indicators
            for indicator_set, title in grouped_plots:
                df = get_filtered_data(indicator_set)
                fig = plot_data(df, indicator_set, title)
                if fig:
                    plots.append(fig)

            # Plot individual indicators separately
            individual_plots = [indicators[0] for indicators in indicator_groups["Disease Burden"] if len(indicators) == 1]

            for indicator in individual_plots:
                df = get_filtered_data([indicator])
                fig = plot_data(df, [indicator], indicator)
                if fig:
                    plots.append(fig)
            return plots 
                
        def plot_health_services_access():
            plots = []
            for indicator in indicator_groups["Health Services Access"]:
                df = get_filtered_data([indicator])
                fig = plot_data(df, [indicator], indicator)
                if fig:
                    plots.append(fig)
            return plots
                
        def plot_maternal_child_health():
            plots = []
            grouped_life_expectancy = [
                'Life expectancy at birth, female (years)',
                'Life expectancy at birth, male (years)',
                'Life expectancy at birth, total (years)'
            ]

            df_grouped = get_filtered_data(grouped_life_expectancy)
            fig = plot_data(df_grouped, grouped_life_expectancy, "Life Expectancy at Birth")
            if fig:
                plots.append(fig)

            for indicator in indicator_groups["Maternal & Child Health"]:
                if indicator not in grouped_life_expectancy:
                    df = get_filtered_data([indicator])
                    fig = plot_data(df, [indicator], indicator)
                    if fig:
                        plots.append(fig)

            return plots

        def plot_other():
            plots = []
            
            for item in indicator_groups["Other"]:  
                if isinstance(item, str):  
                    df = get_filtered_data([item])
                    fig = plot_data(df, [item], item)
                    if fig:
                        plots.append(fig)

            return plots
                    
        # Flatten indicator list to remove any nested lists
        def flatten_indicators(category):
            indicators = []
            for item in indicator_groups[category]:
                if isinstance(item, list):
                    indicators.extend(item) 
                else:
                    indicators.append(item)
            return indicators

        def display_plots(plots):
            num_cols = 3
            cols = st.columns(num_cols)
            for i, plot in enumerate(plots):
                with cols[i % num_cols]:  
                    st.plotly_chart(plot)

        # **Dropdown for selecting category**
        selected_category = st.selectbox("Select a category", list(indicator_groups.keys()))
        selected_indicators = flatten_indicators(selected_category)  # Flatten nested lists
        df = get_filtered_data(selected_indicators)

        plots = []
        if selected_category == "Disease Burden":
            plots.extend(plot_disease_burden())
        elif selected_category == "Health Services Access":
            plots.extend(plot_health_services_access())
        elif selected_category == "Maternal & Child Health":
            plots.extend(plot_maternal_child_health())
        elif selected_category == "Alcohol & Substance Use":
            plots.extend(plot_alcohol_substance_use())
        elif selected_category == "Other":
            plots.extend(plot_other())

        display_plots(plots)
        
    with VI_tabs[4]:
        indicator_groups = {
            "Demographic" : ['Population density (people per sq. km of land area)', 'Population in largest city', 
                    'Population in the largest city (% of urban population)', 'Population in urban agglomerations of more than 1 million', 
                    'Population in urban agglomerations of more than 1 million (% of total population)', 'Urban population growth (annual %)', 'Urban population', 'Urban population (% of total)'
                    ],
            "Health & Safety" : ['Mortality caused by road traffic injury (per 100,000 population)'],
            "Infrastructure": ['Access to electricity, urban (% of urban population)']
        }

        @st.cache_data
        def get_filtered_data(indicator_list):
            conn = psycopg2.connect(**DB_CONFIG)
            # Handle single and multiple indicators correctly
            if len(indicator_list) == 1:
                query = f"""
                SELECT year, indicator_name, value 
                FROM urban_development
                WHERE indicator_name = '{indicator_list[0]}'
                ORDER BY year;
                """
            else:
                query = f"""
                SELECT year, indicator_name, value 
                FROM urban_development
                WHERE indicator_name IN {tuple(indicator_list)}
                ORDER BY year;
                """
            df = pd.read_sql(query, conn)
            conn.close()
            
            df["year"] = pd.to_datetime(df["year"], format='%Y')
            df.set_index("year", inplace=True)
            df = df.pivot(columns="indicator_name", values="value").sort_index()
            return df

        selected_category = st.selectbox("Select a Poverty Category", list(indicator_groups.keys()))
        selected_indicators = indicator_groups[selected_category]
        df = get_filtered_data(selected_indicators)

        rows = [st.columns(3) for _ in range((len(df.columns) + 2) // 3)]  
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for i, indicator in enumerate(df.columns):
            color = colors[i % len(colors)]
            min_year = df[indicator].dropna().index.min()  # Get first available year
            max_year = df[indicator].dropna().index.max()  # Get last available year
            filtered_df = df.loc[min_year:max_year]  # Filter dataframe within this range
                
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df[indicator], mode="lines+markers", name=indicator, line=dict(color=color)))
            fig.update_layout(title=indicator, xaxis_title="Year", yaxis_title="value", width=450, height=400, template="plotly_white", legend=dict(x=1, y=0.5, bgcolor="rgba(255,255,255,0.5)", font=dict(size=10)))

            row = rows[i // 3]  
            row[i % 3].plotly_chart(fig)

        
with tabs[4]:

    indicator_categories = {
        "Digital Connectivity & ICT": [
            'Mobile cellular subscriptions', 'Fixed telephone subscriptions', 'Fixed broadband subscriptions', 
            'Secure Internet servers', 'Secure Internet servers (per 1 million people)', 
            'Individuals using the Internet (% of population)'
        ],
        "Energy Infrastructure": [
            'Investment in energy with private participation (current US$)', 
            'Public private partnerships investment in energy (current US$)'
        ],
        "Innovation & Industry": [
            'Industrial design applications, nonresident, by count', 'Industrial design applications, resident, by count', 
            'Trademark applications, nonresident, by count', 'Trademark applications, resident, by count'
        ],
        "Transport Infrastructure": [
            'Investment in transport with private participation (current US$)', 
            'Public private partnerships investment in transport (current US$)',
            'Air transport, registered carrier departures worldwide', 'Air transport, freight (million ton-km)', 
            'Air transport, passengers carried', 'Railways, goods transported (million ton-km)', 
            'Railways, passengers carried (million passenger-km)', 'Rail lines (total route-km)'
        ],
        "Water Infrastructure": [
            'Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)', 
            'Annual freshwater withdrawals, domestic (% of total freshwater withdrawal)', 
            'Annual freshwater withdrawals, industry (% of total freshwater withdrawal)', 
            'Annual freshwater withdrawals, total (billion cubic meters)', 
            'Annual freshwater withdrawals, total (% of internal resources)', 
            'Renewable internal freshwater resources per capita (cubic meters)', 
            'Investment in water and sanitation with private participation (current US$)', 
            'Public private partnerships investment in water and sanitation (current US$)'
        ]
    }

    @st.cache_data
    def get_filtered_data(indicator_list):
        """Fetch data from the specified table and filter by given indicators."""
        conn = psycopg2.connect(**DB_CONFIG)
        # Handle single and multiple indicators correctly
        if len(indicator_list) == 1:
            query = f"""
            SELECT year, indicator_name, value 
            FROM infrastructure 
            WHERE indicator_name = '{indicator_list[0]}'
            ORDER BY year;
            """
        else:
            query = f"""
            SELECT year, indicator_name, value 
            FROM infrastructure 
            WHERE indicator_name IN {tuple(indicator_list)}
            ORDER BY year;
            """
            
        df = pd.read_sql(query, conn)
        conn.close()
        
        df["year"] = pd.to_datetime(df["year"], format='%Y')
        df.set_index("year", inplace=True)
        df = df.pivot(columns="indicator_name", values="value").sort_index()
        #df.fillna(method="ffill", inplace=True)
        return df

    selected_category = st.selectbox("Select a category", list(indicator_categories.keys()))
    selected_indicators = indicator_categories[selected_category]
    df = get_filtered_data(selected_indicators)

    rows = [st.columns(3) for _ in range((len(df.columns) + 2) // 3)]  
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        
    for i, indicator in enumerate(df.columns):
        color = colors[i % len(colors)]
        min_year = df[indicator].dropna().index.min()  # Get first available year
        max_year = df[indicator].dropna().index.max()  # Get last available year
        filtered_df = df.loc[min_year:max_year]  # Filter dataframe within this range
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df[indicator], mode="lines+markers", name=indicator, line=dict(color=color)))
        fig.update_layout(title=indicator, xaxis_title="Year", yaxis_title="Population (%)", width=450, height=400, template="plotly_white", legend=dict(x=1, y=0.5, bgcolor="rgba(255,255,255,0.5)", font=dict(size=10)))

        row = rows[i // 3]  
        row[i % 3].plotly_chart(fig)


with tabs[5]:
    # --- Indicator mapping ---
    st.write('AGRICULTURE')
    agriculture_mapping = {
        "Fertilizer consumption (% of fertilizer production)": "Agricultural Inputs",
        "Fertilizer consumption (kilograms per hectare of arable land)": "Agricultural Inputs",
        "Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)": "Agricultural Value & GDP Contribution",
        "Agriculture, forestry, and fishing, value added (current US$)": "Agricultural Value & GDP Contribution",
        "Agriculture, forestry, and fishing, value added (% of GDP)": "Agricultural Value & GDP Contribution",
        "Employment in agriculture, female (% of female employment) (modeled ILO estimate)": "Agricultural Value & GDP Contribution",
        "Employment in agriculture, male (% of male employment) (modeled ILO estimate)": "Agricultural Value & GDP Contribution",
        "Employment in agriculture (% of total employment) (modeled ILO estimate)": "Agricultural Value & GDP Contribution",
        "Agricultural land (sq. km)": "Crop Production & Yield",
        "Agricultural land (% of land area)": "Crop Production & Yield",
        "Arable land (hectares)": "Crop Production & Yield",
        "Arable land (hectares per person)": "Crop Production & Yield",
        "Arable land (% of land area)": "Crop Production & Yield",
        "Land under cereal production (hectares)": "Crop Production & Yield",
        "Permanent cropland (% of land area)": "Crop Production & Yield",
        "Forest area (% of land area)": "Crop Production & Yield",
        "Agricultural irrigated land (% of total agricultural land)": "Crop Production & Yield",
        "Cereal production (metric tons)": "Crop Production & Yield",
        "Crop production index (2014–2016 = 100)": "Crop Production & Yield",
        "Food production index (2014–2016 = 100)": "Crop Production & Yield",
        "Livestock production index (2014–2016 = 100)": "Crop Production & Yield",
        "Cereal yield (kg per hectare)": "Crop Production & Yield",
        "Forest area (sq. km)": "Forestry & Land Use",
        "Agricultural raw materials imports (% of merchandise imports)": "Other",
        "Agricultural raw materials exports (% of merchandise exports)": "Other",
        "Access to electricity, rural (% of rural population)": "Rural Population & Development",
        "Rural population": "Rural Population & Development",
        "Rural population growth (annual %)": "Rural Population & Development",
        "Rural population (% of total population)": "Rural Population & Development"
    }

    # --- Normalize mapping for robust matching ---
    normalized_mapping = {k.strip().lower(): v for k, v in agriculture_mapping.items()}

    # --- Cached function to load all data from the agriculture table ---
    @st.cache_data(show_spinner="Loading data from database...")
    def load_agriculture_data():
        conn = psycopg2.connect(**DB_CONFIG)
        query = """
            SELECT country_name,  year, indicator_name, value
            FROM agriculture
        """
        df = pd.read_sql(query, conn)
        conn.close()
        # Normalize indicator names for robust matching
        df['indicator_name'] = df['indicator_name'].astype(str).str.strip().str.lower()
        return df

    df = load_agriculture_data()

    # --- UI for category and country selection ---
    categories = sorted(set(normalized_mapping.values()))
    selected_category = st.selectbox("Select a main category", categories)

    # All indicators for selected category
    indicators = [k for k, v in normalized_mapping.items() if v == selected_category]

    # --- Filter for selected country and indicators ---
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    filtered_df = df[
        (df['country_name'] == 'Argentina') &
        (df['indicator_name'].isin(indicators))
    ]

    color_cycle = ['orange', 'blue', 'green', 'purple']

    # --- Plot in rows of 3 columns ---
    for i in range(0, len(indicators), 3):
        cols = st.columns(3)
        for j, indicator in enumerate(indicators[i:i+3]):
            indicator_df = filtered_df[filtered_df['indicator_name'] == indicator]
            if not indicator_df.empty:
                color = color_cycle[(i + j) % len(color_cycle)]
                fig = px.bar(
                    indicator_df,
                    x="year",
                    y="value",
                    title=indicator.title(),
                    color_discrete_sequence=[color]
                )
                cols[j].plotly_chart(fig, use_container_width=True)

                
    st.write('Enviornment')
    indicator_categories = {
        "Agriculture & Rural Development": [
        'Agricultural land (% of land area)', 'Arable land (% of land area)', 
        'Forest area (sq. km)', 'Forest area (% of land area)','Access to clean fuels and technologies for cooking, rural (% of rural population)',
        'Terrestrial protected areas (% of total land area)',
        'Forest rents (% of GDP)','People using at least basic drinking water services, rural (% of rural population)',       
        'People using at least basic sanitation services, rural (% of rural population)'
        ],
        "Environment & Climate": {
            "Pollution": [
                'PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)',
                'PM2.5 pollution, population exposed to levels exceeding WHO Interim Target-1 value (% of total)',
                'PM2.5 pollution, population exposed to levels exceeding WHO Interim Target-2 value (% of total)',
                'PM2.5 pollution, population exposed to levels exceeding WHO Interim Target-3 value (% of total)',
                'PM2.5 air pollution, population exposed to levels exceeding WHO guideline value (% of total)'
            ],
            "Greenhouse Gas Emission": [
            'Total greenhouse gas emissions including LULUCF (Mt CO2e)','Total greenhouse gas emissions excluding LULUCF (Mt CO2e)',
            'Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)', 'Total greenhouse gas emissions excluding LULUCF (% change from 1990)'

            ],
            "Methane (CH4) Emission": [
                'Methane (CH4) emissions from Agriculture (Mt CO2e)','Methane (CH4) emissions from Building (Energy) (Mt CO2e)',
                'Methane (CH4) emissions from Fugitive Emissions (Energy) (Mt CO2e)','Methane (CH4) emissions from Industrial Combustion (Energy) (Mt CO2e)',
                'Methane (CH4) emissions from Industrial Processes (Mt CO2e)','Methane (CH4) emissions (total) excluding LULUCF (Mt CO2e)',
                'Methane (CH4) emissions from Power Industry (Energy) (Mt CO2e)','Methane (CH4) emissions from Transport (Energy) (Mt CO2e)',
                'Methane (CH4) emissions from Waste (Mt CO2e)', 'Methane (CH4) emissions (total) excluding LULUCF (% change from 1990)'
                ],
            "Carbon Dioxide (CO2) Emission": [
                'Carbon dioxide (CO2) emissions from Agriculture (Mt CO2e)', 'Carbon dioxide (CO2) emissions from Building (Energy) (Mt CO2e)',
                'Carbon dioxide (CO2) emissions from Fugitive Emissions (Energy) (Mt CO2e)','Carbon dioxide (CO2) emissions from Industrial Combustion (Energy) (Mt CO2e)',
                'Carbon dioxide (CO2) emissions from Industrial Processes (Mt CO2e)','Carbon dioxide (CO2) net fluxes from LULUCF - Deforestation (Mt CO2e)',
                'Carbon dioxide (CO2) net fluxes from LULUCF - Forest Land (Mt CO2e)','Carbon dioxide (CO2) net fluxes from LULUCF - Total excluding non-tropical fires (Mt CO2e)',
                'Carbon dioxide (CO2) net fluxes from LULUCF - Other Land (Mt CO2e)','Carbon dioxide (CO2) net fluxes from LULUCF - Organic Soil (Mt CO2e)',
                'Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)','Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)',
                'Carbon dioxide (CO2) emissions from Power Industry (Energy) (Mt CO2e)'
            ],
            "F Gas Emission": [
                'F-gases emissions from Industrial Processes (Mt CO2e)'
            ],
            "Nitrous Oxide (N2O) Emission": [
                'Nitrous oxide (N2O) emissions from Agriculture (Mt CO2e)','Nitrous oxide (N2O) emissions from Building (Energy) (Mt CO2e)',
                'Nitrous oxide (N2O) emissions from Fugitive Emissions (Energy) (Mt CO2e)','Nitrous oxide (N2O) emissions from Industrial Combustion (Energy) (Mt CO2e)', 
                'Nitrous oxide (N2O) emissions from Industrial Processes (Mt CO2e)','Nitrous oxide (N2O) emissions (total) excluding LULUCF (Mt CO2e)', 
                'Nitrous oxide (N2O) emissions from Power Industry (Energy) (Mt CO2e)','Nitrous oxide (N2O) emissions from Transport (Energy) (Mt CO2e)',
                'Nitrous oxide (N2O) emissions from Waste (Mt CO2e)','Nitrous oxide (N2O) emissions (total) excluding LULUCF (% change from 1990)'
            ]
        },
        "Health": [
        'Mortality rate attributed to unintentional poisoning (per 100,000 population)',
        'Mortality rate attributed to unintentional poisoning, female (per 100,000 female population)',
        'Mortality rate attributed to unintentional poisoning, male (per 100,000 male population)',
        'Mortality rate attributed to household and ambient air pollution, age-standardized (per 100,000 population)',
        'Mortality rate attributed to unsafe water, unsafe sanitation and lack of hygiene (per 100,000 population)'
        ],
        "Infrastructure & Urban Development": [
            'Access to clean fuels and technologies for cooking, urban (% of urban population)','Access to electricity (% of population)', 
            'Urban population living in areas where elevation is below 5 meters (% of total population)','Population living in slums (% of urban population)',
            'People using at least basic sanitation services, urban (% of urban population)','People using at least basic drinking water services, urban (% of urban population)',
            'People practicing open defecation, urban (% of urban population)','People using safely managed sanitation services, urban (% of urban population)'
        ],
        "Other" : [
            'Energy intensity level of primary energy (MJ/$2017 PPP GDP)','Renewable energy consumption (% of total final energy consumption)','Plant species (higher), threatened', 
            'Mammal species, threatened','Population living in areas where elevation is below 5 meters (% of total population)',
            'Water productivity, total (constant 2015 US$ GDP per cubic meter of total freshwater withdrawal)',  'Level of water stress: freshwater withdrawal as a proportion of available freshwater resources',
            'Marine protected areas (% of territorial waters)','Terrestrial and marine protected areas (% of total territorial area)','Coal rents (% of GDP)','Mineral rents (% of GDP)', 
            'Natural gas rents (% of GDP)','Oil rents (% of GDP)', 'Total natural resources rents (% of GDP)','People using at least basic drinking water services (% of population)',
            'People using at least basic sanitation services (% of population)','People practicing open defecation (% of population)','People using safely managed sanitation services (% of population)'
        ]
    }

    @st.cache_data
    def get_filtered_data(indicator_list):
        """Fetch data from the specified table and filter by given indicators."""
        conn = psycopg2.connect(**DB_CONFIG)
        # Handle single and multiple indicators correctly
        if len(indicator_list) == 1:
            query = f"""
            SELECT year, indicator_name, value 
            FROM environment 
            WHERE indicator_name = '{indicator_list[0]}'
            ORDER BY year;
            """
        else:
            query = f"""
            SELECT year, indicator_name, value 
            FROM environment 
            WHERE indicator_name IN {tuple(indicator_list)}
            ORDER BY year;
            """
            
        df = pd.read_sql(query, conn)
        conn.close()
        
        df["year"] = pd.to_datetime(df["year"], format='%Y')
        df.set_index("year", inplace=True)
        df = df.pivot(columns="indicator_name", values="value").sort_index()
        return df

    selected_category = st.selectbox("Select a category", list(indicator_categories.keys()))
    if isinstance(indicator_categories[selected_category], dict):
        selected_subcategory = st.selectbox("Select a Subcategory", list(indicator_categories[selected_category].keys()))
        indicators = indicator_categories[selected_category][selected_subcategory]
    else:
        selected_subcategory = None
        indicators = indicator_categories[selected_category]
        
    df = get_filtered_data(indicators)

    rows = [st.columns(3) for _ in range((len(df.columns) + 2) // 3)]  
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        
    for i, indicator in enumerate(df.columns):
        color = colors[i % len(colors)]
        min_year = df[indicator].dropna().index.min()  # Get first available year
        max_year = df[indicator].dropna().index.max()  # Get last available year
        filtered_df = df.loc[min_year:max_year]  # Filter dataframe within this range
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df[indicator], mode="lines+markers", name=indicator, line=dict(color=color)))
        fig.update_layout(title=indicator, xaxis_title="Year", yaxis_title="Population (%)", width=450, height=400, template="plotly_white", legend=dict(x=1, y=0.5, bgcolor="rgba(255,255,255,0.5)", font=dict(size=10)))

        row = rows[i // 3]  
        row[i % 3].plotly_chart(fig)

