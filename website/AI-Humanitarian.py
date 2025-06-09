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
import seaborn as sns # type: ignore
import psycopg2
from pmdarima import auto_arima #type: ignore
import bcrypt
from datetime import datetime
import pandas as pd
import calendar
import json
import numpy as np
import plotly.graph_objects as go
import folium # type: ignore
from streamlit_folium import st_folium # type: ignore
from sqlalchemy import create_engine # type: ignore
import matplotlib.pyplot as plt # type: ignore
from pandas.tseries.offsets import DateOffset    
from pmdarima import auto_arima # type: ignore
import pmdarima as pm # type: ignore
from pandas import json_normalize
import plotly.graph_objects as go
from datetime import datetime
import os
from components.header import show_header
import urllib.parse
from sqlalchemy import create_engine # type: ignore
import pandas as pd
from sqlalchemy import text
from sqlalchemy import bindparam

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

DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres.ajbcqqwgdmfscvmkbtqz',
    'password': 'StrateenaAIML',
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

        # --- City list ---
        city_list = ["Bariloche", "Buenos Aires", "Cordoba", "El Calafate", "Iguazu", "Mar del Plata", "Mendoza", "Salta", "Trelew", "Ushuaia"]

        # --- Load city geojson ---
        with open("map.geojson", "r", encoding="utf-8") as f:
            city_geojson = json.load(f)

        # --- Dropdown at the top (controls everything) ---
        selected_city = st.selectbox(
            "Select a city to view climate data:",
            city_list,
            key="city_select"
        )

        # --- Map creation (purely for display, not interactive for selection) ---
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

        left_col, right_col = st.columns([1,2])
        with left_col:
            st_folium(city_map, width=450, height=480, use_container_width=True)

        def get_engine():
            password = urllib.parse.quote_plus(DB_CONFIG['password'])
            url = (
                f"postgresql+psycopg2://{DB_CONFIG['user']}:{password}"
                f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
            )
            return create_engine(url)

        engine = get_engine()

        # --- Fetch climate data ---
        query_climate = """
            SELECT month, min_temp_c, max_temp_c, rainfall
            FROM climate_months
            WHERE city_name = %s
            ORDER BY month
        """
        df_climate = pd.read_sql_query(query_climate, engine, params=(selected_city,))


        # --- Fetch forecast data ---
        query_forecast = """
            SELECT *
            FROM forecast_days
            WHERE city_name = %s
            ORDER BY "forecastdate"
            LIMIT 6
        """
        df_forecast = pd.read_sql_query(query_forecast, engine, params=(selected_city,))

        # --- Icon map for weather types ---
        icon_map = {
            'Showers': '🌧️', 'Partly Cloudy': '⛅', 'Mostly Cloudy': '☁️', 'Sunny': '☀️',
            'Cloudy': '☁️', 'Rain': '🌧️', 'Thunderstorms': '⛈️', 'Thundershowers': '⛈️',
            'Sleet': '🌧️❄️', 'Snow': '❄️', 'Light Rain': '🌦️', 'Sunny Periods': '🌥️',
            'Fine': '🌞', 'Mist': '🌫️', 'Storm': '⛈️'
        }

        # --- Visualization ---
        with right_col:
            if df_climate.empty:
                st.warning("No data available for the selected city.")
            else:
                # --- Monthly Climate Overview ---
                months = df_climate['month'].astype(int).apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%b'))
                min_temp = np.round(df_climate['min_temp_c'], 2)
                max_temp = np.round(df_climate['max_temp_c'], 2)
                rainfall = np.round(df_climate['rainfall'], 2)

                st.markdown('<p class="forecast-summary-header">📊 Monthly Climate Overview</p>', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=months, y=min_temp, name='Min Temp (°C)', mode='lines+markers',
                    marker=dict(symbol='circle', size=8), line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=months, y=max_temp, name='Max Temp (°C)', mode='lines+markers',
                    marker=dict(symbol='circle', size=8), line=dict(color='orange')
                ))
                fig.add_trace(go.Bar(
                    x=months, y=rainfall, name='Rainfall (mm)', yaxis='y2',
                    marker=dict(color='rgba(30, 144, 255, 0.3)')
                ))
                fig.update_xaxes(
                    showspikes=False, color='black',
                    title_font=dict(color='black'), tickfont=dict(color='black')
                )
                fig.update_yaxes(
                    showspikes=False, color='black',
                    title_font=dict(color='black'), tickfont=dict(color='black')
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
                query_actual = """
                    SELECT month, year, min_temp_c, max_temp_c, rainfall
                    FROM climate_months
                    WHERE city_name = %s AND is_forecast = FALSE
                    ORDER BY year, month
                """
                df_actual = pd.read_sql_query(query_actual, engine, params=(selected_city,))

                query_forecast = """
                    SELECT month, year, min_temp_c, max_temp_c, rainfall
                    FROM climate_months_forecast
                    WHERE city_name = %s
                    ORDER BY year, month
                """
                df_forecast = pd.read_sql_query(query_forecast, engine, params=(selected_city,))

                if not df_actual.empty:
                    # Prepare actuals
                    months_actual = df_actual['month'].astype(int)
                    min_temp_actual = df_actual['min_temp_c'].astype(float)
                    max_temp_actual = df_actual['max_temp_c'].astype(float)
                    rainfall_actual = df_actual['rainfall'].astype(float)
                    year_actual = df_actual['year'].iloc[0]

                    # Prepare forecast
                    months_forecast = df_forecast['month'].astype(int) if not df_forecast.empty else []
                    min_temp_forecast = df_forecast['min_temp_c'].astype(float) if not df_forecast.empty else []
                    max_temp_forecast = df_forecast['max_temp_c'].astype(float) if not df_forecast.empty else []
                    rainfall_forecast = df_forecast['rainfall'].astype(float) if not df_forecast.empty else []
                    year_forecast = df_forecast['year'].iloc[0] if not df_forecast.empty else year_actual + 1

                    # Build x-axis labels
                    import pandas as pd
                    periods_actual = pd.period_range(start=f'{year_actual}-01', periods=len(months_actual), freq='M')
                    periods_forecast = pd.period_range(start=f'{year_forecast}-01', periods=len(months_forecast), freq='M') if not df_forecast.empty else []
                    all_periods = list(periods_actual) + list(periods_forecast)
                    all_month_labels = [p.strftime('%b %Y') for p in all_periods]

                    min_temp_full = list(min_temp_actual) + list(min_temp_forecast)
                    max_temp_full = list(max_temp_actual) + list(max_temp_forecast)
                    rainfall_full  = list(rainfall_actual) + list(rainfall_forecast)

                    st.markdown('<p class="forecast-summary-header">📉 Forecast (Next 3 Months)</p>', unsafe_allow_html=True)
                    fig2 = go.Figure()
                    # Actuals
                    fig2.add_trace(go.Scatter(
                        x=all_month_labels[:len(months_actual)], y=min_temp_full[:len(months_actual)], name='Min Temp (°C)',
                        mode='lines+markers', marker=dict(symbol='circle', size=8), line=dict(color='blue')
                    ))
                    fig2.add_trace(go.Scatter(
                        x=all_month_labels[:len(months_actual)], y=max_temp_full[:len(months_actual)], name='Max Temp (°C)',
                        mode='lines+markers', marker=dict(symbol='circle', size=8), line=dict(color='orange')
                    ))
                    fig2.add_trace(go.Bar(
                        x=all_month_labels[:len(months_actual)], y=rainfall_full[:len(months_actual)], name='Rainfall - Actual',
                        yaxis='y2', marker=dict(color='rgba(242, 5, 13, 0.3)')
                    ))
                    # Forecasts (if available)
                    if not df_forecast.empty:
                        fig2.add_trace(go.Scatter(
                            x=all_month_labels[len(months_actual):], y=min_temp_full[len(months_actual):], name='Min Temp - Forecast',
                            mode='lines+markers', line=dict(dash='dash', color='green'), marker=dict(symbol='square', size=7)
                        ))
                        fig2.add_trace(go.Scatter(
                            x=all_month_labels[len(months_actual):], y=max_temp_full[len(months_actual):], name='Max Temp - Forecast',
                            mode='lines+markers', line=dict(dash='dash', color='red'), marker=dict(symbol='square', size=7)
                        ))
                        fig2.add_trace(go.Bar(
                            x=all_month_labels[len(months_actual):], y=rainfall_full[len(months_actual):], name='Rainfall - Forecast',
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
                    fig2.update_xaxes(
                        showspikes=False, color='black',
                        title_font=dict(color='black'), tickfont=dict(color='black')
                    )
                    fig2.update_yaxes(
                        showspikes=False, color='black',
                        title_font=dict(color='black'), tickfont=dict(color='black')
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("No climate data available for the selected city.")

        with left_col:
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
                df_forecast['forecastdate'] = pd.to_datetime(df_forecast['forecastdate'], format='%d-%m-%Y %H:%M')
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
        st.header("Sea level Dashboard")
        
    with CLI_tabs[2]:
        st.header("Droughts, Hurricanes and Floods data")
        
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
            df['daily_date'] = pd.to_datetime(df['daily_date'])
            return df

        @st.cache_data
        def get_hydro_forecast(city, df, forecast_days=30):
            city_df = df[df['hydrological_station_name'] == city].sort_values('daily_date')
            ts = city_df.set_index('daily_date')['value_index'].astype(float)
            hist_forecast = city_df.set_index('daily_date')['historical_forecast'].astype(float)
            if len(ts) < 14:
                return None, None, None
            model_arima = auto_arima(
                ts,
                start_p=1, start_q=1,
                max_p=3, max_q=3,
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

                # 15-day forecast
                fig.add_trace(go.Scatter(
                    x=forecast_series.index, y=forecast_series, mode='lines+markers',
                    name='30-Day Forecast', line=dict(color='orange', dash='dash')
                ))
                fig.update_layout(
                    title=f'Forecast for {hydro_city}',
                    xaxis_title='Date',
                    yaxis_title='Drought Index Value',
                    legend=dict(x=0, y=1),
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
    
    with CLI_tabs[3]:
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
    
    SE_tabs = st.tabs([
    "Poverty and Inequality",
    "Health and Sanitation",
    "Employment and Economy",
    "Education"])
    
    with SE_tabs[0]:
        
        @st.cache_data
        def get_filtered_data(indicator_list):
            """Fetch specified indicators from the database and preprocess."""
            conn = psycopg2.connect(**DB_CONFIG)
            query = f"""
                SELECT year, indicator_name, value
                FROM poverty
                WHERE indicator_name IN {tuple(indicator_list)}
                ORDER BY year;
            """
            df = pd.read_sql(query, conn)
            conn.close()
            
            df["year"] = pd.to_datetime(df["year"])
            df.set_index("year", inplace=True)
            df = df.pivot(columns="indicator_name", values="value").sort_index()
            df.fillna(method="ffill", inplace=True)
            return df

        def plot_all_indicators(df):
            for indicator in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[indicator],
                    mode="lines+markers",
                    name=indicator
                ))
                fig.update_layout(
                    title=indicator,
                    xaxis_title="Year",
                    yaxis_title="Value",
                    template="plotly_white",
                    hovermode="x"
                )
                st.plotly_chart(fig)
                
        poverty_indicators = [
            "Poverty gap at $2.15 a day (2017 PPP) (%)",
            "Gini index",
            "Poverty gap at $3.65 a day (2017 PPP) (%)",
            "Poverty gap at $6.85 a day (2017 PPP) (%)"
        ]
        df = get_filtered_data(poverty_indicators)
        plot_all_indicators(df)

        def forecast_indicators(df):
            forecast_df = pd.DataFrame()
            future_years = pd.date_range(start=str(df.index[-1].year + 1), periods=5, freq="YS")

            for indicator in df.columns:
                model = auto_arima(df[indicator], seasonal=False, stepwise=True, suppress_warnings=True)
                forecast_values = model.predict(n_periods=5)
                forecast_values_np=np.array(forecast_values)
                forecast_df[indicator] = forecast_values_np

            forecast_df.index = future_years
            return forecast_df

        st.title("Poverty Headcount Ratio with Forecast")

        # Define target indicators
        target_indicators = [
            "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)",
            "Poverty headcount ratio at $3.65 a day (2017 PPP) (% of population)",
            "Poverty headcount ratio at $6.85 a day (2017 PPP) (% of population)"
        ]

        df = get_filtered_data(target_indicators)
        forecast_df = forecast_indicators(df)

        colors = ["blue", "green", "orange"] 
        forecast_style = "dash" 

        fig = go.Figure()
        for i, indicator in enumerate(target_indicators):
            actual_color = colors[i % len(colors)]  
            # Actual values
            clean_label = next(word for word in indicator.split() if "$" in word) + " a day"
            fig.add_trace(go.Scatter(
                x=df.index, y=df[indicator], mode="lines+markers",
                name=f"Actual: {clean_label}", line=dict(color=actual_color)
            ))
            # Forecasted values
            fig.add_trace(go.Scatter(
                x=forecast_df.index, y=forecast_df[indicator], mode="lines",
                name=f"Forecast: {clean_label}", line=dict(dash=forecast_style, color=actual_color)
            ))
            fig.update_layout(
            title="Forecast of Poverty Headcount Ratios",
            xaxis_title="Year",
            yaxis_title="Population (%)",
            width=1000, 
            height=500,
            legend=dict(
                x=1,  
                y=0.5,  
                bgcolor="rgba(255,255,255,0.5)",  
                font=dict(size=10)  
            ),
            yaxis_tickformat=",.0f")
        st.plotly_chart(fig)                
        
    with SE_tabs[1]:
        st.write('Health and Sanitation Dashboard')
        
    with SE_tabs[2]:
        st.write('Employment and Economy Dashboard')
        
    with SE_tabs[3]:
        st.write('Education dashboard')
        

with tabs[3]:
    st.header("Vulnerability Indicators")    
    SE_tabs = st.tabs([
    "Age",
    "Gender",
    "Health",
    "Housing conditions and habitability"])
          

with tabs[4]:

    def get_engine():
            password = urllib.parse.quote_plus(DB_CONFIG['password'])
            url = (
                f"postgresql+psycopg2://{DB_CONFIG['user']}:{password}"
                f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
            )
            return create_engine(url)

    engine = get_engine()

    # --- Helper: Fetch data for multiple indicators ---
    @st.cache_data
    def get_filtered_data(table_name, indicator_names):
        if len(indicator_names) == 1:
            query = f"""
            SELECT year, indicator_name, value 
            FROM {table_name}
            WHERE indicator_name = '{indicator_names[0]}'
            ORDER BY year;
            """
        else:
            query = f"""
            SELECT year, indicator_name, value 
            FROM {table_name}
            WHERE indicator_name IN {tuple(indicator_names)}
            ORDER BY year;
            """
        df = pd.read_sql(query, engine)
        return df

    # --- Main category to subcategory mapping ---
    indicator_hierarchy = {
        "Information and Communication Technology": [
            "Mobile Cellular Subscriptions",
            "Telephone Subscriptions",
            "Broadband Subscriptions",
            "Internet Usage"
        ],
        "Water Resources": [
            "Investment in Water and Sanitation"
        ],
        "Energy Infrastructure": [
            "Investment in Energy"
        ],
        "Transport Infrastructure": [
            "Investment in Transport"
        ],
        "Innovation / Intellectual Property": [
            "Industrial Design Applications",
            "Trademark Applications"
        ]
    }

    # --- Subcategory to table/indicator mapping ---
    indicator_table_mapping = {
        "Industrial Design Applications": ("infrastructure", [
            "Industrial design applications, nonresident, by count",
            "Industrial design applications, resident, by count"
        ]),
        "Trademark Applications": ("infrastructure", [
            "Trademark applications, nonresident, by count",
            "Trademark applications, resident, by count"
        ]),
        "Internet Usage": ("infrastructure", [
            "Individuals using the Internet (% of population)"
        ]),
        "Mobile Cellular Subscriptions": ("infrastructure", [
            "Mobile cellular subscriptions",
            "Mobile cellular subscriptions (per 100 people)"
        ]),
        "Telephone Subscriptions": ("infrastructure", [
            "Fixed telephone subscriptions",
            "Fixed telephone subscriptions (per 100 people)"
        ]),
        "Broadband Subscriptions": ("infrastructure", [
            "Fixed broadband subscriptions",
            "Fixed broadband subscriptions (per 100 people)"
        ]),
        "Investment in Energy": ("infrastructure", [ 
            "Investment in energy with private participation (current US$)",
            "Public private partnerships investment in energy (current US$)"
        ]),
        "Investment in Transport": ("infrastructure", [
            "Investment in transport with private participation (current US$)",
            "Public private partnerships investment in transport (current US$)"
        ]),
        "Investment in Water and Sanitation": ("infrastructure", [
            "Investment in water and sanitation with private participation (current US$)",
            "Public private partnerships investment in water and sanitation (current US$)"
        ])
    }

    st.header("Resilience Indicators")

    # --- Main selection only ---
    main_category = st.selectbox("Select Main Indicator Category", list(indicator_hierarchy.keys()))

    # --- For each subcategory, show all relevant charts ---
    for subcategory in indicator_hierarchy[main_category]:
        table_name, indicators = indicator_table_mapping[subcategory]
        query = text("SELECT * FROM infrastructure WHERE indicator_name IN :indicators").bindparams(bindparam("indicators", expanding=True))
        df = pd.read_sql(query, engine, params={"indicators": indicators})

        if df.empty:
            st.warning(f"No data available for {subcategory}")
            continue

        # Prepare DataFrame for plotting
        df["year"] = pd.to_datetime(df["year"])
        df.set_index("year", inplace=True)
        df_pivot = df.pivot(columns="indicator_name", values="value").sort_index()
        df_pivot.fillna(method="ffill", inplace=True)

        st.subheader(subcategory)

        # Plotting logic for each subcategory
        if subcategory == "Industrial Design Applications":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_pivot.index,
                y=df_pivot["Industrial design applications, nonresident, by count"],
                mode="lines+markers",
                name="Nonresident Count",
                line=dict(color="red")
            ))
            fig.add_trace(go.Scatter(
                x=df_pivot.index,
                y=df_pivot["Industrial design applications, resident, by count"],
                mode="lines+markers",
                name="Resident Count",
                line=dict(color="blue")
            ))
            fig.update_layout(
                title="Industrial Design Applications: Nonresident vs Resident",
                xaxis_title="Year",
                yaxis_title="Count",
                width=1000,
                height=500,
                legend=dict(x=0, y=1)
            )
            st.plotly_chart(fig)

        elif subcategory == "Trademark Applications":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_pivot.index,
                y=df_pivot["Trademark applications, nonresident, by count"],
                mode="lines+markers",
                name="Nonresident Count",
                line=dict(color="red")
            ))
            fig.add_trace(go.Scatter(
                x=df_pivot.index,
                y=df_pivot["Trademark applications, resident, by count"],
                mode="lines+markers",
                name="Resident Count",
                line=dict(color="blue")
            ))
            fig.update_layout(
                title="Trademark Applications: Nonresident vs Resident",
                xaxis_title="Year",
                yaxis_title="Count",
                width=1000,
                height=500,
                legend=dict(x=0, y=1)
            )
            st.plotly_chart(fig)

        elif subcategory == "Internet Usage":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_pivot.index,
                y=df_pivot["Individuals using the Internet (% of population)"],
                mode="lines+markers",
                name="Internet Usage",
                line=dict(color="blue")
            ))
            fig.update_layout(
                title="Individuals using the Internet (% of population)",
                xaxis_title="Year",
                yaxis_title="Population %",
                width=1000,
                height=400,
                legend=dict(x=0, y=1)
            )
            st.plotly_chart(fig)

        elif subcategory == "Mobile Cellular Subscriptions":
            # Plot 1: Mobile Cellular Subscriptions
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_pivot.index,
                y=df_pivot["Mobile cellular subscriptions"],
                mode="lines+markers",
                name="Subscription",
                line=dict(color="red")
            ))
            fig1.update_layout(
                title="Mobile Cellular Subscriptions",
                xaxis_title="Year",
                yaxis_title="Subscriptions",
                width=1000,
                height=400
            )
            st.plotly_chart(fig1)

        elif subcategory == "Telephone Subscriptions":
            # Plot Fixed Telephone Subscriptions
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_pivot.index,
                y=df_pivot["Fixed telephone subscriptions"],
                mode="lines+markers",
                name="Fixed Telephone Subscriptions",
                line=dict(color="red")
            ))
            fig1.update_layout(
                title="Fixed Telephone Subscriptions",
                xaxis_title="Year",
                yaxis_title="Subscriptions",
                width=1000,
                height=400
            )
            st.plotly_chart(fig1)

        elif subcategory == "Broadband Subscriptions":
            # Plot Fixed Broadband Subscriptions
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_pivot.index,
                y=df_pivot["Fixed broadband subscriptions"],
                mode="lines+markers",
                name="Fixed Broadband Subscriptions",
                line=dict(color="red")
            ))
            fig1.update_layout(
                title="Fixed Broadband Subscriptions",
                xaxis_title="Year",
                yaxis_title="Subscriptions",
                width=1000,
                height=400
            )
            st.plotly_chart(fig1)
            

        elif subcategory == "Investment in Energy":
            # Plot Investment in Energy with Private Participation
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_pivot.index,
                y=df_pivot["Investment in energy with private participation (current US$)"],
                mode="lines+markers",
                name="Private Investment",
                line=dict(color="red")
            ))
            fig1.update_layout(
                title="Investment in Energy with Private Participation",
                xaxis_title="Year",
                yaxis_title="Investment (Current US$)",
                width=1000,
                height=400
            )
            st.plotly_chart(fig1)
            # Plot Public-Private Partnerships Investment in Energy
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_pivot.index,
                y=df_pivot["Public private partnerships investment in energy (current US$)"],
                mode="lines+markers",
                name="Public-Private Partnerships",
                line=dict(color="blue")
            ))
            fig2.update_layout(
                title="Public-Private Partnerships Investment in Energy",
                xaxis_title="Year",
                yaxis_title="Investment (Current US$)",
                width=1000,
                height=400
            )
            st.plotly_chart(fig2)

        elif subcategory == "Investment in Transport":
            # Plot Investment in Transport with Private Participation
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_pivot.index,
                y=df_pivot["Investment in transport with private participation (current US$)"],
                mode="lines+markers",
                name="Private Investment",
                line=dict(color="red")
            ))
            fig1.update_layout(
                title="Investment in Transport with Private Participation",
                xaxis_title="Year",
                yaxis_title="Investment (Current US$)",
                width=1000,
                height=400
            )
            st.plotly_chart(fig1)
            

        elif subcategory == "Investment in Water and Sanitation":
            # Plot Investment in Water & Sanitation with Private Participation
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_pivot.index,
                y=df_pivot["Investment in water and sanitation with private participation (current US$)"],
                mode="lines+markers",
                name="Private Investment",
                line=dict(color="red")
            ))
            fig1.update_layout(
                title="Investment in Water & Sanitation with Private Participation",
                xaxis_title="Year",
                yaxis_title="Investment (Current US$)",
                width=1000,
                height=400
            )
            st.plotly_chart(fig1)  

with tabs[5]:
    st.header("Humanitarian Indicators")
    st.write("Critical needs and response capacities for humanitarian aid.")

st.markdown('</div>', unsafe_allow_html=True)