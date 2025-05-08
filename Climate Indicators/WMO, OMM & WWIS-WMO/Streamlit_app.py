import streamlit as st

# --- Custom Header with Links ---
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        font-weight: bold;
        font-size: 16px;
        color: #7a2e2b;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f0f0;
        border-bottom: 3px solid #7a2e2b;
    }
    .custom-header {
        background-color: #ffffff;
        padding: 0px 0px;
        border-bottom: 2px solid #eaeaea;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0px;
    }
    .custom-header img {
        height: 100px;
    }
    .custom-header .header-links a {
        color: #7a2e2b;
        margin-left: 30px;
        font-size: 28px;
        text-decoration: none;
        font-weight: bold;
    }
    .custom-header .header-links a:hover {
        text-decoration: underline;
    }
    </style>

    <div class="custom-header">
        <div>
            <img src="https://cruzroja.org.ar/observatorio-humanitario/wp-content/uploads/2020/09/logos-cra-mr-2023.png" alt="Logo">
        </div>
        <div class="header-links">
            <a href="/Login" target="_self" size=16>Login</a>
            <a href="/Signup" target="_self">Signup</a>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center'>ARGENTINA</h1>", unsafe_allow_html=True)

tabs = st.tabs([
    "Climate Indicators",
    "Socio-economic Indicators",
    "Vulnerability Indicators",
    "Resilience Indicators",
    "Humanitarian Indicators"
])

with tabs[0]:
    st.header("Climate Indicators")
    st.write("Data and insights on climate patterns, risks, and forecasts.")

    import pandas as pd
    import requests
    import json
    import folium
    from streamlit_folium import st_folium
    import matplotlib.pyplot as plt
    from pmdarima import auto_arima
    from pandas import json_normalize
    import plotly.graph_objects as go

    # --- Step 1: Load city list and data ---
    city_list = [
        'Bariloche', 'Buenos Aires', 'Cordoba', 'El Calafate', 'Iguazu',
        'Mar del Plata', 'Mendoza', 'Salta', 'Trelew', 'Ushuaia'
    ]

    df = pd.read_csv('full_city_list.txt', sep=';')
    df_arg = df[df['Country'] == 'Argentina']
    df_matched = df_arg[df_arg['City'].isin(city_list)].copy()

    # --- Step 2: Add coordinates ---
    coords = {
        'Bariloche': (-41.1335, -71.3103), 'Buenos Aires': (-34.6037, -58.3816),
        'Cordoba': (-31.4201, -64.1888), 'El Calafate': (-50.3379, -72.2648),
        'Iguazu': (-25.5972, -54.5512), 'Mar del Plata': (-38.0055, -57.5575),
        'Mendoza': (-32.8908, -68.8272), 'Salta': (-24.7829, -65.4232),
        'Trelew': (-43.2532, -65.3090), 'Ushuaia': (-54.8019, -68.3030)
    }

    df_matched['lat'] = df_matched['City'].map(lambda x: coords[x][0])
    df_matched['lon'] = df_matched['City'].map(lambda x: coords[x][1])

    # --- Step 3: Fetch weather and store ---
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

    # --- Step 4: Load GeoJSON map ---
    with open("ar.json", "r", encoding="utf-8") as f:
        city_geojson = json.load(f)

    geojson_city_names = [feature['properties']['name'] for feature in city_geojson['features']]
    df_cities = pd.DataFrame({'city': geojson_city_names, 'volumen': range(10, 10 + len(geojson_city_names))})

    city_map = folium.Map(location=[-38.4161, -63.6167], zoom_start=3)

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

    map_data = st_folium(city_map, width=700, height=500)

    clicked_city = None
    if map_data and "last_active_drawing" in map_data and map_data["last_active_drawing"]:
        clicked_city = map_data["last_active_drawing"].get("properties", {}).get("name")

    if clicked_city and clicked_city in city_list:
        selected_city = st.selectbox("Select a city to view full weather data:", city_list, index=city_list.index(clicked_city))
    else:
        selected_city = st.selectbox("Select a city to view full weather data:", city_list)

    # --- Weather and Climate Data ---
    def get_city_weather_df(city_name):
        entry = weather_dict.get(city_name, None)
        if entry and 'city' in entry:
            return pd.json_normalize(entry)
        else:
            return pd.DataFrame()

    df = get_city_weather_df(selected_city)

    if df.empty:
        st.warning("No data available for selected city.")
    else:
        # --- Forecast Table ---
        forecast_cols = ['city.forecast.issueDate', 'city.forecast.timeZone']
        nested_col = 'city.forecast.forecastDay'

        df_base = df[forecast_cols].copy().reset_index(drop=True)
        exploded = df[nested_col].apply(pd.Series).stack().reset_index(level=1, drop=True).reset_index(drop=True)
        forecast_normalized = json_normalize(exploded)
        forecast_normalized['ForecastDay'] = [f'Today+{i}' for i in range(len(forecast_normalized))]

        df_repeated = pd.concat([df_base.loc[i].repeat(7) for i in df_base.index], ignore_index=True)
        city_forecast = pd.concat([df_repeated.reset_index(drop=True), forecast_normalized], axis=1)
        city_forecast = city_forecast[['ForecastDay', 'forecastDate', 'wxdesc', 'weather', 'minTemp', 'maxTemp', 'minTempF', 'maxTempF', 'weatherIcon']]
        city_forecast = city_forecast.head(7)
        city_forecast['minTemp'] = pd.to_numeric(city_forecast['minTemp'])
        city_forecast['maxTemp'] = pd.to_numeric(city_forecast['maxTemp'])
        city_forecast['forecastDate'] = pd.to_datetime(city_forecast['forecastDate'])
        city_forecast['Date'] = city_forecast['forecastDate'].dt.strftime('%d %b (%a)')

        icon_map = {
            'Showers': '🌧️', 'Partly Cloudy': '⛅', 'Mostly Cloudy': '☁️', 'Sunny': '☀️',
            'Cloudy': '☁️', 'Rain': '🌧️', 'Thunderstorms': '⛈️', 'Thundershowers': '⛈️',
            'Sleet': '🌧️❄️', 'Snow': '❄️', 'Light Rain': '🌦️', 'Sunny Periods': '🌥️',
            'Fine': '🌞', 'Mist': '🌫️', 'Storm': '⛈️'
        }

        city_forecast['Icon'] = city_forecast['weather'].map(icon_map).fillna('🌈')  
        city_forecast['Temp (°C)'] = city_forecast['minTemp'].astype(str) + '°C | ' + city_forecast['maxTemp'].astype(str) + '°C'
        forecast_table_df = city_forecast[['Date', 'Temp (°C)', 'Icon', 'weather']].rename(columns={'weather': 'Description'})

        fig = go.Figure(data=[go.Table(
            header=dict(values=list(forecast_table_df.columns), fill_color='#99ccff', align='center'),
            cells=dict(values=[forecast_table_df[col] for col in forecast_table_df.columns], align='center')
        )])
        fig.update_layout(title_text="7-Day Weather Forecast", title_font_size=20)
        st.plotly_chart(fig, use_container_width=False)

        # --- Climate Chart ---
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

        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()
        ax1.plot(city_climate['city.climate.climateMonth.month'], city_climate['city.climate.climateMonth.minTemp'], label='Min Temp (°C)', marker='o')
        ax1.plot(city_climate['city.climate.climateMonth.month'], city_climate['city.climate.climateMonth.maxTemp'], label='Max Temp (°C)', marker='o')
        ax2 = ax1.twinx()
        ax2.bar(city_climate['city.climate.climateMonth.month'], city_climate['city.climate.climateMonth.rainfall'], alpha=0.3, color='blue', label='Rainfall (mm)')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Temperature (°C)')
        ax2.set_ylabel('Rainfall (mm)')
        plt.title('Monthly Climate Overview')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        st.pyplot(plt.gcf())

        # --- Forecast Extension ---
        city_climate.index = pd.period_range(start='2025-01', periods=12, freq='M')
        forecast_periods = 3
        future_months = pd.period_range(start='2026-01', periods=forecast_periods, freq='M').strftime('%b')

        def forecast_series(series, periods=3):
            model = auto_arima(series, seasonal=False)
            forecast = model.predict(n_periods=periods)
            return forecast

        minTemp_forecast = forecast_series(city_climate['city.climate.climateMonth.minTemp'])
        maxTemp_forecast = forecast_series(city_climate['city.climate.climateMonth.maxTemp'])
        rainfall_forecast = forecast_series(city_climate['city.climate.climateMonth.rainfall'])

        all_months = list(city_climate.index.strftime('%b')) + list(future_months)
        min_temp_full = list(city_climate['city.climate.climateMonth.minTemp']) + list(minTemp_forecast)
        max_temp_full = list(city_climate['city.climate.climateMonth.maxTemp']) + list(maxTemp_forecast)
        rainfall_full  = list(city_climate['city.climate.climateMonth.rainfall']) + list(rainfall_forecast)

        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()
        ax1.plot(all_months[:12], min_temp_full[:12], label='Min Temp (°C) - Actual', marker='o', color='blue')
        ax1.plot(all_months[:12], max_temp_full[:12], label='Max Temp (°C) - Actual', marker='o', color='orange')
        ax1.plot(all_months[12:], min_temp_full[12:], linestyle='--', color='cyan', marker='s', label='Min Temp - Forecast')
        ax1.plot(all_months[12:], max_temp_full[12:], linestyle='--', color='red', marker='s', label='Max Temp - Forecast')
        ax2 = ax1.twinx()
        ax2.bar(all_months[:12], rainfall_full[:12], alpha=0.3, color='purple', label='Rainfall - Actual')
        ax2.bar(all_months[12:], rainfall_full[12:], alpha=0.3, color='green', label='Rainfall - Forecast')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Temperature (°C)')
        ax2.set_ylabel('Rainfall (mm)')
        plt.title('Monthly Climate Overview with Forecast')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout()
        st.pyplot(plt.gcf())

        # Summary table
        forecast_df_next3Month = pd.DataFrame({
            'Month': future_months,
            'Min Temp (°C)': minTemp_forecast,
            'Max Temp (°C)': maxTemp_forecast,
            'Rainfall (mm)': rainfall_forecast
        })
        st.subheader("📈 Forecast Summary for Next 3 Months")
        st.dataframe(forecast_df_next3Month, use_container_width=True)

with tabs[1]:
    st.header("Socio-economic Indicators")
    st.write("Information on demographics, economics, and social factors.")

with tabs[2]:
    st.header("Vulnerability Indicators")
    st.write("Details on exposure, sensitivity, and susceptibility to risks.")

with tabs[3]:
    st.header("Resilience Indicators")
    st.write("Data measuring ability to prepare for, respond to, and recover.")

with tabs[4]:
    st.header("Humanitarian Indicators")
    st.write("Critical needs and response capacities for humanitarian aid.")

st.markdown('</div>', unsafe_allow_html=True)
