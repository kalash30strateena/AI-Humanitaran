import streamlit as st
import pandas as pd
import requests
import json
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from pandas import json_normalize
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# --- Custom CSS and Header HTML ---
if "language" not in st.session_state:
    st.session_state["language"] = "English"

st.markdown("""
<style>
/* Header styling */
.custom-header {{
    background-color: #ffffff;
    font-size: 20.5px;
    padding: 0px 0px;
    border-bottom: 2px solid #eaeaea;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0px;
}}
.custom-header img {{
    height: 100px;
}}
.custom-header .header-center {{
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}}
.custom-header .header-links {{
    display: flex;
    align-items: center;
}}
.custom-header .header-links a {{
    color: #7a2e2b;
    margin-left: 10px;
    font-size: 16px;
    text-decoration: none;
    padding: 8px 8px;
    border-radius: 18px;
    transition: background 0.2s, color 0.2s;
}}
.custom-header .header-links a:hover {{
    background-color: #b22222;
    color: #fff !important;
    border-radius: 18px;
    text-decoration: none;
}}
.language-select {{
    margin-right: 30px;
    font-size: 16px;
    padding: 4px 8px;
    border-radius: 4px;
    border: 1px solid #ccc;
    color: #7a2e2b;
    background-color: #f9f9f9;
}}
.stTabs [data-baseweb="tab"] {{
    margin-right: 7px !important;
}}
.stTabs [data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {{
    font-size: 20px !important;
    color: #7a2e2b;
    transition: color 0.2s;
}}
/* Active (clicked) tab styling */
.stTabs [data-baseweb="tab"][aria-selected="true"] > div[data-testid="stMarkdownContainer"] > p {{
    color: #e53935 !important; 
}}
/* Weather forecast card styling */
.forecast-card {{
    background: #62a1c7;
    color: #fff;
    border-radius: 10px;
    padding: 6px 6px 6px 6px;
    margin: 2px;
    text-align: center;
    min-width: 200px;
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

<div class="custom-header">
    <div>
        <img src="https://cruzroja.org.ar/observatorio-humanitario/wp-content/uploads/2020/09/logos-cra-mr-2023.png" alt="Logo" width="400">
    </div>
    <div class="header-center"></div>
    <div class="header-links">
        <form action="" method="get" id="lang-form">
            <select name="language" class="language-select" onchange="document.getElementById('lang-form').submit();">
                <option value="English" {eng_selected}>English</option>
                <option value="Español" {esp_selected}>Español</option>
            </select>
        </form>
        <a href="/Login" target="_self">Login</a>
        <a href="/Signup" target="_self">Signup</a>
    </div>
</div>
""".format(
    eng_selected="selected" if st.session_state.get("language", "English") == "English" else "",
    esp_selected="selected" if st.session_state.get("language", "English") == "Español" else "",
), unsafe_allow_html=True)

st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center'>ARGENTINA</h1>", unsafe_allow_html=True)

tabs = st.tabs([
    "Climate Indicators",
    "Socio-economic Indicators",
    "Vulnerability Indicators",
    "Resilience Indicators",
    "Humanitarian Indicators"
])

# --- Climate Indicators Tab ---
with tabs[0]:
    st.header("Climate Indicators")

    # --- City selection dropdown ---
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
        map_data = st_folium(city_map, width=450, height=480)

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
            plt.figure(figsize=(17, 10))
            ax1 = plt.gca()
            ax1.plot(city_climate['city.climate.climateMonth.month'], city_climate['city.climate.climateMonth.minTemp'], label='Min Temp (°C)', marker='o')
            ax1.plot(city_climate['city.climate.climateMonth.month'], city_climate['city.climate.climateMonth.maxTemp'], label='Max Temp (°C)', marker='o')
            ax2 = ax1.twinx()
            ax2.bar(city_climate['city.climate.climateMonth.month'], city_climate['city.climate.climateMonth.rainfall'], alpha=0.3, color='blue', label='Rainfall (mm)')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Temperature (°C)')
            ax2.set_ylabel('Rainfall (mm)')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            plt.tight_layout()
            plt.grid(True)
            st.pyplot(plt.gcf())

            st.markdown('<p class="forecast-summary-header">📉 Forecast (Next 3 Months)</p>', unsafe_allow_html=True)
            city_climate.index = pd.period_range(start='01', periods=12, freq='M')
            forecast_periods = 3
            future_months = pd.period_range(start='01', periods=forecast_periods, freq='M').strftime('%b')

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
            ax2.bar(all_months[:12], rainfall_full[:12], alpha=0.3, color='#f2050d', label='Rainfall - Actual')
            ax2.bar(all_months[12:], rainfall_full[12:], alpha=0.3, color='green', label='Rainfall - Forecast')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Temperature (°C)')
            ax2.set_ylabel('Rainfall (mm)')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=1)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            
        with left_col:
            forecast_df_next3Month = pd.DataFrame({
                'Month': future_months,
                'Min Temp (°C)': minTemp_forecast,
                'Max Temp (°C)': maxTemp_forecast,
                'Rainfall (mm)': rainfall_forecast
            })
            # Convert 'Month' from '2026-01' to '2026-Jan'
            month_map = {
                '01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr',
                '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug',
                '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'
            }
            forecast_df_next3Month['Month'] = forecast_df_next3Month['Month'].apply(
                lambda x: x[:4] + '-' + month_map.get(x[5:7], x[5:7])
            )

            # Round numeric columns to 2 decimals
            for col in ['Min Temp (°C)', 'Max Temp (°C)', 'Rainfall (mm)']:
                forecast_df_next3Month[col] = forecast_df_next3Month[col].round(2)

            st.markdown('<p class="forecast-summary-header">📈 Forecast Summary (for Next 3 Months)</p>', unsafe_allow_html=True)
            st.dataframe(forecast_df_next3Month, use_container_width=True)

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

        cols = st.columns(6)
        for i, row in city_forecast.iterrows():
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
            with cols[i]:
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