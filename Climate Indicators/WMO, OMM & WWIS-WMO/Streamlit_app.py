import pandas as pd
import folium
import json
import requests
import streamlit as st
from streamlit_folium import st_folium

# --- Step 1: Load city list and data ---
city_list = [
    'Bariloche', 'Buenos Aires', 'Cordoba', 'El Calafate', 'Iguazu',
    'Mar del Plata', 'Mendoza', 'Salta', 'Trelew', 'Ushuaia'
]

# Load from file
df = pd.read_csv('full_city_list.txt', sep=';')
df_arg = df[df['Country'] == 'Argentina']
df_matched = df_arg[df_arg['City'].isin(city_list)].copy()

# --- Step 2: Add coordinates ---
coords = {
    'Bariloche': (-41.1335, -71.3103),
    'Buenos Aires': (-34.6037, -58.3816),
    'Cordoba': (-31.4201, -64.1888),
    'El Calafate': (-50.3379, -72.2648),
    'Iguazu': (-25.5972, -54.5512),
    'Mar del Plata': (-38.0055, -57.5575),
    'Mendoza': (-32.8908, -68.8272),
    'Salta': (-24.7829, -65.4232),
    'Trelew': (-43.2532, -65.3090),
    'Ushuaia': (-54.8019, -68.3030)
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

# Extract city names from GeoJSON
geojson_city_names = [feature['properties']['name'] for feature in city_geojson['features']]

# Dummy volume data (if needed)
df_cities = pd.DataFrame({
    'city': geojson_city_names,
    'volumen': list(range(10, 10 + len(geojson_city_names)))
})

# Create base map
city_map = folium.Map(location=[-38.4161, -63.6167], zoom_start=3)

# Add GeoJson layer with click event tracking
def on_click_js(city_name):
    return f"""
    function(feature, layer) {{
        layer.on({{
            click: function(e) {{
                var data = {{
                    clicked_city: "{city_name}"
                }};
                window.parent.postMessage({{ type: 'folium_click', data: data }}, "*");
            }}
        }});
    }}
    """

# Apply the click handler to each city in GeoJSON
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
        tooltip=folium.GeoJsonTooltip(
            fields=['name'],
            aliases=['Ciudad:'],
            style=(
                "background-color: white; color: black; font-weight: bold; "
                "padding: 4px; border-radius: 4px;"
            ),
            highlight_style=(
                "background-color: #003366; color: white; font-weight: bold; "
                "padding: 4px; border-radius: 4px;"
            )
        )
    )
    gj.add_child(folium.features.GeoJsonPopup(fields=["name"]))
    gj.add_to(city_map)

# Render map and track interaction
map_data = st_folium(city_map, width=700, height=500)

# --- Step 5: Detect clicked city from GeoJSON (postMessage fallback used by streamlit-folium) ---
clicked_city = None
if map_data and "last_active_drawing" in map_data and map_data["last_active_drawing"]:
    clicked_city = map_data["last_active_drawing"].get("properties", {}).get("name")

# Dropdown update logic
if clicked_city:
    if clicked_city in city_list:
        selected_city = st.selectbox("Select a city to view full weather data:", city_list, index=city_list.index(clicked_city))
    else:
        st.warning(f"No data available for the clicked city: {clicked_city}")
        selected_city = st.selectbox("Select a city to view full weather data:", city_list)
else:
    selected_city = st.selectbox("Select a city to view full weather data:", city_list)

# Fetch the weather data for the selected city
def get_city_weather_df(city_name):
    entry = weather_dict.get(city_name, None)
    if entry and 'city' in entry:
        return pd.json_normalize(entry)
    else:
        return pd.DataFrame()

df = get_city_weather_df(selected_city)

if df.empty:
    st.warning(f"Data is fetched and displayed below")

import pandas as pd
from pandas import json_normalize

# Define the columns to keep (excluding the nested column to normalize)
climate_cols = [
    'city.climate.raintype', 'city.climate.raindef', 'city.climate.rainunit',
    'city.climate.datab', 'city.climate.datae', 'city.climate.tempb', 'city.climate.tempe',
    'city.climate.rdayb', 'city.climate.rdaye', 'city.climate.rainfallb', 'city.climate.rainfalle',
    'city.climate.climatefromclino'
]

changes = ['city.climate.climateMonth']  # nested column to normalize

# Extract non-nested columns
df_base = df[climate_cols].copy().reset_index(drop=True)

# Normalize nested column
normalized_parts = []
for col in changes:
    nested_data = df[col]
    exploded = nested_data.explode().reset_index(drop=True)
    norm = json_normalize(exploded)
    norm.columns = [f"{col}.{sub}" for sub in norm.columns]
    normalized_parts.append(norm)

# Combine base + normalized
city_climate = pd.concat([df_base] + normalized_parts, axis=1)

import pandas as pd
from pandas import json_normalize

# Define forecast columns
forecast_cols = ['city.forecast.issueDate', 'city.forecast.timeZone']
nested_col = 'city.forecast.forecastDay'

# Extract base columns (excluding nested list)
df_base = df[forecast_cols].copy().reset_index(drop=True)

# Explode and normalize nested forecast list (assuming it's a list of dicts)
exploded = df[nested_col].apply(pd.Series).stack().reset_index(level=1, drop=True)
exploded = exploded.reset_index(drop=True)

# Normalize each forecast entry
forecast_normalized = json_normalize(exploded)

# Add a label: Today, Today+1, ..., Today+6
forecast_normalized['ForecastDay'] = ['Today+{}'.format(i) for i in range(len(forecast_normalized))]

# If your original df has multiple cities, repeat each city’s info for 7 rows
df_repeated = pd.concat([df_base.loc[i].repeat(7) for i in df_base.index], ignore_index=True)

# Combine everything
city_forecast = pd.concat([df_repeated.reset_index(drop=True), forecast_normalized], axis=1)

# Reorder columns if needed
city_forecast = city_forecast[['ForecastDay', 'forecastDate', 'wxdesc', 'weather', 'minTemp', 'maxTemp', 'minTempF', 'maxTempF', 'weatherIcon']]

city_forecast = city_forecast.head(7)

city_forecast['minTemp'] = pd.to_numeric(city_forecast['minTemp'])
city_forecast['maxTemp'] = pd.to_numeric(city_forecast['maxTemp'])

city_forecast['forecastDate'] = pd.to_datetime(city_forecast['forecastDate'])
city_forecast['Date'] = city_forecast['forecastDate'].dt.strftime('%d %b (%a)')

icon_map = {
    'Showers': '🌧️',
    'Partly Cloudy': '⛅',
    'Mostly Cloudy': '☁️',
    'Sunny': '☀️',
    'Cloudy': '☁️',
    'Rain': '🌧️',
    'Thunderstorms': '⛈️',
    'Thundershowers': '⛈️',
    'Sleet': '🌧️❄️',
    'Snow': '❄️',
    'Light Rain': '🌦️',
    'Sunny Periods': '🌥️',
    'Fine': '🌞',
    'Mist': '🌫️',
    'Storm': '⛈️'
}

city_forecast['Icon'] = city_forecast['weather'].map(icon_map).fillna('🌈')  

city_forecast['Temp (°C)'] = city_forecast['minTemp'].astype(str) + '°C | ' + city_forecast['maxTemp'].astype(str)+ '°C'

forecast_table_df = city_forecast[['Date', 'Temp (°C)', 'Icon', 'weather']]
forecast_table_df.rename(columns={
    'weather': 'Description'
}, inplace=True)

import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(
    header=dict(
        values=list(forecast_table_df.columns),
        fill_color='#99ccff',
        font=dict(color='black', size=11),
        align='center',
        line=dict(color='black', width=1)  # Border for header
    ),
    cells=dict(
        values=[forecast_table_df[col] for col in forecast_table_df.columns],
        fill_color=[['white']],
        align='center',
        font=dict(color='black', size=11),
        line=dict(color='black', width=1)  # Border for cells
    )
)])

fig.update_layout(
    title_text=" 7-Day Weather Forecast ",
    title_font_size=20,
    font=dict(color="black")
)

st.plotly_chart(fig, use_container_width=True)


city_climate['city.climate.climateMonth.month'] = pd.to_datetime(city_climate['city.climate.climateMonth.month'], format='%m').dt.strftime('%b')
city_climate['city.climate.climateMonth.minTemp'] = pd.to_numeric(city_climate['city.climate.climateMonth.minTemp'])
city_climate['city.climate.climateMonth.maxTemp'] = pd.to_numeric(city_climate['city.climate.climateMonth.maxTemp'])
city_climate['city.climate.climateMonth.rainfall'] = pd.to_numeric(city_climate['city.climate.climateMonth.rainfall'])

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Min and Max Temperature
plt.plot(city_climate['city.climate.climateMonth.month'], city_climate['city.climate.climateMonth.minTemp'], label='Min Temperature (°C)', marker='o')
plt.plot(city_climate['city.climate.climateMonth.month'], city_climate['city.climate.climateMonth.maxTemp'], label='Max Temperature (°C)', marker='o')

# Twin y-axis for rainfall
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.bar(city_climate['city.climate.climateMonth.month'], city_climate['city.climate.climateMonth.rainfall'], alpha=0.3, color='blue', label='Rainfall (mm)')

# Labels and titles
ax1.set_xlabel('Month')
ax1.set_ylabel('Temperature (°C)')
ax2.set_ylabel('Rainfall (mm)')
plt.title('Monthly Climate Overview')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
st.pyplot(plt.gcf())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima

city_climate.index = pd.period_range(start='2025-01', periods=12, freq='M')

def forecast_series(series, periods=3):
    model = auto_arima(series, seasonal=False)
    forecast = model.predict(n_periods=periods)
    return forecast

# Forecast periods
forecast_periods = 3
future_months = pd.period_range(start='2026-01', periods=forecast_periods, freq='M').strftime('%b')

# Forecast values
minTemp_forecast = forecast_series(city_climate['city.climate.climateMonth.minTemp'])
maxTemp_forecast = forecast_series(city_climate['city.climate.climateMonth.maxTemp'])
rainfall_forecast = forecast_series(city_climate['city.climate.climateMonth.rainfall'])

# Combine month names for x-axis labels
all_months = list(city_climate.index.strftime('%b')) + list(future_months)

# Combine actual and forecasted values for plotting
min_temp_full = list(city_climate['city.climate.climateMonth.minTemp']) + list(minTemp_forecast)
max_temp_full = list(city_climate['city.climate.climateMonth.maxTemp']) + list(maxTemp_forecast)
rainfall_full  = list(city_climate['city.climate.climateMonth.rainfall'])  + list(rainfall_forecast)

plt.figure(figsize=(12, 6))

# Actual Temperatures
ax1 = plt.gca()
ax1.plot(all_months[:12], min_temp_full[:12], label='Min Temp (°C) - Actual', marker='o', color='blue')
ax1.plot(all_months[:12], max_temp_full[:12], label='Max Temp (°C) - Actual', marker='o', color='orange')

# Forecast Temperatures (starting properly at forecast months)
ax1.plot(all_months[12:15], min_temp_full[12:15], linestyle='--', color='cyan', marker='s', label='Min Temp (°C) - Forecast 2026')
ax1.plot(all_months[12:15], max_temp_full[12:15], linestyle='--', color='red', marker='s', label='Max Temp (°C) - Forecast 2026')

# Twin axis for Rainfall
ax2 = ax1.twinx()

# Actual Rainfall
ax2.bar(all_months[:12], rainfall_full[:12], alpha=0.3, color='purple', label='Rainfall (mm) - Actual')

# Forecast Rainfall
ax2.bar(all_months[12:15], rainfall_full[12:15], alpha=0.3, color='green', label='Rainfall (mm) - Forecast 2026')

# Labels and Title
ax1.set_xlabel('Month')
ax1.set_ylabel('Temperature (°C)')
ax2.set_ylabel('Rainfall (mm)')
plt.title('Monthly Climate Overview with 3-Month Forecast')

# Combine legends from both axes and place them outside the plot
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left', bbox_to_anchor=(1.05, 0.5))

# Adjust layout to fit everything cleanly
plt.tight_layout()
st.pyplot(plt.gcf())

forecast_df_next3Month = pd.DataFrame({
    'Month': future_months,
    'Min Temp (°C)': minTemp_forecast,
    'Max Temp (°C)': maxTemp_forecast,
    'Rainfall (mm)': rainfall_forecast
})

st.subheader("📈 Forecast Summary for Next 3 Months")
st.dataframe(forecast_df_next3Month, use_container_width=True)