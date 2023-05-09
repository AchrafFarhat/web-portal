import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import timedelta


# Mapping of full region names to their abbreviations
region_mapping = {
    "Ariana": "ARI",
    "Ben Arous": "BAR",
    "Beja": "BEJ",
    "Bizerte": "BIZ",
    "Gabes": "GAB",
    "Gafsa": "GAF",
    "Jendouba": "JEN",
    "Kairaouane": "KAI",
    "Kasserine": "KAS",
    "Kebeli": "KEB",
    "Kef": "KEF",
    "Mahdia": "MAH",
    "Manouba": "MAN",
    "Medenine": "MED",
    "Monastir": "MON",
    "Nabeul": "NAB",
    "Sidi Bouzid": "SBO",
    "Sfax": "SFA",
    "Siliana": "SIL",
    "Sousse": "SOU",
    "Tataouine": "TAT",
    "Tozeur": "TOZ",
    "Tunis": "TUN",
    "Zaghouane": "ZAG",

    # Add the remaining regions here
}

def load_csv_data(region_abbr, folder_path):
    file_path = os.path.join(folder_path, f"{region_abbr}_data.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    return None

def filter_data(df, start_date, end_date, base_station, cell):
    filtered_data = df[(df["Time"].between(start_date, end_date)) & (df["eNodeB Name"] == base_station) & (df["Cell Name"] == cell)]
    return filtered_data

# List all available regions
regions = list(region_mapping.keys())
selected_region = st.sidebar.selectbox("Choose a region", regions)
selected_date = st.sidebar.date_input("Choose a date")
time_period = st.sidebar.selectbox("Choose a time period", ["Daily", "Weekly", "Monthly"])

if time_period == "Daily":
    start_date = selected_date.strftime('%m-%d-%Y')
    end_date = selected_date.strftime('%m-%d-%Y')
elif time_period == "Weekly":
    start_date = selected_date.strftime('%m-%d-%Y')
    end_date = (selected_date + timedelta(days=6)).strftime('%m-%d-%Y')
else:  # Monthly
    start_date = selected_date.strftime('%m-%d-%Y')
    end_date = (selected_date + timedelta(days=30)).strftime('%m-%d-%Y')

# Load CSV file for the selected region
region_abbr = region_mapping[selected_region]
df = load_csv_data(region_abbr, 'csv_data')

if df is not None:
    base_stations = df["eNodeB Name"].unique()
    selected_base_station = st.sidebar.selectbox("Choose a base station", base_stations)
    
    cells = df[df["eNodeB Name"] == selected_base_station]["Cell Name"].unique()
    selected_cell = st.sidebar.selectbox("Choose a cell", cells)
    
    # Filter data based on user-selected date range, base station, and cell
    filtered_data = filter_data(df, start_date, end_date, selected_base_station, selected_cell)

    if not filtered_data.empty:
        # Create a line chart for FT_AVERAGE_NB_OF_USERS
        user_fig = go.Figure()
        user_fig.add_trace(go.Scatter(x=filtered_data["Time"], y=filtered_data["FT_AVERAGE NB OF USERS (UEs RRC CONNECTED)"], mode='lines', name="Average Number of Users"))
        user_fig.update_layout(title="Average Number of Users")

        st.plotly_chart(user_fig)

        # Create a line chart for FT_4G/LTE DL TRAFFIC VOLUME (GBYTES)
        traffic_fig = go.Figure()
        traffic_fig.add_trace(go.Scatter(x=filtered_data["Time"], y=filtered_data["FT_4G/LTE DL TRAFFIC VOLUME (GBYTES)"], mode='lines', name="4G LTE Downlink Traffic Volume"))
        traffic_fig.update_layout(title="4G LTE Downlink Traffic Volume (GBYTES)")

        st.plotly_chart(traffic_fig)
    else:
        st.write("No data available for the selected region, date, base station, and cell.")
else:
    st.write("No data available for the selected region.")