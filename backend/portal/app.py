import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

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

def filter_data(df, date):
    filtered_data = df[df["Time"].str.startswith(date)]
    return filtered_data

# List all available regions
regions = list(region_mapping.keys())
selected_region = st.sidebar.selectbox("Choose a region", regions)
selected_date = st.sidebar.date_input("Choose a date")

# Load CSV file for the selected region
region_abbr = region_mapping[selected_region]
df = load_csv_data(region_abbr, 'csv_data')

if df is not None:
    # Filter data based on user-selected date
    filtered_data = filter_data(df, selected_date.strftime('%m-%d-%Y'))

    if not filtered_data.empty:
        # Create a line chart for FT_AVERAGE_NB_OF_USERS
        user_fig = go.Figure()
        user_fig.add_trace(go.Scatter(x=filtered_data["Time"], y=filtered_data["FT_AVERAGE NB OF USERS (UEs RRC CONNECTED)"], mode='lines', name="Average Number of Users"))
        user_fig.update_layout(title="Average Number of Users")

        st.plotly_chart(user_fig)

        # Create a line chart for FT_4G/LTE DL TRAFFIC VOLUME (GBYTES)
        traffic_fig = go.Figure()
        traffic_fig.add_trace(go.Scatter(x=filtered_data["Time"], y=filtered_data["FT_4G/LTE DL TRAFFIC VOLUME (GBYTES)"], mode='lines', name="4G LTE Downlink Traffic Volume"))
        traffic_fig.update_layout(title="4G LTE Downlink Traffic Volume")

        st.plotly_chart(traffic_fig)
    else:
        st.write("No data available for the selected region and date.")
else:
    st.write("No data available for the selected region.")
