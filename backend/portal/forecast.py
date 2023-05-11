import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from keras.models import load_model
import plotly.graph_objects as go

# Load models and scalers
model_users = load_model('models/lstm_model_ARI_users.h5')
model_traffic = load_model('models/lstm_model_ARI_traffic.h5')
scaler_users = joblib.load('models/scaler_ARI_users(1).gz')
scaler_traffic = joblib.load('models/scaler_ARI_traffic(1).gz')

# Function to create sequences
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

# Load CSV data
@st.cache_data
def load_data():
    return pd.read_csv('csv_data/ARI_data.csv')

df = load_data()

base_stations = df["eNodeB Name"].unique()
selected_base_station = st.sidebar.selectbox("Choose a base station", base_stations)
    
cells = df[df["eNodeB Name"] == selected_base_station]["Cell Name"].unique()
selected_cell = st.sidebar.selectbox("Choose a cell", cells)

neurons = st.sidebar.slider("Select number of neurons", 10, 100, step=10)
epochs = st.sidebar.slider("Select number of epochs", 5, 100, step=5)
batch_size = st.sidebar.slider("Select batch size", 1, 20, step=1)

filtered_data = df[(df["eNodeB Name"] == selected_base_station) & (df["Cell Name"] == selected_cell)]

seq_length = 7

# Create sequences and make predictions
X_users, _ = create_sequences(filtered_data["FT_AVERAGE NB OF USERS (UEs RRC CONNECTED)"].values, seq_length)
X_traffic, _ = create_sequences(filtered_data["FT_4G/LTE DL TRAFFIC VOLUME (GBYTES)"].values, seq_length)

last_sequence_users = np.array([X_users[-1]])
last_sequence_traffic = np.array([X_traffic[-1]])

# Reshape sequences to 2D before scaling
last_sequence_users_scaled = scaler_users.transform(last_sequence_users.reshape(-1, 1))
last_sequence_traffic_scaled = scaler_users.transform(last_sequence_traffic.reshape(-1, 1))

prediction_users_scaled = model_users.predict(last_sequence_users_scaled[np.newaxis, :, :])
prediction_traffic_scaled = model_traffic.predict(last_sequence_traffic_scaled[np.newaxis, :, :])

prediction_users = scaler_users.inverse_transform(prediction_users_scaled)
prediction_traffic = scaler_users.inverse_transform(prediction_traffic_scaled)

# Create time sequences for plotting
time_seq_users = np.append(filtered_data["Time"].values[seq_length:], 'next')
time_seq_traffic = np.append(filtered_data["Time"].values[seq_length:], 'next')

# Add the real values and the predicted value to create sequences for plotting
plot_values_users = np.append(filtered_data["FT_AVERAGE NB OF USERS (UEs RRC CONNECTED)"].values[seq_length:], prediction_users)
plot_values_traffic = np.append(filtered_data["FT_4G/LTE DL TRAFFIC VOLUME (GBYTES)"].values[seq_length:], prediction_traffic)

# Create a line chart for FT_AVERAGE_NB_OF_USERS
fig_users = go.Figure()
fig_users.add_trace(go.Scatter(x=time_seq_users[:-1], y=plot_values_users[:-1], mode='lines', name='Number of Users'))
fig_users.add_trace(go.Scatter(x=time_seq_users[-2:], y=plot_values_users[-2:], mode='lines', name='Predicted Number of Users', line=dict(color='orange')))

fig_users.update_layout(title="Average Number of Users (Forecasted)")

st.plotly_chart(fig_users)

# Create a line chart for FT_4G/LTE DL TRAFFIC VOLUME (GBYTES)
fig_traffic = go.Figure()
fig_traffic.add_trace(go.Scatter(x=time_seq_traffic[:-1], y=plot_values_traffic[:-1], mode='lines', name='Traffic Volume'))
fig_traffic.add_trace(go.Scatter(x=time_seq_traffic[-2:], y=plot_values_traffic[-2:], mode='lines', name='Predicted Traffic Volume', line=dict(color='orange')))

fig_traffic.update_layout(title="4G LTE Downlink Traffic Volume (GBYTES) (Forecasted)")

st.plotly_chart(fig_traffic)
