import os
import io
import pandas as pd
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt

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

csv_file = "BEJ_data.csv"
csv_folder = "csv_data"
os.path.join(csv_folder, csv_file)
dataframe = pd.read_csv(os.path.join(csv_folder, csv_file), usecols=[13], engine="python", skipfooter=3)
dataset = dataframe.values.astype("float32")
    # Replace these paths with the actual paths of the files on your system
model_path = "models/ARI_data_model.h5"
scaler_path = "models/scaler.pkl"


# Load the pre-trained model and scaler
model = load_model(model_path)
scaler = joblib.load(scaler_path)


# Sidebar for model parameters
st.sidebar.header("Model Parameters")
look_back = st.sidebar.slider("Look Back", 1, 20, 12)
epochs = st.sidebar.slider("Epochs", 10, 200, 100)
batch_size = st.sidebar.slider("Batch Size", 1, 10, 1)

    # Prepare the data for the LSTM model
def prepare_data(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i : i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

    # Normalize the dataset
dataset = scaler.transform(dataset)
X, y = prepare_data(dataset, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Retrain the model based on selected hyperparameters
if st.button("Retrain Model"):
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2)

    # Make predictions
predictions = model.predict(X)

    # Invert predictions and target values to the original scale
predictions = scaler.inverse_transform(predictions)
y_original = scaler.inverse_transform([y])

    # Plot original and predicted values
plt.figure(figsize=(8, 6))
plt.plot(y_original[0], label="Original values")
plt.plot(predictions[:, 0], label="Predicted values")
plt.legend()
st.pyplot(plt)