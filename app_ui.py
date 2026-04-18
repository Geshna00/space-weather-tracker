import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# Load API key
load_dotenv(dotenv_path=".env")
API_KEY = os.getenv("API_KEY")

# Fetch recent data
today = datetime.today()
past = today - timedelta(days=30)

url = f"https://api.nasa.gov/DONKI/FLR?startDate={past.date()}&endDate={today.date()}&api_key={API_KEY}"

try:
    response = requests.get(url, timeout=10)

    if response.status_code == 200 and response.text.strip() != "":
        data = response.json()
        df = pd.DataFrame(data)
    else:
        st.warning("⚠️ API not responding. Using fallback data.")
        df = pd.DataFrame({
            'classType': ['C1.0', 'M1.2', 'X1.0'],
            'peakTime': [
                '2026-01-01T10:00Z',
                '2026-01-02T12:00Z',
                '2026-01-03T15:00Z'
            ]
        })

except Exception as e:
    st.error("🚨 API Error. Using fallback data.")
    df = pd.DataFrame({
        'classType': ['C1.0', 'M1.2', 'X1.0'],
        'peakTime': [
            '2026-01-01T10:00Z',
            '2026-01-02T12:00Z',
            '2026-01-03T15:00Z'
        ]
    })

# Preprocess
df = df[['classType', 'peakTime']]

def encode_class(x):
    if isinstance(x, str) and x.startswith('C'):
        return 1
    elif isinstance(x, str) and x.startswith('M'):
        return 2
    elif isinstance(x, str) and x.startswith('X'):
        return 3
    else:
        return 0

df['flare_level'] = df['classType'].apply(encode_class)
df['peakTime'] = pd.to_datetime(df['peakTime'], errors='coerce')

df['hour'] = df['peakTime'].dt.hour
df['day'] = df['peakTime'].dt.day
df['month'] = df['peakTime'].dt.month

df = df.dropna()

# Train model
X = df[['hour', 'day', 'month']]
y = df['flare_level']

model = RandomForestClassifier()
model.fit(X, y)

# UI
st.title("🌌 Space Weather Tracker")
st.write("Predict Solar Flare Intensity")

with st.form("prediction_form"):
    hour = st.slider("🕐 Hour", 0, 23)
    day = st.slider("📅 Day", 1, 31)
    month = st.slider("📆 Month", 1, 12)

    submit = st.form_submit_button("🚀 Predict")

if submit:
    sample = pd.DataFrame([[hour, day, month]], columns=['hour','day','month'])
    prediction = model.predict(sample)

    if prediction[0] == 3:
        st.error("🚨 High Solar Activity (X-class)")
    elif prediction[0] == 2:
        st.warning("⚠️ Moderate Activity (M-class)")
    else:
        st.success("✅ Low Activity (C-class)")