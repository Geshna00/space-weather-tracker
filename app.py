import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv(dotenv_path=".env")

# Get API key
API_KEY = os.getenv("API_KEY")

# Debug print
print("API KEY:", API_KEY)

# API URL
url = f"https://api.nasa.gov/DONKI/FLR?startDate=2023-01-01&api_key={API_KEY}"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)

    print("✅ Data fetched successfully!\n")
    print("\nColumns:\n", df.columns)   # 👈 ADD HERE

    print("\nFirst 5 rows:\n")
    print(df.head())
    df = df[['classType', 'peakTime']]

    print("\nAfter selecting columns:\n")
    print(df.head())
    # Convert classType → numeric levels
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

    # Convert peakTime → datetime
    df['peakTime'] = pd.to_datetime(df['peakTime'], errors='coerce')

    # Extract time-based features
    df['hour'] = df['peakTime'].dt.hour
    df['day'] = df['peakTime'].dt.day
    df['month'] = df['peakTime'].dt.month

    print("\nAfter feature engineering:\n")
    print(df.head())
    df = df.dropna()

    print("\nAfter cleaning:\n")
    print(df.head())
    print("\nAfter cleaning:\n")
    print(df.head())

    # ================= ML STARTS HERE =================

    # Features (inputs)
    X = df[['hour', 'day', 'month']]

    # Target (output)
    y = df['flare_level']

    print("\nFeatures (X):\n", X.head())
    print("\nTarget (y):\n", y.head())

    # Train-test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    print("\n✅ Model trained successfully!")

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print("\nModel Accuracy:", accuracy)

    # Prediction
    sample = [[12, 10, 1]]

    prediction = model.predict(sample)
    print("\nPrediction for sample input:", prediction)

else:
    print("❌ Failed to fetch data")
    print("Status Code:", response.status_code)
    print("Response:", response.text)