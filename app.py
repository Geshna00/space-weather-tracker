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
    print(df.head())

else:
    print("❌ Failed to fetch data")
    print("Status Code:", response.status_code)
    print("Response:", response.text)