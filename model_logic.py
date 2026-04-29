import pandas as pd
import joblib
import requests
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ---------------- LOAD FILES ----------------
df = pd.read_csv("data/organic_crop_data_with_target.csv", encoding="utf-8")
df["Crop_Name"] = df["Crop_Name"].astype(str).str.strip().str.title()

model = joblib.load("model/organic_rf_model.pkl")
le_crop = joblib.load("model/crop_label_encoder.pkl")


# ---------------- HELPER FUNCTIONS ----------------
def fetch_weather(city):
    try:
        if not API_KEY:
            return None

        url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"
        r = requests.get(url)

        if r.status_code != 200:
            return None

        data = r.json()
        return data["main"]["temp"]

    except:
        return None


def get_crop_row(crop_name):
    crop_name = crop_name.strip().title()
    row = df[df["Crop_Name"] == crop_name]

    if row.empty:
        return None

    return row.iloc[0]


# ---------------- MAIN FUNCTION ----------------
def get_crop_advisory(state, district, city, crop_name):

    crop_row = get_crop_row(crop_name)

    if crop_row is None:
        return {"error": "Crop not found in dataset"}

    current_temp = fetch_weather(city)

    ideal_min = crop_row["Temperature_Min_C"]
    ideal_max = crop_row["Temperature_Max_C"]

    if current_temp is not None:
        weather_status = "Ideal" if ideal_min <= current_temp <= ideal_max else "Not Ideal"
    else:
        weather_status = "Weather data not available"

    tips = str(crop_row["Additional_Tips"]).split(";")

    return {
        "location": {
            "state": state,
            "district": district,
            "city": city
        },
        "crop": crop_name,

        "weather": {
            "current_temperature": current_temp,
            "ideal_range": f"{ideal_min} - {ideal_max}",
            "status": weather_status
        },

        "farming_path": {
            "sowing_season": crop_row["Sowing_Season"],
            "organic_fertilizers": crop_row["Organic_Fertilizers"],
            "pest_management": crop_row["Pest_Management"],
            "watering_frequency": crop_row["Watering_Frequency"],
            "harvest_days": int(crop_row["Harvest_Days"]),
            "growth_stages": crop_row["Growth_Stages_Detailed"]
        },

        "additional_tips": [tip.strip() for tip in tips]
    }