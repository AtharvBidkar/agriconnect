import joblib
import requests

# Load trained model & encoder
model = joblib.load("organic_rf_model.pkl")
le_crop = joblib.load("crop_label_encoder.pkl")

# --------- USER INPUT (frontend ) ----------
crop_name = "Tomato"
city_name = "Nashik"

# --------- WEATHER FETCH ----------
API_KEY = "e9d5e3b92ecf1f26b5f2742d202d1de6"
url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}&units=metric"

response = requests.get(url)
data = response.json()

current_temp = data["main"]["temp"]

# --------- CROP ENCODING ----------
crop_encoded = le_crop.transform([crop_name])[0]

# --------- MODEL PREDICTION ----------
prediction = model.predict([[crop_encoded, current_temp - 2, current_temp + 2]])

# --------- FINAL OUTPUT ----------
print("Crop:", crop_name)
print("City:", city_name)
print("Current Temperature:", current_temp)

if prediction[0] == 1:
    print("Weather is suitable for this crop.")
else:
    print("Weather is not ideal, but organic path will still be provided.")

print("\n--- ORGANIC FARMING PATH ---")
print("• Use organic seeds")
print("• Apply vermicompost")
print("• Use neem oil for pest control")
print("• Follow proper irrigation schedule")
