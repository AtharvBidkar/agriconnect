import streamlit as st
import pandas as pd
import joblib
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Organic Farming Recommendation System",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.title {
    color: #2e7d32;
    font-size: 32px;
    font-weight: 700;
}
.subtitle {
    color: #1f4e79;
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<div class='title'>Organic Farming Recommendation System</div>", unsafe_allow_html=True)
st.write("Crop-wise organic farming guidance based on live weather and agronomic data")

# ---------------- LOAD DATA & MODEL ----------------
df = pd.read_csv("data/organic_crop_data_with_target.csv", encoding="utf-8")
df["Crop_Name"] = df["Crop_Name"].astype(str).str.strip().str.title()

model = joblib.load("model/organic_rf_model.pkl")
le_crop = joblib.load("model/crop_label_encoder.pkl")

API_KEY = "e9d5e3b92ecf1f26b5f2742d202d1de6"

# ---------------- FUNCTIONS ----------------
def fetch_weather(place):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={place},IN&appid={API_KEY}&units=metric"
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def get_crop_row(crop_name):
    row = df[df["Crop_Name"] == crop_name]
    if row.empty:
        return None
    return row.iloc[0]

def clean_text(text):
    return (
        str(text)
        .replace("â", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


# ---------------- INPUT SECTION ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Input Details</div>", unsafe_allow_html=True)

state = st.text_input("State")
district = st.text_input("District")
city = st.text_input("City / Village")
crop_name = st.text_input("Crop Name")

st.markdown("</div>", unsafe_allow_html=True)

state = state.strip().title()
district = district.strip().title()
city = city.strip().title()
crop_name = crop_name.strip().title()

# ---------------- MAIN LOGIC ----------------
if st.button("Generate Organic Farming Path"):

    crop_row = get_crop_row(crop_name)

    if crop_row is None:
        st.error("Crop not found in dataset.")
    else:
        weather = fetch_weather(city)

        if weather:
            current_temp = weather["main"]["temp"]
        else:
            current_temp = None

        #  Correct column names (FIXED)
        ideal_min = crop_row["Temperature_Min_C"]
        ideal_max = crop_row["Temperature_Max_C"]

        # ---------------- WEATHER CARD ----------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Weather Analysis</div>", unsafe_allow_html=True)

        if current_temp is not None:
            st.write("Current Temperature:", f"{current_temp} °C")
            if ideal_min <= current_temp <= ideal_max:
                st.success("Current weather is suitable for this crop.")
            else:
                st.warning("Current weather is not ideal for this crop.")
                st.write("Ideal Temperature Range:", f"{ideal_min} °C – {ideal_max} °C")
        else:
            st.warning("Live weather not available. Using ideal crop conditions.")
            st.write("Ideal Temperature Range:", f"{ideal_min} °C – {ideal_max} °C")

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- FARMING PATH CARD ----------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Organic Farming Path (Seed to Harvest)</div>", unsafe_allow_html=True)

        st.write("Sowing Season:", crop_row["Sowing_Season"])
        st.write("Organic Fertilizers:", crop_row["Organic_Fertilizers"])
        st.write("Pest Management:", crop_row["Pest_Management"])
        st.write("Watering Frequency:", crop_row["Watering_Frequency"])
        st.write("Harvest Duration:", f'{crop_row["Harvest_Days"]} days')
        st.write("Growth Stages:", crop_row["Growth_Stages_Detailed"])


        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- ADDITIONAL TIPS CARD ----------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Additional Crop-Specific Tips</div>", unsafe_allow_html=True)

        tips = str(crop_row["Additional_Tips"]).split(";")
        for tip in tips:
            st.write("-", tip.strip())

        st.markdown("</div>", unsafe_allow_html=True)

       
