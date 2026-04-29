from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load model
model = pickle.load(open("model/organic_rf_model.pkl", "rb"))

class InputData(BaseModel):
    temperature: float
    humidity: float
    rainfall: float

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([[data.temperature, data.humidity, data.rainfall]])
    return {"prediction": str(prediction[0])}