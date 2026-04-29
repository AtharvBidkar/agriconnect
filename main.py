from fastapi import FastAPI
from pydantic import BaseModel
from model_logic import get_crop_advisory

app = FastAPI()

class InputData(BaseModel):
    state: str
    district: str
    city: str
    crop: str

@app.post("/predict")
def predict(data: InputData):
    return get_crop_advisory(
        data.state,
        data.district,
        data.city,
        data.crop
    )