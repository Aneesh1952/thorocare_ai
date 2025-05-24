from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load artifacts
health_model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")
readiness_model = joblib.load('readiness.pkl')

class HorseData(BaseModel):
    Heart_Rate: float
    Respiration_Rate: float
    Temperature: float
    Movement_Activity: float
    Reaction_Score: float
    Therapy_Readiness: str

@app.post("/predict")
def predict(data: HorseData):
    try:
        # Process health prediction
        therapy_encoded = encoders['Therapy_Readiness'].transform([data.Therapy_Readiness])[0]
        numerical_features = [
            data.Heart_Rate,
            data.Respiration_Rate,
            data.Temperature,
            data.Movement_Activity,
            data.Reaction_Score
        ]
        scaled_features = scaler.transform([numerical_features])
        final_input = np.append(scaled_features[0], therapy_encoded)

        health_status = int(health_model.predict([final_input])[0])
        readiness_score = float(readiness_model.predict([final_input])[0])

        return {"Health_Status": health_status, "Readiness_Score": readiness_score}

    except Exception as e:
        return {"error": str(e)}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)