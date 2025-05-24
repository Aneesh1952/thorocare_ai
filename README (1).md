# API Integration
app = FastAPI()

@app.get("/predict_health")
def predict_health(heart_rate: float, respiration_rate: float, temperature: float, movement_activity: float, reaction_score: float):
    input_data = np.array([[heart_rate, respiration_rate, temperature, movement_activity, reaction_score]])
    input_data = scaler.transform(input_data)
    prediction = health_model.predict(input_data)
    return {"Predicted Health Status": int(prediction[0])}

@app.get("/predict_readiness")
def predict_readiness(heart_rate: float, respiration_rate: float, temperature: float, movement_activity: float, reaction_score: float):
    input_data = np.array([[heart_rate, respiration_rate, temperature, movement_activity, reaction_score]])
    input_data = scaler.transform(input_data)
    prediction = readiness_model.predict(input_data)
    return {"Predicted Readiness Score": float(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Dockerfile configuration
dockerfile_content = """
FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
with open("Dockerfile", "w") as f:
    f.write(dockerfile_content)

print("Dockerfile created successfully!")
