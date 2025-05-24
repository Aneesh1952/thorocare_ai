# ThoroCareAI API

## Overview
This FastAPI-based API predicts the health status and readiness score of Thoroughbred horses based on biometric and behavioral data. The models used are trained XGBoost models.

## Installation

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Start the API:
   ```sh
   uvicorn thorocare_api:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Usage

### Endpoint: `/predict`
- **Method:** `POST`
- **Request Body (JSON):**
  ```json
  {
    "Heart_Rate": 42,
    "Respiration_Rate": 19,
    "Temperature": 37.2,
    "Movement_Activity": 6,
    "Reaction_Score": 3
  }
  ```
- **Response:**
  ```json
  {
    "Health_Status": 1,
    "Readiness_Score": 0.59
  }
  ```

## Dependencies
Ensure that `health_model.pkl`, `readiness_model.pkl`, and `scaler.pkl` are available in the working directory.

## Notes
- This API is designed for research and monitoring purposes.
- Ensure that all dependencies are installed before running the server.
- For deployment, consider using Docker or a cloud service provider.
