import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor

# Load dataset (ensure your CSV file is named 'horse_data.csv' and in the same directory)
df = pd.read_csv("thorocare_3000_dataset.csv")

# Display first few rows for debugging
print(df.head())

# Define feature columns and target column
num_features = ["Heart_Rate", "Respiration_Rate", "Temperature", "Movement_Activity", "Reaction_Score"]
cat_feature = "Therapy_Readiness"
target = "Readiness_Score"

# Encode the categorical feature
le = LabelEncoder()
df["Therapy_Readiness_encoded"] = le.fit_transform(df[cat_feature])

# Scale numerical features
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(df[num_features])

# Combine scaled numerical features with the encoded categorical feature.
# This follows the same order used later in your FastAPI prediction endpoint.
X = np.hstack([scaled_nums, df[["Therapy_Readiness_encoded"]].values])
y = df[target].values

# Initialize and train the XGBoost regressor
model = XGBRegressor(objective="reg:squarederror", random_state=42)
model.fit(X, y)

# Save the trained model to disk
joblib.dump(model, "readiness.pkl")

# Optionally, save the scaler and the label encoder (or a dict with encoders) for consistent preprocessing during inference.
joblib.dump(scaler, "scaler.pkl")
joblib.dump({"Therapy_Readiness": le}, "label_encoders.pkl")

print("Readiness model saved as 'readiness.pkl'")
