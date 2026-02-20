import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data" / "trained_ga_model.pkl"
DATA_PATH = BASE_DIR / "data" / "amazon_delivery_with_priority_and_links.csv"

model_data = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Clean input data
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

X_input = df[model_data['features']].copy()
for col, enc in model_data['feature_encoders'].items():
    X_input[col] = X_input[col].map(lambda s: enc.transform([s])[0] if s in enc.classes_ else 0)

# Predict
scores = np.dot(X_input.values, model_data['weights'])
df["Priority_Level"] = model_data['target_encoder'].inverse_transform(np.argmax(scores, axis=1))

print("\n--- Predicted Distribution ---")
print(df["Priority_Level"].value_counts())

df.to_csv(DATA_PATH, index=False)
print("Data updated successfully.")