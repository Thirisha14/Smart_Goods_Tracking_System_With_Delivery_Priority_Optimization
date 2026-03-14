import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data" / "trained_ga_model.pkl"

# FIX: Use the same CSV that train_model.py reads.
# The old code pointed at "amazon_delivery_with_priority_and_links.csv"
# which is a different (and possibly missing) file.
DATA_PATH  = BASE_DIR / "data" / "amazon_delivery_with_priority_and_links.csv"

# ── GUARD CLAUSES ──────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Run train_model.py first to generate the model.")
    exit(1)

if not DATA_PATH.exists():
    print(f"ERROR: Dataset not found at {DATA_PATH}")
    exit(1)

# ── LOAD ───────────────────────────────────────────────────────────────────────
model_data = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

print(f"Loaded model  -> features: {model_data['features']}")
print(f"Loaded data   -> {len(df):,} rows")

# Clean strings
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# ── ENCODE ─────────────────────────────────────────────────────────────────────
features = model_data["features"]
encoders = model_data["feature_encoders"]

X_input = df[features].copy()

for col in features:
    enc   = encoders[col]
    known = set(enc.classes_)
    # FIX: The old code used .map(lambda) which raises ValueError for unseen labels.
    # This version safely falls back to 0 for any value not seen during training.
    X_input[col] = X_input[col].apply(
        lambda v: int(enc.transform([v])[0]) if v in known else 0
    )

# ── PREDICT ────────────────────────────────────────────────────────────────────
scores = np.dot(X_input.values, model_data["weights"])
df["Priority_Level"] = model_data["target_encoder"].inverse_transform(
    np.argmax(scores, axis=1)
)

print("\n--- Predicted Priority Distribution ---")
print(df["Priority_Level"].value_counts())
print(f"\nPct High   : {(df['Priority_Level']=='High').mean()*100:.1f}%")
print(f"Pct Medium : {(df['Priority_Level']=='Medium').mean()*100:.1f}%")
print(f"Pct Low    : {(df['Priority_Level']=='Low').mean()*100:.1f}%")

# ── SAVE ───────────────────────────────────────────────────────────────────────
df.to_csv(DATA_PATH, index=False)
print(f"\nData updated with Priority_Level column -> {DATA_PATH}")