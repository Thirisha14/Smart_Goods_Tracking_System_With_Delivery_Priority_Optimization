import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings("ignore")

# 1. Load your actual dataset
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "amazon_delivery_with_priority.csv")
df = pd.read_csv(data_path)

# --- CLEANING STEP ---
# Strip whitespace from columns and string data to avoid 'High' vs 'High ' errors
df.columns = df.columns.str.strip()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# 2. Basic Pre-processing
features = ['Delivery_Time', 'Traffic', 'Weather', 'Category']
X = df[features].copy()
y = df['Priority_Level']

# Use separate encoders for each column
feature_encoders = {}
for col in features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    feature_encoders[col] = le

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Get the internal integer values for labels
# We use a loop to safely find the index even if labels are shuffled
labels_list = list(target_encoder.classes_)
high_val = labels_list.index('High') if 'High' in labels_list else 0
low_val = labels_list.index('Low') if 'Low' in labels_list else 1

# Split data (same 80/20 split as your GA)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("--- STARTING BENCHMARK TEST ---")

# --- METHOD 1: STATIC HEURISTIC (Manual Rule) ---
# Logic: If Traffic is 'Jam', predict 'High'. Otherwise predict 'Low'.
if 'Traffic' in feature_encoders and 'Jam' in list(feature_encoders['Traffic'].classes_):
    jam_encoded_val = list(feature_encoders['Traffic'].classes_).index('Jam')
    y_heuristic = np.where(X_test['Traffic'] == jam_encoded_val, high_val, low_val)

    print("\n[1] STATIC HEURISTIC RESULTS (Rule-Based):")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_heuristic):.4f}")
    print(f"High Priority Recall: {recall_score(y_test, y_heuristic, labels=[high_val], average='macro'):.4f}")
else:
    print("\n[1] STATIC HEURISTIC: Skip (Traffic 'Jam' label not found)")


# --- METHOD 2: LOGISTIC REGRESSION (Standard ML) ---
lr_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\n[2] LOGISTIC REGRESSION RESULTS (Standard ML):")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_lr):.4f}")
print(f"High Priority Recall: {recall_score(y_test, y_pred_lr, labels=[high_val], average='macro'):.4f}")


# --- METHOD 3: YOUR GA RESULTS (From your previous logs) ---
print("\n[3] PROPOSED GA MODEL (Actual Results):")
print("Balanced Accuracy: 0.6161")
print("High Priority Recall: 0.9800")


# --- METHOD 4: ABLATION TEST (Random Weights / Untrained AI) ---
# This simulates the GA model at 'Generation 0' before evolution.
# We generate random weights between -1 and 1 for the 4 features.
np.random.seed(42)
random_weights = np.random.uniform(-1, 1, (len(features), len(target_encoder.classes_)))

# Simple dot product to simulate an untrained neural/weight layer
random_logits = np.dot(X_test, random_weights)
y_pred_random = np.argmax(random_logits, axis=1)

print("\n[4] ABLATION STUDY (Untrained / Random Weights):")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_random):.4f}")
print(f"High Priority Recall: {recall_score(y_test, y_pred_random, labels=[high_val], average='macro'):.4f}")