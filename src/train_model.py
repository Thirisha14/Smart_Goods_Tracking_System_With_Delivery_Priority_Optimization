import pandas as pd
import numpy as np
import random
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

# --- PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "amazon_delivery_with_priority.csv"
MODEL_PATH = BASE_DIR / "data" / "trained_ga_model.pkl"
CHART_PATH = BASE_DIR / "data" / "training_results_chart.png"

# --- LOAD & CLEAN ---
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# LOGIC: Force 'High' and 'Medium' labels so the model learns them
df.loc[(df['Traffic'] == 'Jam') & (df['Weather'] == 'Stormy'), 'Priority_Level'] = 'High'
df.loc[(df['Traffic'] == 'Jam') & (df['Weather'] != 'Stormy'), 'Priority_Level'] = 'Medium'
df.loc[df['Priority_Level'].isna(), 'Priority_Level'] = 'Low'

print("Class Counts:\n", df['Priority_Level'].value_counts())

# --- PREP DATA ---
features = ["Delivery_Time", "Traffic", "Weather", "Category"]
X = df[features].copy()
y = df['Priority_Level'].copy()

feature_encoders = {}
for col in X.columns:
    enc = LabelEncoder()
    X[col] = enc.fit_transform(X[col])
    feature_encoders[col] = enc

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# --- THE 80/20 SPLIT ---
# Stratify ensures both sets get a fair share of 'High' and 'Medium' labels
X_train, X_test, y_train, y_test = train_test_split(
    X.values, 
    y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

# --- GENETIC ALGORITHM ---
num_features, num_classes = X_train.shape[1], len(np.unique(y_encoded))
POP_SIZE, GENERATIONS = 100, 200

def fitness(weights):
    preds = np.argmax(np.dot(X_train, weights), axis=1)
    return balanced_accuracy_score(y_train, preds)

population = [np.random.randn(num_features, num_classes) for _ in range(POP_SIZE)]
history = []

print("\nTraining GA Model...")
for gen in range(GENERATIONS):
    population.sort(key=lambda w: fitness(w), reverse=True)
    history.append(fitness(population[0]))
    
    if (gen + 1) % 10 == 0: 
        print(f"Gen {gen+1}: Balanced Accuracy {history[-1]:.4f}")
    
    next_gen = population[:10]
    while len(next_gen) < POP_SIZE:
        p1, p2 = random.sample(population[:15], 2)
        child = p1.copy()
        mask = np.random.rand(*child.shape) > 0.5
        child[mask] = p2[mask]
        if random.random() < 0.2: 
            child += np.random.normal(0, 0.2, child.shape)
        next_gen.append(child)
    population = next_gen

# --- SAVE MODEL ---
joblib.dump({
    'weights': population[0], 
    'target_encoder': target_encoder, 
    'feature_encoders': feature_encoders, 
    'features': features
}, MODEL_PATH)

# --- GENERATE DIAGRAM (TESTING) ---
y_pred = np.argmax(np.dot(X_test, population[0]), axis=1)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=target_encoder.classes_, 
            yticklabels=target_encoder.classes_)
plt.title("Confusion Matrix: 20% Unseen Test Data")
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# This line saves the image to your disk
plt.savefig(CHART_PATH)
print(f"\nModel saved. Diagram saved at: {CHART_PATH}")