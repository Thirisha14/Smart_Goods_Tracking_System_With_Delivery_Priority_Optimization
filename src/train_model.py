import pandas as pd
import numpy as np
import random
import joblib
import matplotlib
matplotlib.use("Agg")

import sys
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "amazon_delivery_with_priority_and_links.csv"
MODEL_PATH = BASE_DIR / "data" / "trained_ga_model.pkl"
CHART_PATH = BASE_DIR / "data" / "training_results_chart.png"

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
if not DATA_PATH.exists():
    print("ERROR: Dataset not found:", DATA_PATH)
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

print("Loaded", len(df), "rows")
print("Columns:", list(df.columns))

# ─────────────────────────────────────────────
# NORMALIZE TEXT VALUES
# ─────────────────────────────────────────────
if "Traffic" in df.columns:
    df["Traffic"] = df["Traffic"].str.title()

if "Weather" in df.columns:
    df["Weather"] = df["Weather"].str.title()

print("\nTraffic values:", df["Traffic"].unique())
print("Weather values:", df["Weather"].unique())

print("\nTraffic vs Weather table:")
print(pd.crosstab(df["Traffic"], df["Weather"]))

# ─────────────────────────────────────────────
# PRIORITY LABELLING
# ─────────────────────────────────────────────
df["Priority_Level"] = "Low"

# HIGH PRIORITY
df.loc[
    (df["Traffic"].isin(["Jam", "High"])) &
    (df["Weather"].isin(["Stormy", "Rainy"])),
    "Priority_Level"
] = "High"

# MEDIUM PRIORITY
df.loc[
    (df["Traffic"].isin(["Jam", "High"])) |
    (df["Weather"].isin(["Stormy", "Rainy"])),
    "Priority_Level"
] = "Medium"

print("\nPriority distribution:")
print(df["Priority_Level"].value_counts())

# ─────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────
all_candidates = ["Delivery_Time", "Traffic", "Weather", "Category"]
features = [f for f in all_candidates if f in df.columns]

if not features:
    print("ERROR: No valid training features found")
    sys.exit(1)

print("\nTraining features:", features)

X = df[features].copy()
y = df["Priority_Level"].copy()

# ─────────────────────────────────────────────
# ENCODE FEATURES
# ─────────────────────────────────────────────
feature_encoders = {}

for col in X.columns:
    enc = LabelEncoder()
    X[col] = enc.fit_transform(X[col].astype(str))
    feature_encoders[col] = enc
    print(col, "classes:", list(enc.classes_))

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print("\nTarget classes:", list(target_encoder.classes_))

# ─────────────────────────────────────────────
# TRAIN TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X.values,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain size:", len(X_train))
print("Test size :", len(X_test))

# ─────────────────────────────────────────────
# GENETIC ALGORITHM
# ─────────────────────────────────────────────
num_features = X_train.shape[1]
num_classes  = len(np.unique(y_encoded))

POP_SIZE = 100
GENERATIONS = 200

def fitness(weights):
    preds = np.argmax(np.dot(X_train, weights), axis=1)
    return balanced_accuracy_score(y_train, preds)

population = [
    np.random.randn(num_features, num_classes)
    for _ in range(POP_SIZE)
]

history = []

print("\nTraining Genetic Algorithm...")
print("-" * 40)

for gen in range(GENERATIONS):

    population.sort(key=lambda w: fitness(w), reverse=True)
    best_score = fitness(population[0])
    history.append(best_score)

    if (gen + 1) % 10 == 0:
        print(
            "Gen", gen + 1,
            "| Balanced Accuracy:", round(best_score, 4)
        )

    # Elitism
    next_gen = population[:10]

    # Crossover + Mutation
    while len(next_gen) < POP_SIZE:

        p1, p2 = random.sample(population[:20], 2)

        child = p1.copy()
        mask = np.random.rand(*child.shape) > 0.5
        child[mask] = p2[mask]

        if random.random() < 0.2:
            child += np.random.normal(0, 0.2, child.shape)

        next_gen.append(child)

    population = next_gen

best_weights = population[0]
final_acc = history[-1]

print("\nFinal Train Balanced Accuracy:", round(final_acc, 4))

# ─────────────────────────────────────────────
# TEST EVALUATION
# ─────────────────────────────────────────────
y_pred = np.argmax(np.dot(X_test, best_weights), axis=1)

test_acc = balanced_accuracy_score(y_test, y_pred)

print("\nTest Balanced Accuracy:", round(test_acc, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_
    )
)

# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "weights": best_weights,
    "target_encoder": target_encoder,
    "feature_encoders": feature_encoders,
    "features": features,
    "history": history,
    "test_accuracy": test_acc,
}, MODEL_PATH)

print("\nModel saved ->", MODEL_PATH)

# ─────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0d1625")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_encoder.classes_,
    yticklabels=target_encoder.classes_,
    ax=axes[0]
)

axes[0].set_title("Confusion Matrix", color="white")
axes[0].tick_params(colors="white")
axes[0].set_facecolor("#0d1625")

# Training Curve
axes[1].plot(history, linewidth=2)
axes[1].set_title("GA Training Curve", color="white")
axes[1].set_xlabel("Generation")
axes[1].set_ylabel("Balanced Accuracy")
axes[1].tick_params(colors="white")
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=120, facecolor="#0d1625")

print("Charts saved ->", CHART_PATH)