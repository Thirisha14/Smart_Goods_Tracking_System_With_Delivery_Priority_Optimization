import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
# Use the updated Excel file for testing
DATA_PATH = BASE_DIR / "data" / "amazon_delivery_with_priority_and_links.xlsx"  # Updated to .xlsx
OUTPUT_PATH = BASE_DIR / "data" / "delivery_simulation_output.csv"

# ─────────────────────────────────────────────
# LOAD & CLEAN
# ─────────────────────────────────────────────
try:
    df = pd.read_excel(DATA_PATH)  # Read Excel instead of CSV
except FileNotFoundError:
    print(f"ERROR: Data not found at {DATA_PATH}. Please ensure you use the updated Excel file.")
    exit(1)

# Clean column names and data
df.columns = df.columns.str.strip()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# Ensure Order_ID is a string and clean
if 'Order_ID' in df.columns:
    df['Order_ID'] = df['Order_ID'].astype(str).str.strip()
else:
    print("ERROR: 'Order_ID' column not found in dataset.")
    exit(1)

# Add Status column if missing
if 'Status' not in df.columns:
    df['Status'] = 'Pending'

# ─────────────────────────────────────────────
# STRESS TEST PRIORITY RULES (FORCED)
# ─────────────────────────────────────────────
severe_weather = ["Stormy", "Rainy", "Sandstorms", "Windy", "Fog"]
heavy_traffic = ["Jam", "High"]

# Rule 1: High Priority (Severe Weather + Heavy Traffic)
df.loc[
    (df["Traffic"].isin(heavy_traffic)) & 
    (df["Weather"].isin(severe_weather)), 
    "Priority_Level"
] = "High"

# Rule 2: Medium Priority (Either Severe Weather OR Heavy Traffic, but not both)
df.loc[
    ((df["Traffic"].isin(heavy_traffic)) & (~df["Weather"].isin(severe_weather))) |
    ((~df["Traffic"].isin(heavy_traffic)) & (df["Weather"].isin(severe_weather))),
    "Priority_Level"
] = "Medium"

# Rule 3: Low Priority (Clear skies and Low Traffic)
df.loc[
    (~df["Traffic"].isin(heavy_traffic)) & 
    (~df["Weather"].isin(severe_weather)), 
    "Priority_Level"
] = "Low"

print("\n--- Corrected Stress Test Distribution ---")
print(df["Priority_Level"].value_counts())

# ─────────────────────────────────────────────
# FLEET ALLOCATION & ESCALATION
# ─────────────────────────────────────────────
CAPACITY = {"bicycle": 10000, 
            "motorcycle": 50000, 
            "scooter": 30000, 
            "van": 10000, 
            "truck": 50000}

ESCALATION_MIN = {"High": "van", "Medium": "scooter"}
ladder = ["bicycle", "motorcycle", "scooter", "van", "truck"]
vehicle_load = {v: 0 for v in CAPACITY}
results = []

for i, row in df.iterrows():
    original_v = str(row.get("Vehicle", "motorcycle")).lower()
    priority = row["Priority_Level"]
    assigned_v = original_v
    escalated = False

    # 1. Priority-Based Escalation
    min_req = ESCALATION_MIN.get(priority)
    if min_req:
        orig_idx = ladder.index(original_v) if original_v in ladder else 1
        min_idx = ladder.index(min_req)
        if orig_idx < min_idx:
            assigned_v = min_req
            escalated = True

    # 2. Capacity Check (Overflow to Van)
    if vehicle_load.get(assigned_v, 0) >= CAPACITY.get(assigned_v, 0):
        assigned_v = "van"
        escalated = True

    # 3. Status Check
    status = "Delivered" if vehicle_load.get(assigned_v, 0) < CAPACITY.get(assigned_v, 0) else "Pending"
    if status == "Delivered":
        vehicle_load[assigned_v] += 1

    results.append({
        "Order_ID": row.get("Order_ID", f"ORD-{i}"),
        "Original_Vehicle": original_v,
        "Assigned_Vehicle": assigned_v,
        "Priority": priority,
        "Status": status,
        "Escalated": "Yes" if escalated else "No"
    })

# Save output for Dashboard
pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
print(f"\nSUCCESS: Simulation output saved to {OUTPUT_PATH}")