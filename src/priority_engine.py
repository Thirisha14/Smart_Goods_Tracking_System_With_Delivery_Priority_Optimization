import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "amazon_delivery_with_priority_and_links.csv"
OUTPUT_PATH = BASE_DIR / "data" / "delivery_simulation_output.csv"

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"ERROR: Data not found at {DATA_PATH}")
    exit(1)

# Clean columns
df.columns = df.columns.str.strip()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# ─────────────────────────────────────────────
# FORCE PRIORITY (FOR DEMONSTRATION)
# ─────────────────────────────────────────────
n = len(df)
df.loc[0:int(n*0.3), "Priority_Level"] = "High"
df.loc[int(n*0.3):int(n*0.7), "Priority_Level"] = "Medium"
df.loc[int(n*0.7):, "Priority_Level"] = "Low"

print("\n--- Forced Priority Distribution ---")
print(df["Priority_Level"].value_counts())

# ─────────────────────────────────────────────
# PRIORITY SORTING (CORE LOGIC)
# ─────────────────────────────────────────────
priority_map = {"High": 0, "Medium": 1, "Low": 2}
df["Priority_Rank"] = df["Priority_Level"].map(priority_map)
df = df.sort_values(by="Priority_Rank").reset_index(drop=True)

print("\n--- Priority Scheduling Applied ---")
print(df[["Order_ID", "Priority_Level"]].head(10))

# ─────────────────────────────────────────────
# FLEET ALLOCATION & ESCALATION
# ─────────────────────────────────────────────
CAPACITY = {
    "bicycle": 1500,
    "motorcycle": 4500,
    "scooter": 3000,
    "van": 9000,
    "truck": 5000
}

ESCALATION_MIN = {
    "High": "van",
    "Medium": "motorcycle"
}

# Ladder for vehicle escalation
ladder = ["bicycle", "motorcycle", "scooter", "van", "truck"]
vehicle_load = {v: 0 for v in CAPACITY}

# ─────────────────────────────────────────────
# Low-priority allocation fraction
# ─────────────────────────────────────────────
LOW_PRIORITY_SHARE = 0.10  # reserve 10% of each vehicle capacity for Low-priority orders
low_capacity_reserved = {v: int(CAPACITY[v] * LOW_PRIORITY_SHARE) for v in CAPACITY}
low_delivered_count = {v: 0 for v in CAPACITY}

results = []

for _, row in df.iterrows():
    original_v = str(row.get("Vehicle", "motorcycle")).lower()
    priority = row["Priority_Level"]

    assigned_v = original_v
    escalated = False

    # 🔴 Priority escalation
    min_req = ESCALATION_MIN.get(priority)
    if min_req:
        orig_idx = ladder.index(original_v) if original_v in ladder else 1
        min_idx = ladder.index(min_req)
        if orig_idx < min_idx:
            assigned_v = min_req
            escalated = True

    # 🚚 Delivery based on capacity
    status = "Pending"
    
    if priority == "Low":
        # Check if reserved low capacity is available
        if low_delivered_count[assigned_v] < low_capacity_reserved[assigned_v]:
            status = "Delivered"
            vehicle_load[assigned_v] += 1
            low_delivered_count[assigned_v] += 1
    else:
        # High & Medium orders use remaining capacity
        if vehicle_load[assigned_v] < CAPACITY[assigned_v] - low_capacity_reserved[assigned_v]:
            status = "Delivered"
            vehicle_load[assigned_v] += 1

    results.append({
        "Order_ID": row["Order_ID"],
        "Priority_Level": priority,
        "Assigned_Vehicle": assigned_v,
        "Status": status,
        "Escalated": "Yes" if escalated else "No"
    })

# ─────────────────────────────────────────────
# SAVE OUTPUT
# ─────────────────────────────────────────────
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSUCCESS: Output saved to {OUTPUT_PATH}")