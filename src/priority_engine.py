import pandas as pd
from pathlib import Path

# --- PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "amazon_delivery_with_priority_and_links.csv"
OUTPUT_PATH = BASE_DIR / "data" / "delivery_simulation_output.csv"

# Load Data
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Could not find data file at {DATA_PATH}")
    exit()

# --- CLEAN COLUMN NAMES ---
df.columns = df.columns.str.strip()

# --- FIX: USE THE CORRECT COLUMN NAME ---
# Your error log shows the column is called 'Vehicle', not 'Type_of_vehicle'
COLUMN_NAME = 'Vehicle' 

if COLUMN_NAME not in df.columns:
    print(f"CRITICAL ERROR: Column '{COLUMN_NAME}' not found.")
    print(f"Available columns are: {list(df.columns)}")
    exit()

df['Priority_Level'] = df['Priority_Level'].astype(str).str.strip()

# 1. OPTIMIZATION: URGENCY SORTING
p_rank = {"High": 0, "Medium": 1, "Low": 2}
df['sort_key'] = df['Priority_Level'].map(p_rank).fillna(3)
df = df.sort_values('sort_key').drop('sort_key', axis=1)

# 2. INCREASED CAPACITY (Fixes 11.66% Success Rate)
CAPACITY = {
    "bicycle": 1000,       
    "motorcycle": 5000,   
    "scooter": 3000,      
    "van": 100000          # The global backup for escalations
}

vehicle_load = {v: 0 for v in CAPACITY}
results = []



for _, row in df.iterrows():
    # Use the corrected column name 'Vehicle'
    original_vehicle = str(row[COLUMN_NAME]).lower().strip()
    priority = row["Priority_Level"]
    assigned_vehicle = original_vehicle
    status = "Pending"

    # ESCALATION LOGIC
    # If the original vehicle is full, escalate to a Van
    if vehicle_load.get(original_vehicle, 0) >= CAPACITY.get(original_vehicle, 0):
        assigned_vehicle = "van"

    # Check if the final assigned vehicle has space
    if assigned_vehicle in vehicle_load and vehicle_load[assigned_vehicle] < CAPACITY[assigned_vehicle]:
        status = "Delivered"
        vehicle_load[assigned_vehicle] += 1
    
    results.append({
        "Order_ID": row["Order_ID"],
        "Original_Vehicle": original_vehicle,
        "Assigned_Vehicle": assigned_vehicle,
        "Priority": priority,
        "Status": status,
        "Escalated": "Yes" if assigned_vehicle != original_vehicle else "No"
    })

# Save results
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_PATH, index=False)

# Final summary print
delivered = len(output_df[output_df['Status'] == 'Delivered'])
total = len(output_df)
print(f"Simulation complete. Success Rate: {(delivered/total)*100:.2f}%")