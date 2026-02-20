import pandas as pd
from pathlib import Path

# --- PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "amazon_delivery_with_priority_and_links.csv"
OUTPUT_PATH = BASE_DIR / "data" / "delivery_simulation_output.csv"

# Load Data
df = pd.read_csv(DATA_PATH)
df['Priority_Level'] = df['Priority_Level'].astype(str).str.strip()

# 1. OPTIMIZATION: URGENCY SORTING
p_rank = {"High": 0, "Medium": 1, "Low": 2}
df['sort_key'] = df['Priority_Level'].map(p_rank).fillna(3)
df = df.sort_values('sort_key').drop('sort_key', axis=1)

# 2. OPTIMIZATION: 
# We limit small vehicles to very few orders to show the logic working
CAPACITY = {
    "bicycle": 10,       # Only 10 orders total for all bikes
    "motorcycle": 50,    # Only 50 orders total for all motorcycles
    "scooter": 40,       # Only 40 orders total for all scooters
    "van": 5000          # Large capacity to act as the 'Escalation' destination
}

vehicle_load = {v: 0 for v in CAPACITY}
results = []

for _, row in df.iterrows():
    original_vehicle = str(row["Vehicle"]).lower().strip()
    priority = row["Priority_Level"]
    
    assigned_vehicle = original_vehicle
    status = "Queued"

    # Optimization Logic: Resource Escalation
    # If it's High Priority and the small vehicle is full, move it to a Van
    if priority == "High" and vehicle_load.get(original_vehicle, 0) >= CAPACITY.get(original_vehicle, 0):
        if vehicle_load.get("van", 0) < CAPACITY["van"]:
            assigned_vehicle = "van" 
    
    # Check if the chosen vehicle has space
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

# --- SUMMARY REPORT ---
delivered_count = output_df[output_df['Status'] == 'Delivered'].shape[0]
total_count = output_df.shape[0]
escalated_count = output_df[output_df['Escalated'] == 'Yes'].shape[0]

high_priority_success = output_df[(output_df['Priority'] == 'High') & (output_df['Status'] == 'Delivered')].shape[0]
high_total = output_df[output_df['Priority'] == 'High'].shape[0]

print("\n--- Final Optimization Summary ---")
print(f"Total Success Rate:          {(delivered_count/total_count)*100:.2f}%")
print(f"High Priority Success Rate:   {(high_priority_success/high_total)*100:.2f}%")
print(f"Orders Escalated to Van:     {escalated_count}")
print(f"Results saved to: {OUTPUT_PATH}")