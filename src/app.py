import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim

# --- CONFIG & PATHS ---
st.set_page_config(page_title="Smart Goods Tracker", layout="wide")
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
MODEL_PATH = ROOT_DIR / "data" / "trained_ga_model.pkl"
OUTPUT_PATH = ROOT_DIR / "data" / "delivery_simulation_output.csv"

# --- LOAD AI MODEL ---
@st.cache_resource
def load_ai_brain():
    if not MODEL_PATH.exists(): return None
    return joblib.load(MODEL_PATH)

model_data = load_ai_brain()

# --- DYNAMIC DATA SIMULATION ---
def get_area_name(lat, lon):
    try:
        geolocator = Nominatim(user_agent="smart_tracker")
        location = geolocator.reverse(f"{lat}, {lon}", timeout=3)
        return location.address.split(',')[0] if location else "Colombo Sector"
    except: return f"Zone {round(lat, 2)}"

def get_live_weather(lat, lon):
    idx = int((abs(lat) + abs(lon)) * 100) % 5
    return ["Stormy", "Sandstorms", "Windy", "Cloudy", "Sunny"][idx]

def get_traffic_status(lat, lon):
    idx = int((abs(lat) * abs(lon)) * 1000) % 4
    return ["Jam", "High", "Medium", "Low"][idx]

# --- UI TABS ---
st.title("📦 Smart Goods Tracking System")
tab1, tab2 = st.tabs(["📍 Live Map Inspector", "📊 Fleet Optimization"])

with tab1:
    col_map, col_res = st.columns([2, 1])
    with col_map:
        m = folium.Map(location=[6.9271, 79.8612], zoom_start=12)
        map_output = st_folium(m, height=500, width=700)
    
    with col_res:
        if map_output.get("last_clicked"):
            lat, lon = map_output["last_clicked"]["lat"], map_output["last_clicked"]["lng"]
            area, weather, traffic = get_area_name(lat, lon), get_live_weather(lat, lon), get_traffic_status(lat, lon)
            
            st.metric("Area", area)
            st.write(f"**Weather:** {weather} | **Traffic:** {traffic}")
            
            if model_data:
                # Prepare data for Genetic Algorithm 
                input_row = {"Type_of_vehicle": "motorcycle", "Type_of_order": "Snack", 
                            "Weather": weather, "Traffic": traffic, "Festival": "No", "City": "Metropolitian"}
                encoded = [model_data['feature_encoders'][f].transform([input_row[f]])[0] for f in model_data['features']]
                scores = np.dot([encoded], model_data['weights'])
                priority = model_data['target_encoder'].inverse_transform([np.argmax(scores)])[0]
                
                if priority == "High":
                    st.error(f"### {priority.upper()} PRIORITY")
                    st.warning("🚨 **Escalation Logic Triggered:** Moving to Van support.")
                else:
                    st.success(f"### {priority.upper()} Priority")
        else:
            st.info("Click the map to analyze a location.")

with tab2:
    if st.button("🚀 Run Global Optimization Engine"):
        import subprocess
        subprocess.run([sys.executable, str(CURRENT_DIR / "priority_engine.py")])
        st.rerun()
    
    if OUTPUT_PATH.exists():
        df = pd.read_csv(OUTPUT_PATH)
        st.metric("Delivery Success Rate", f"{(len(df[df['Status']=='Delivered'])/len(df))*100:.2f}%")
        st.dataframe(df.sample(50))