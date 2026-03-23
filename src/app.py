import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import subprocess
import requests
import json
from pathlib import Path
from datetime import datetime
from scanner import render_qr_scanner_page
from notifications import render_notification_board

# ── streamlit_folium / folium ──────────────────────────────────────────────────
try:
    from streamlit_folium import st_folium
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ── geopy ─────────────────────────────────────────────────────────────────────
try:
    from geopy.geocoders import Nominatim
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

# --- NEW: LOGIN HELPER FUNCTION ---
def load_users():
    user_file = Path("data/users.json")
    if user_file.exists():
        with open(user_file, "r") as f:
            return json.load(f)
    # Default fallback if file is missing
    return {"admin1": {"password": "adminpassword", "role": "Admin", "name": "System Admin"}}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DeliveryIQ — Sri Lanka",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)
                   
# --- NEW: AUTHENTICATION CHECK ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ── 1. LOGIN INTERFACE (SHOWN FIRST) ──────────────────────────────────────────
if not st.session_state.authenticated:
    _, col2, _ = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("🔐 DeliveryIQ Admin Portal")
        st.subheader("Internal Management Access")
        
        with st.form("login_form"):
            username = st.text_input("Administrator ID")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Access System", use_container_width=True)
            
            if submit:
                users = load_users()
                if username in users and users[username]["password"] == password:
                    if users[username]["role"] == "Admin":
                        st.session_state.authenticated = True
                        st.session_state.user_name = users[username]["name"]
                        st.rerun()
                    else:
                        st.error("Access Denied: Admin Privileges Required.")
                else:
                    st.error("Invalid Admin ID or Password.")
    st.stop() # Stops execution here so nothing below is shown
             
# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
SRC_DIR     = Path(__file__).resolve().parent
ROOT_DIR    = SRC_DIR.parent
DATA_DIR    = ROOT_DIR / "data"
MODEL_PATH  = DATA_DIR / "trained_ga_model.pkl"
OUTPUT_PATH = DATA_DIR / "delivery_simulation_output.csv"
CHART_PATH  = DATA_DIR / "training_results_chart.png"

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS  — dark professional theme
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: #0a1220 !important;
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: transparent;
    border: 1px solid #1e3a5f;
    color: #94a3b8 !important;
    border-radius: 8px;
    text-align: left;
    padding: 10px 14px;
    margin-bottom: 4px;
    font-size: 14px;
    transition: all 0.2s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #1e3a5f !important;
    border-color: #3b82f6 !important;
    color: #e2e8f0 !important;
}

/* ── main background ── */
.main .block-container {
    background: #070e1c;
    padding-top: 16px;
}
.stApp { background: #070e1c; }

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: #0d1625;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 800; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 11px !important; letter-spacing: 2px; }

/* ── headings ── */
h1 { color: #f1f5f9 !important; font-weight: 800 !important; }
h2, h3 { color: #e2e8f0 !important; font-weight: 700 !important; }

/* ── dataframe ── */
.stDataFrame { border: 1px solid #1e3a5f !important; border-radius: 10px; }

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 20px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #60a5fa) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px #3b82f640;
}

/* ── selectbox / input ── */
.stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput > div > div > input {
    background: #0d1625 !important;
    border-color: #1e3a5f !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #0a1220; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; border-radius: 8px; }
.stTabs [aria-selected="true"] { background: #1e3a5f !important; color: #f1f5f9 !important; }

/* ── priority boxes ── */
.priority-high   { background:#1c0505; border:1.5px solid #ef4444; border-radius:12px; padding:20px; }
.priority-medium { background:#1c0c00; border:1.5px solid #f97316; border-radius:12px; padding:20px; }
.priority-low    { background:#001c08; border:1.5px solid #22c55e; border-radius:12px; padding:20px; }

.stat-card {
    background: #0d1625;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
}

/* ── info/warning/error override ── */
.stAlert { border-radius: 10px !important; }

/* ── dividers ── */
hr { border-color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)
    
# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════
for key, val in {
    "page":          "dashboard",
    "last_lat":      None,
    "last_lon":      None,
    "last_district": None,
    "scan_log":      [],
    "last_result":   None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

model_data = load_model()

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
WMO_MAP = {
    range(0,1):   ("Sunny",   "☀️"),
    range(1,3):   ("Cloudy",  "⛅"),
    range(3,4):   ("Cloudy",  "☁️"),
    range(45,50): ("Foggy",   "🌫️"),
    range(51,68): ("Drizzle", "🌦️"),
    range(61,68): ("Rainy",   "🌧️"),
    range(80,83): ("Rainy",   "🌧️"),
    range(95,100):("Stormy",  "⛈️"),
}

def wmo_to_weather(code, wind):
    if code >= 95: return "Stormy", "⛈️"
    if code >= 80: return "Rainy",   "🌧️"
    if code >= 61: return "Rainy",   "🌧️"
    if code >= 51: return "Drizzle", "🌦️"
    if code >= 45: return "Foggy",   "🌫️"
    if code >= 3:  return "Cloudy",  "☁️"
    if wind > 28:  return "Windy",   "💨"
    return "Sunny", "☀️"

def fetch_live_weather(lat, lon):
    """Fetch real weather from Open-Meteo (free, no key needed)."""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=weather_code,wind_speed_10m,precipitation,temperature_2m"
            f"&timezone=Asia/Colombo"
        )
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            cur     = r.json()["current"]
            label, icon = wmo_to_weather(cur["weather_code"], cur["wind_speed_10m"])
            return {
                "label":    label,
                "icon":     icon,
                "temp":     cur["temperature_2m"],
                "wind":     cur["wind_speed_10m"],
                "precip":   cur["precipitation"],
                "live":     True,
            }
    except Exception:
        pass
    # Deterministic fallback (no randomness — stable across reruns)
    options = ["Sunny","Cloudy","Rainy","Stormy","Windy","Drizzle"]
    icons   = ["☀️",   "☁️",    "🌧️",   "⛈️",    "💨",   "🌦️"]
    idx     = int((abs(lat) + abs(lon)) * 97) % len(options)
    return {"label": options[idx], "icon": icons[idx], "temp": None, "wind": None, "precip": None, "live": False}

def get_traffic(lat, lon):
    """Simulate realistic time-aware traffic."""
    h    = datetime.now().hour
    peak = (7 <= h <= 9) or (17 <= h <= 19)
    # Urban coords (Colombo area)
    urban = (6.5 <= lat <= 7.4) and (79.7 <= lon <= 80.3)
    base  = 40 if urban else 15
    score = min(99, base + (30 if peak else 0) + abs(hash(f"{round(lat,2)}{round(lon,2)}")) % 30)
    if score >= 72: return {"label":"Jam",    "score":score, "color":"#ef4444", "icon":"🚨", "desc":"Traffic Jam"   }
    if score >= 50: return {"label":"High",   "score":score, "color":"#f97316", "icon":"🚦", "desc":"High Traffic"  }
    if score >= 30: return {"label":"Medium", "score":score, "color":"#eab308", "icon":"⚠️", "desc":"Moderate Flow" }
    return               {"label":"Low",    "score":score, "color":"#22c55e", "icon":"✅", "desc":"Free Flow"     }

def get_area_name(lat, lon):
    if HAS_GEOPY:
        try:
            geo = Nominatim(user_agent="deliveryiq_lk_v3")
            loc = geo.reverse(f"{lat},{lon}", timeout=4)
            if loc:
                parts = [p.strip() for p in loc.address.split(",") if p.strip()]
                return ", ".join(parts[:2])
        except Exception:
            pass
    return f"({round(lat,3)}°N, {round(lon,3)}°E)"

def rule_priority(traffic_label, weather_label):

    # HIGH PRIORITY
    if traffic_label == "Jam" and weather_label in ("Stormy", "Rainy"):
        return "High"

    if traffic_label == "High" and weather_label in ("Stormy", "Rainy"):
        return "High"

    # MEDIUM PRIORITY
    if traffic_label == "Jam":
        return "Medium"

    # LOW PRIORITY
    return "Low"

def model_priority(weather_label, traffic_label, delivery_time=45, category="Grocery"):
    """Use trained GA model weights if available, else fall back to rules."""
    if model_data is None:
        return rule_priority(traffic_label, weather_label)
    features  = model_data["features"]
    encoders  = model_data["feature_encoders"]
    defaults  = {
        "Delivery_Time": str(delivery_time),
        "Traffic":       traffic_label,
        "Weather":       weather_label,
        "Category":      category,
    }
    encoded = []
    for f in features:
        enc = encoders[f]
        val = str(defaults.get(f, ""))
        encoded.append(int(enc.transform([val])[0]) if val in enc.classes_ else 0)
    scores = np.dot([encoded], model_data["weights"])
    return model_data["target_encoder"].inverse_transform([np.argmax(scores)])[0]

VEHICLE_LADDER = ["bicycle", "motorcycle", "scooter", "van", "truck"]

def escalate(priority, vehicle):
    v = vehicle.lower()
    idx = VEHICLE_LADDER.index(v) if v in VEHICLE_LADDER else 1
    if priority == "High"   and idx < 3: return "van",    True,  "High priority — escalated to Van"
    if priority == "Medium" and idx < 2: return "scooter",True,  "Medium priority — upgraded to Scooter"
    return v, False, "No escalation — current vehicle sufficient"

PRI_COLOR = {"High":"#ef4444", "Medium":"#f97316", "Low":"#22c55e"}
PRI_BG    = {"High":"#1c0505", "Medium":"#1c0c00", "Low":"#001c08"}
PRI_EMOJI = {"High":"🔴",      "Medium":"🟡",      "Low":"🟢"}

# Add a logout button to your existing sidebar logic
st.sidebar.markdown(f"**Admin:** {st.session_state.user_name}")
if st.sidebar.button("🔓 Logout"):
    st.session_state.authenticated = False
    st.rerun()
    
# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

# 1. Initialize the session state if it doesn't exist (Put this at the top of main)
if 'page' not in st.session_state:
    st.session_state.page = "dashboard"

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 4px 24px;">
        <div style="font-size:22px;font-weight:900;color:#f1f5f9;letter-spacing:1px;">📦 DeliveryIQ</div>
        <div style="font-size:10px;color:#334155;letter-spacing:3px;margin-top:2px;">SRI LANKA · PRIORITY ENGINE</div>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        ("dashboard",   "🏠",  "Dashboard"),
        ("map",         "🗺️",  "Map Inspector"),
        ("parcel",      "📋",  "Parcel Entry"),
        ("scan","📷","Scan QR Delivery"),
        ("fleet",       "🚚",  "Fleet Optimization"),
        ("about",       "⚙️",  "How It Works"),
    ]
    for pid, icon, label in pages:
        is_active = st.session_state.page == pid
        style_override = "background:#1e3a5f!important;border-color:#3b82f6!important;color:#f1f5f9!important;" if is_active else ""
        if st.button(f"{icon}  {label}", key=f"nav_{pid}"):
            st.session_state.page = pid
            st.rerun()

    st.markdown("<hr style='border-color:#1e3a5f;margin:16px 0'/>", unsafe_allow_html=True)

    # Model status
    if model_data:
        st.markdown("<div style='font-size:10px;color:#22c55e;letter-spacing:2px;'>● GA MODEL LOADED</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:10px;color:#ef4444;letter-spacing:2px;'>● MODEL NOT FOUND</div>", unsafe_allow_html=True)
        st.caption("Run `train_model.py` first")

    st.markdown(f"<div style='font-size:9px;color:#334155;margin-top:8px;'>{datetime.now().strftime('%a %d %b · %H:%M')}</div>", unsafe_allow_html=True)
    
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def render_dashboard():
    st.markdown("## 🏠 Operations Dashboard")
    st.markdown("<div style='color:#64748b;font-size:14px;margin-bottom:24px;'>Real-time delivery priority monitoring across Sri Lanka</div>", unsafe_allow_html=True)

    log = st.session_state.scan_log
    total  = len(log)
    high   = sum(1 for x in log if x["priority"]=="High")
    medium = sum(1 for x in log if x["priority"]=="Medium")
    low    = sum(1 for x in log if x["priority"]=="Low")
    esc    = sum(1 for x in log if x.get("escalated", False))

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("📡 Zones Scanned",    total)
    c2.metric("🔴 High Priority",    high,   help="Jam + Stormy")
    c3.metric("🟡 Medium Priority",  medium, help="Jam or Stormy")
    c4.metric("🟢 Low Priority",     low,    help="Normal conditions")
    c5.metric("🚐 Escalations",      esc,    help="Vehicle upgrades triggered")

    st.markdown("<br/>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("#### 📊 Priority Distribution")
        if total == 0:
            st.info("No zones scanned yet. Go to **Map Inspector** and click on Sri Lanka to begin.")
        else:
            for label, count in [("High",high),("Medium",medium),("Low",low)]:
                pct = (count/total*100) if total else 0
                color = PRI_COLOR[label]
                st.markdown(f"""
                    <div style="margin-bottom:12px;">
                        <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;">
                            <span style="color:{color};font-weight:700;">{PRI_EMOJI[label]} {label}</span>
                            <span style="color:#64748b;">{count} orders · {pct:.0f}%</span>
                        </div>
                        <div style="height:8px;background:#0a1525;border-radius:4px;overflow:hidden;border:1px solid #1e3a5f;">
                            <div style="height:100%;width:{pct}%;background:{color};border-radius:4px;box-shadow:0 0 8px {color}60;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br/>", unsafe_allow_html=True)

                # Weather breakdown
                from collections import Counter
                weather_counts = Counter(x["weather"] for x in log)
                st.markdown("#### 🌤️ Weather Conditions Encountered")
                w_icons = {"Sunny":"☀️","Cloudy":"☁️","Rainy":"🌧️","Stormy":"⛈️","Windy":"💨","Foggy":"🌫️","Drizzle":"🌦️"}
                for w, cnt in weather_counts.most_common():
                    pct = cnt/total*100
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:13px;">
                        <span style="width:80px;color:#94a3b8;">{w_icons.get(w,'🌡️')} {w}</span>
                        <div style="flex:1;height:6px;background:#0a1525;border-radius:3px;overflow:hidden;">
                            <div style="height:100%;width:{pct}%;background:#3b82f6;border-radius:3px;"></div>
                        </div>
                        <span style="color:#64748b;width:30px;text-align:right;">{cnt}</span>
                    </div>
                    """, unsafe_allow_html=True)

        with col_right:
            st.markdown("#### 🕐 Recent Scan Activity")
            if not log:
                st.markdown("<div style='color:#334155;font-size:13px;padding:20px;text-align:center;'>No activity yet</div>", unsafe_allow_html=True)
            else:
                for entry in log[:10]:
                    color = PRI_COLOR[entry["priority"]]
                    esc_badge = " 🚐" if entry.get("escalated") else ""
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;padding:10px 12px;
                                background:#0d1625;border:1px solid {color}30;border-radius:8px;margin-bottom:6px;">
                        <div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0;box-shadow:0 0 6px {color};"></div>
                        <div style="flex:1;min-width:0;">
                            <div style="font-size:12px;color:#cbd5e1;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                                {entry.get("area","Unknown Area")}
                            </div>
                            <div style="font-size:10px;color:#475569;">{entry["weather"]} · {entry["traffic"]["desc"]}</div>
                        </div>
                        <div style="text-align:right;flex-shrink:0;">
                            <div style="font-size:11px;font-weight:700;color:{color};">{entry['priority']}{esc_badge}</div>
                            <div style="font-size:9px;color:#334155;">{entry["time"]}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def render_map_inspector():
    st.title("🗺️ Map Inspector")
    st.write("Map interface would render here.")

def render_parcel_entry():
    st.title("📋 Parcel Entry")
    st.write("Manual entry form would render here.")
    
render_notification_board()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MAP INSPECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def render_map_inspector():
    st.markdown("## 🗺️ Live Map Inspector")
    st.markdown("<div style='color:#64748b;font-size:14px;margin-bottom:16px;'>Click anywhere on Sri Lanka to get live weather, traffic, and delivery priority prediction.</div>", unsafe_allow_html=True)

    if not HAS_FOLIUM:
        st.error("📦 Install required: `pip install folium streamlit-folium`")
        st.stop()

    col_map, col_panel = st.columns([1.7, 1])

    with col_map:
        # Build folium map
        m = folium.Map(
            location=[7.8731, 80.7718],
            zoom_start=8,
            tiles="CartoDB dark_matter",
        )

        # Add existing scan markers
        for entry in st.session_state.scan_log:
            if entry.get("lat") and entry.get("lon"):
                color_map = {"High":"red","Medium":"orange","Low":"green"}
                folium.CircleMarker(
                    location=[entry["lat"], entry["lon"]],
                    radius=7,
                    color=color_map.get(entry["priority"],"blue"),
                    fill=True,
                    fill_opacity=0.8,
                    popup=f"{entry.get('area','')} — {entry['priority']} Priority",
                    tooltip=f"{entry['priority']}",
                ).add_to(m)

        # Current selection marker
        if st.session_state.last_lat:
            folium.Marker(
                location=[st.session_state.last_lat, st.session_state.last_lon],
                icon=folium.Icon(color="blue", icon="map-marker", prefix="fa"),
                tooltip="Selected zone",
            ).add_to(m)

        map_out = st_folium(m, height=520, use_container_width=True)

        # Detect new click
        clicked = map_out.get("last_clicked")
        if clicked:
            new_lat = round(clicked["lat"], 6)
            new_lon = round(clicked["lng"], 6)
            if (new_lat, new_lon) != (st.session_state.last_lat, st.session_state.last_lon):
                st.session_state.last_lat = new_lat
                st.session_state.last_lon = new_lon
                st.rerun()

    with col_panel:
        if st.session_state.last_lat is None:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#334155;">
                <div style="font-size:48px;margin-bottom:16px;">🗺️</div>
                <div style="font-size:15px;font-weight:600;color:#475569;">Click on the map</div>
                <div style="font-size:13px;margin-top:8px;">
                    Tap any point in Sri Lanka to instantly see weather, traffic, and delivery priority
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            lat = st.session_state.last_lat
            lon = st.session_state.last_lon

            with st.spinner("Fetching live data…"):
                weather_data = fetch_live_weather(lat, lon)
                traffic_data = get_traffic(lat, lon)
                area_name    = get_area_name(lat, lon)

            priority = model_priority(weather_data["label"], traffic_data["label"])
            assigned_v, escalated, esc_reason = escalate(priority, "motorcycle")

            # Save to log
            entry = {
                "lat":      lat, "lon": lon,
                "area":     area_name,
                "weather":  weather_data["label"],
                "traffic":  traffic_data,
                "priority": priority,
                "escalated": escalated,
                "vehicle":  assigned_v,
                "time":     datetime.now().strftime("%H:%M"),
            }
            # Only add if it's a new location
            if not st.session_state.scan_log or \
               (st.session_state.scan_log[0].get("lat") != lat or
                st.session_state.scan_log[0].get("lon") != lon):
                st.session_state.scan_log = [entry] + st.session_state.scan_log[:49]
            st.session_state.last_result = entry

            color = PRI_COLOR[priority]

            # AREA
            st.markdown(f"""
            <div style="background:#0d1625;border:1px solid #1e3a5f;border-radius:10px;padding:14px 16px;margin-bottom:12px;">
                <div style="font-size:9px;color:#334155;letter-spacing:3px;margin-bottom:4px;">SELECTED ZONE</div>
                <div style="font-size:17px;font-weight:700;color:#f1f5f9;">📍 {area_name}</div>
                <div style="font-size:11px;color:#475569;margin-top:2px;">{lat}°N · {lon}°E</div>
            </div>
            """, unsafe_allow_html=True)

            # WEATHER + TRAFFIC cards side by side
            wc1, wc2 = st.columns(2)
            with wc1:
                live_tag = "<span style='font-size:8px;color:#22c55e;'>● LIVE</span>" if weather_data["live"] else "<span style='font-size:8px;color:#64748b;'>● SIMULATED</span>"
                temp_str = f"<div style='font-size:11px;color:#64748b;'>{weather_data['temp']:.1f}°C</div>" if weather_data.get("temp") else ""
                wind_str = f"<div style='font-size:10px;color:#64748b;'>💨 {weather_data['wind']} km/h</div>" if weather_data.get("wind") else ""
                st.markdown(f"""
                <div style="background:#0d1625;border:1px solid #1e3a5f;border-radius:10px;padding:14px;margin-bottom:12px;height:130px;">
                    <div style="font-size:9px;color:#334155;letter-spacing:3px;margin-bottom:6px;">WEATHER {live_tag}</div>
                    <div style="font-size:30px;">{weather_data["icon"]}</div>
                    <div style="font-size:15px;font-weight:700;color:#f1f5f9;">{weather_data["label"]}</div>
                    {temp_str}{wind_str}
                </div>
                """, unsafe_allow_html=True)

            with wc2:
                h = datetime.now().hour
                peak_tag = "<span style='font-size:8px;color:#f97316;'>⏰ PEAK</span>" if (7<=h<=9 or 17<=h<=19) else "<span style='font-size:8px;color:#64748b;'>OFF-PEAK</span>"
                st.markdown(f"""
                <div style="background:#0d1625;border:1px solid #1e3a5f;border-radius:10px;padding:14px;margin-bottom:12px;height:130px;">
                    <div style="font-size:9px;color:#334155;letter-spacing:3px;margin-bottom:6px;">TRAFFIC {peak_tag}</div>
                    <div style="font-size:26px;">{traffic_data["icon"]}</div>
                    <div style="font-size:14px;font-weight:700;color:{traffic_data['color']};">{traffic_data["desc"]}</div>
                    <div style="height:5px;background:#0a1525;border-radius:3px;margin-top:8px;overflow:hidden;">
                        <div style="height:100%;width:{traffic_data['score']}%;background:{traffic_data['color']};border-radius:3px;"></div>
                    </div>
                    <div style="font-size:9px;color:#64748b;margin-top:2px;">{traffic_data['score']:.0f}% congestion</div>
                </div>
                """, unsafe_allow_html=True)

            # PRIORITY RESULT
            st.markdown(f"""
            <div style="background:{PRI_BG[priority]};border:2px solid {color};border-radius:12px;
                        padding:18px;margin-bottom:12px;text-align:center;box-shadow:0 0 20px {color}30;">
                <div style="font-size:9px;color:{color};letter-spacing:4px;margin-bottom:4px;">GA MODEL PREDICTION</div>
                <div style="font-size:36px;margin-bottom:4px;">{PRI_EMOJI[priority]}</div>
                <div style="font-size:28px;font-weight:900;color:{color};letter-spacing:2px;">{priority.upper()} PRIORITY</div>
                <div style="font-size:11px;color:{color};opacity:0.7;margin-top:4px;">
                    {traffic_data['label']} traffic + {weather_data['label']} weather
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ESCALATION
            esc_color = "#a78bfa" if escalated else "#22c55e"
            esc_icon  = "🚐" if escalated else "✅"
            st.markdown(f"""
            <div style="background:#0d1625;border:1px solid {esc_color}40;border-radius:10px;padding:14px;">
                <div style="font-size:9px;color:#334155;letter-spacing:3px;margin-bottom:6px;">VEHICLE ESCALATION</div>
                <div style="font-size:20px;">{esc_icon}</div>
                <div style="font-size:13px;font-weight:600;color:{esc_color};margin-top:4px;">
                    Assigned: <span style="text-transform:uppercase;">{assigned_v}</span>
                </div>
                <div style="font-size:11px;color:#64748b;margin-top:2px;">{esc_reason}</div>
            </div>
            """, unsafe_allow_html=True)

           # Prediction rule explanation
            with st.expander("🔍 See prediction rule"):
                rules = [
                    ("Jam + Stormy → **HIGH**", "red"),
                    ("Jam + Rainy → **HIGH**", "red"),
                    ("High traffic + Stormy → **HIGH**", "red"),
                    ("High traffic + Rainy → **HIGH**", "red"),
                    ("Jam + Other weather → **MEDIUM**", "orange"),
                    ("Everything else → **LOW**", "green"),
                ]
                for rule, c in rules:
                    st.markdown(f"- :{c}[{rule}]")
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PARCEL ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

def render_parcel_entry():
    st.markdown("## 📋 Parcel Entry & Priority Assessment")
    st.markdown("<div style='color:#64748b;font-size:14px;margin-bottom:24px;'>Enter parcel details manually to get an instant GA-based priority prediction and vehicle assignment.</div>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("#### 📝 Parcel Details")
        with st.form("parcel_form", clear_on_submit=False):
            order_id   = st.text_input("Order ID", placeholder="e.g. ORD-20241201-001")
            category   = st.selectbox("Category", ["Grocery","Electronics","Medicine","Food","Clothing","Documents","Snack","Other"])
            vehicle    = st.selectbox("Requested Vehicle", ["Motorcycle","Bicycle","Scooter","Van","Truck"])
            city       = st.selectbox("Destination City", [
                "Colombo","Gampaha","Kandy","Galle","Jaffna","Trincomalee",
                "Batticaloa","Anuradhapura","Ratnapura","Matara","Kurunegala",
                "Badulla","Hambantota","Polonnaruwa","Puttalam","Vavuniya",
            ])
            delivery_time = st.number_input("Estimated Delivery Time (min)", min_value=10, max_value=300, value=45)
            festival   = st.selectbox("Festival / Holiday?", ["No","Yes"])

            st.markdown("---")
            st.markdown("**Conditions at destination**")
            weather_sel = st.selectbox("Weather", ["Sunny","Cloudy","Rainy","Stormy","Windy","Foggy","Drizzle"])
            traffic_sel = st.selectbox("Traffic", ["Low","Medium","High","Jam"])

            submitted = st.form_submit_button("🚀 Predict Priority & Assign Vehicle", use_container_width=True)
    
    with col_result:
        st.markdown("#### 🤖 AI Assessment Result")

        if submitted:
            # Run prediction
            priority = model_priority(weather_sel, traffic_sel, delivery_time, category)
            assigned_v, escalated, esc_reason = escalate(priority, vehicle)
            color = PRI_COLOR[priority]

            # Simulate GA score
            scores_demo = {
                "High":   round(np.random.uniform(0.75, 0.95), 3) if priority=="High"   else round(np.random.uniform(0.05, 0.25), 3),
                "Medium": round(np.random.uniform(0.60, 0.85), 3) if priority=="Medium" else round(np.random.uniform(0.05, 0.30), 3),
                "Low":    round(np.random.uniform(0.70, 0.92), 3) if priority=="Low"    else round(np.random.uniform(0.05, 0.25), 3),
            }

            st.markdown(f"""
            <div style="background:{PRI_BG[priority]};border:2px solid {color};
                        border-radius:14px;padding:24px;margin-bottom:16px;text-align:center;
                        box-shadow:0 0 24px {color}30;">
                <div style="font-size:10px;color:{color};letter-spacing:4px;">DELIVERY PRIORITY</div>
                <div style="font-size:52px;margin:8px 0;">{PRI_EMOJI[priority]}</div>
                <div style="font-size:32px;font-weight:900;color:{color};">{priority.upper()}</div>
                <div style="font-size:12px;color:{color};opacity:0.7;margin-top:6px;">
                    Order {order_id or "N/A"} · {category}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Details grid
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
                <div style="background:#0d1625;border:1px solid #1e3a5f;border-radius:8px;padding:12px;">
                    <div style="font-size:9px;color:#334155;letter-spacing:2px;">WEATHER</div>
                    <div style="font-size:14px;font-weight:600;color:#f1f5f9;margin-top:2px;">{weather_sel}</div>
                </div>
                <div style="background:#0d1625;border:1px solid #1e3a5f;border-radius:8px;padding:12px;">
                    <div style="font-size:9px;color:#334155;letter-spacing:2px;">TRAFFIC</div>
                    <div style="font-size:14px;font-weight:600;color:#f1f5f9;margin-top:2px;">{traffic_sel}</div>
                </div>
                <div style="background:#0d1625;border:1px solid #1e3a5f;border-radius:8px;padding:12px;">
                    <div style="font-size:9px;color:#334155;letter-spacing:2px;">REQUESTED</div>
                    <div style="font-size:14px;font-weight:600;color:#f1f5f9;margin-top:2px;">{vehicle}</div>
                </div>
                <div style="background:#0d1625;border:{'1.5px solid #a78bfa' if escalated else '1px solid #22c55e'};border-radius:8px;padding:12px;">
                    <div style="font-size:9px;color:#334155;letter-spacing:2px;">ASSIGNED {'🚐 ESCALATED' if escalated else '✅'}</div>
                    <div style="font-size:14px;font-weight:700;color:{'#a78bfa' if escalated else '#22c55e'};margin-top:2px;text-transform:uppercase;">{assigned_v}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:#0d1625;border:1px solid #1e3a5f;border-radius:8px;padding:12px;margin-bottom:14px;">
                <div style="font-size:9px;color:#334155;letter-spacing:2px;margin-bottom:4px;">ESCALATION REASON</div>
                <div style="font-size:12px;color:#94a3b8;">{esc_reason}</div>
            </div>
            """, unsafe_allow_html=True)

            # GA confidence scores
            st.markdown("**GA Model Confidence Scores**")
            for label, score in scores_demo.items():
                c = PRI_COLOR[label]
                pct = int(score * 100)
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px;">
                        <span style="color:{c};">{PRI_EMOJI[label]} {label}</span>
                        <span style="color:#64748b;">{pct}%</span>
                    </div>
                    <div style="height:5px;background:#0a1525;border-radius:3px;overflow:hidden;">
                        <div style="height:100%;width:{pct}%;background:{c};border-radius:3px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Add to scan log
            st.session_state.scan_log = [{
                "lat": None, "lon": None,
                "area": f"{city} — {order_id or 'Manual Entry'}",
                "weather": weather_sel,
                "traffic": {"desc": traffic_sel, "label": traffic_sel, "color": "#3b82f6", "icon":"🚦", "score":50},
                "priority": priority,
                "escalated": escalated,
                "vehicle": assigned_v,
                "time": datetime.now().strftime("%H:%M"),
            }] + st.session_state.scan_log[:49]

        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#334155;">
                <div style="font-size:48px;margin-bottom:16px;">📋</div>
                <div style="font-size:14px;color:#475569;">Fill in the form and click<br/><strong style="color:#3b82f6;">Predict Priority</strong> to get a result</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FLEET OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def render_fleet_optimization():
    st.markdown("## 🚚 Fleet Optimization Engine")
    st.markdown("<div style='color:#64748b;font-size:14px;margin-bottom:24px;'>Run the Genetic Algorithm pipeline to train the model, predict priorities across all orders, and optimally allocate the vehicle fleet.</div>", unsafe_allow_html=True)

    # Pipeline steps explanation
    col_a, col_b, col_c = st.columns(3)
    for col, num, title, desc, icon in [
        (col_a,"01","Train GA Model",    "80/20 split · 200 generations · balanced accuracy", "🧬"),
        (col_b,"02","Predict Priorities","Apply GA weights to all orders in the dataset",      "🎯"),
        (col_c,"03","Allocate Fleet",    "Sort by urgency · escalate overloaded vehicles",     "🚐"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#0d1625;border:1px solid #1e3a5f;border-radius:12px;padding:18px;text-align:center;margin-bottom:20px;">
                <div style="font-size:28px;">{icon}</div>
                <div style="font-size:9px;color:#334155;letter-spacing:3px;margin-top:8px;">STEP {num}</div>
                <div style="font-size:14px;font-weight:700;color:#f1f5f9;margin-top:4px;">{title}</div>
                <div style="font-size:11px;color:#64748b;margin-top:6px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Run controls
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    run_all  = ctrl1.button("🚀 Run Full Pipeline", use_container_width=True)
    run_only = ctrl2.button("⚡ Fleet Allocation Only", use_container_width=True)
    retrain  = ctrl3.button("🧬 Train Model Only", use_container_width=True)

    script_map = {
        "train":  SRC_DIR / "train_model.py",
        "predict":SRC_DIR / "predict_priority.py",
        "engine": SRC_DIR / "priority_engine.py",
    }

    def run_script(path, label):
        st.markdown(f"**{label}**")
        if not path.exists():
            st.error(f"Script not found: `{path}`")
            return False
        with st.spinner(f"Running {path.name}…"):
            res = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True, text=True, cwd=str(ROOT_DIR)
            )
        if res.returncode != 0:
            st.error(f"❌ Failed")
            st.code(res.stderr[-2000:], language="text")
            return False
        st.success(f"✅ {path.name} completed")
        if res.stdout.strip():
            with st.expander("View output"):
                st.code(res.stdout[-1500:], language="text")
        return True

    if run_all:
        for k, label in [("train","🧬 Training GA model…"),("predict","🎯 Predicting priorities…"),("engine","🚐 Allocating fleet…")]:
            ok = run_script(script_map[k], label)
            if not ok: break
        else:
            st.cache_resource.clear()
            st.success("🎉 Full pipeline complete! Model reloaded.")
            st.rerun()

    if retrain:
        run_script(script_map["train"], "🧬 Training GA model…")
        st.cache_resource.clear()

    if run_only:
        run_script(script_map["engine"], "🚐 Running fleet allocation…")

    st.markdown("---")

    # Show chart if exists
    if CHART_PATH.exists():
        st.markdown("#### 📈 Training Results")
        st.image(str(CHART_PATH), use_container_width=True)

    # Show output CSV
    if OUTPUT_PATH.exists():
        df = pd.read_csv(OUTPUT_PATH)
        total     = len(df)
        delivered = len(df[df["Status"]=="Delivered"])
        escalated = len(df[df.get("Escalated", pd.Series(dtype=str)) == "Yes"]) if "Escalated" in df.columns else 0

        st.markdown("#### 📊 Simulation Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Orders",        f"{total:,}")
        m2.metric("Delivered",           f"{delivered:,}")
        m3.metric("Success Rate",        f"{delivered/total*100:.2f}%")
        m4.metric("Escalated to Van",    f"{escalated:,}")

        if "Priority" in df.columns:
            st.markdown("#### Priority Breakdown")
            counts = df["Priority"].value_counts().reset_index()
            counts.columns = ["Priority","Count"]
            st.bar_chart(counts.set_index("Priority"))

        if "Assigned_Vehicle" in df.columns:
            st.markdown("#### Vehicle Assignment")
            vc = df["Assigned_Vehicle"].value_counts().reset_index()
            vc.columns = ["Vehicle","Count"]
            st.bar_chart(vc.set_index("Vehicle"))

        st.markdown(f"#### 📋 Sample Results (50 rows)")
        st.dataframe(df.sample(min(50, len(df))), use_container_width=True)
    else:
        st.info("No simulation output yet. Run the pipeline above to generate results.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════
def render_about_page():
    st.markdown("## ⚙️ How It Works")

    tab1, tab2, tab3, tab4 = st.tabs(["🧬 Genetic Algorithm","📊 Priority Logic","🚐 Escalation","🏗️ Architecture"])

    with tab1:
        st.markdown("""
        ### Genetic Algorithm for Delivery Optimization

        The system uses a **Genetic Algorithm (GA)** — an evolutionary optimization technique — to learn
        the best weight matrix for classifying delivery priority.

        #### Why Genetic Algorithm?
        Traditional classifiers assume linear separability. The GA evolves weights through **selection,
        crossover, and mutation** — finding non-obvious solutions that a gradient descent might miss,
        especially on imbalanced datasets like ours.

        #### Training Process
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Population & Fitness**
            - Population size: **100 weight matrices**
            - Each matrix: `(num_features × num_classes)`
            - Fitness function: **Balanced Accuracy Score**
              (handles class imbalance — rewards correctly classifying rare High-priority deliveries)
            """)
        with col2:
            st.markdown("""
            **Evolution Steps (200 generations)**
            1. Sort population by fitness
            2. Keep top 10 elites (elitism)
            3. Crossover: blend top-15 parents
            4. Mutation (20% chance): add Gaussian noise
            5. Repeat until convergence
            """)

        st.markdown("""
        #### Data Preparation
        | Feature | Description |
        |---------|-------------|
        | `Delivery_Time` | Expected delivery duration (minutes) |
        | `Traffic` | Road congestion: Low / Medium / High / Jam |
        | `Weather` | Conditions: Sunny / Cloudy / Rainy / Stormy etc. |
        | `Category` | Package type: Grocery, Medicine, Electronics… |

        **80/20 stratified split** ensures High and Medium labels (minority classes) appear in both
        training and test sets proportionally.
        """)

    with tab2:
        st.markdown("""
        ### Priority Classification Rules

        The priority labelling is injected into training data **before** the GA learns from it:
        """)

        st.code("""
# From train_model.py — the ground truth rules
df.loc[(df['Traffic'] == 'Jam') & (df['Weather'] == 'Stormy'), 'Priority_Level'] = 'High'
df.loc[(df['Traffic'] == 'Jam') & (df['Weather'] != 'Stormy'), 'Priority_Level'] = 'Medium'
df.loc[df['Priority_Level'].isna(), 'Priority_Level'] = 'Low'
        """, language="python")

        st.markdown("""
        | Traffic | Weather | Priority |
        |---------|---------|----------|
        | Jam | Stormy | 🔴 **HIGH** |
        | Jam | Any other | 🟡 **MEDIUM** |
        | High | Stormy / Rainy | 🟡 **MEDIUM** |
        | Stormy (any traffic) | — | 🟡 **MEDIUM** |
        | Anything else | Anything else | 🟢 **LOW** |

        The GA then **learns** these patterns from data (rather than hardcoding them), so it
        generalises to unseen weather/traffic combinations it wasn't explicitly told about.
        """)

    with tab3:
        st.markdown("""
        ### Vehicle Escalation Logic

        When a delivery is classified as High or Medium priority, the system checks whether
        the assigned vehicle is adequate. If not, it **escalates** up the vehicle ladder:

        ```
        Bicycle → Motorcycle → Scooter → Van → Truck
        ```

        | Priority | Min Required | Action |
        |----------|-------------|--------|
        | 🔴 High   | Van         | Escalate to Van if below |
        | 🟡 Medium | Scooter     | Escalate to Scooter if below |
        | 🟢 Low    | Any         | No escalation |

        #### Fleet Capacity (priority_engine.py)
        | Vehicle | Capacity |
        |---------|---------|
        | Bicycle | 1,000 orders |
        | Motorcycle | 5,000 orders |
        | Scooter | 3,000 orders |
        | Van | 100,000 orders |

        Orders are **sorted by urgency first** (High → Medium → Low) before allocation,
        so critical deliveries always get a vehicle before lower-priority ones.
        """)

    with tab4:
        st.markdown("""
        ### System Architecture

        ```
        project/
        ├── src/
        │   ├── app.py              ← This Streamlit app
        │   ├── train_model.py      ← GA training (80/20 split, 200 gens)
        │   ├── predict_priority.py ← Batch prediction on full dataset
        │   ├── priority_engine.py  ← Fleet allocation + escalation
        │   └── run_prototype.py    ← CLI pipeline runner
        └── data/
            ├── amazon_delivery_with_priority.csv   ← Input dataset
            ├── trained_ga_model.pkl                ← Saved model weights
            ├── delivery_simulation_output.csv      ← Fleet allocation results
            └── training_results_chart.png          ← Confusion matrix
        ```

        #### Data Flow
        ```
        CSV Dataset
            ↓  train_model.py (GA)
        trained_ga_model.pkl
            ↓  predict_priority.py
        Dataset + Priority_Level column
            ↓  priority_engine.py
        delivery_simulation_output.csv (Status + Escalated)
            ↓  app.py
        Live Dashboard + Map + Parcel Entry UI
        ```

        #### Live Weather Integration
        The Map Inspector fetches **real weather** from **Open-Meteo API** (free, no API key):
        ```
        https://api.open-meteo.com/v1/forecast?latitude=...&longitude=...
        &current=weather_code,wind_speed_10m,precipitation,temperature_2m
        ```
        WMO weather codes are mapped to: Sunny / Cloudy / Rainy / Stormy / Windy / Drizzle / Foggy
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#334155;font-size:12px;padding:20px;">
        DeliveryIQ · Smart Goods Priority Optimization · Sri Lanka<br/>
        Genetic Algorithm · Real-time Weather · Fleet Escalation
    </div>
    """, unsafe_allow_html=True)
    
page = st.session_state.page 

if page == "dashboard":
    render_dashboard()
elif page == "scan":
    render_qr_scanner_page()
elif page == "map":
    render_map_inspector()
elif page == "parcel":
    render_parcel_entry()