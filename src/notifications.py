import streamlit as st
from datetime import datetime
import json
from pathlib import Path

# Path to store notifications so they don't disappear
NOTIF_FILE = Path("data/notifications_log.json")

def add_notification(order_id, message, status_type="info"):
    """Saves a notification to a JSON file."""
    # Ensure the data folder exists
    NOTIF_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing logs
    logs = []
    if NOTIF_FILE.exists():
        try:
            with open(NOTIF_FILE, "r") as f:
                logs = json.load(f)
        except:
            logs = []

    new_note = {
        "order_id": order_id,
        "msg": message,
        "time": datetime.now().strftime("%H:%M:%S"),
        "type": status_type
    }
    
    # Add to the start (newest first) and keep last 20
    logs = [new_note] + logs[:19]
    
    with open(NOTIF_FILE, "w") as f:
        json.dump(logs, f, indent=4)

def render_notification_board():
    """UI component to display the logs in the sidebar or dashboard."""
    st.markdown("### 🔔 System Notifications")
    
    if not NOTIF_FILE.exists():
        st.info("No recent notifications.")
        return

    with open(NOTIF_FILE, "r") as f:
        logs = json.load(f)

    if not logs:
        st.info("No recent notifications.")
        return

    for note in logs:
        color = "#3b82f6" # Default blue
        if note['type'] == 'delivered': color = "#22c55e"
        elif note['type'] == 'priority': color = "#ef4444"

        st.markdown(f"""
        <div style="padding:10px; border-left: 4px solid {color}; background:#0d1625; border-radius:8px; margin-bottom:8px;">
            <div style="font-size:10px; color:#64748b;">{note['time']}</div>
            <div style="font-size:13px; color:white;"><strong>{note['order_id']}</strong>: {note['msg']}</div>
        </div>
        """, unsafe_allow_html=True)