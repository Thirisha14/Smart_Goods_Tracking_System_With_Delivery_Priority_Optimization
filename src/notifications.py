import streamlit as st
from datetime import datetime

def add_notification(order_id, message, status_type="info"):
    """
    Adds a notification to the global session state log.
    status_type: 'delivered', 'dispatched', or 'info'
    """
    # Initialize the list if it doesn't exist in the session yet
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []

    new_note = {
        "order_id": order_id,
        "msg": message,
        "time": datetime.now().strftime("%H:%M:%S"),
        "type": status_type
    }
    
    # Add to the start of the list so newest is first
    st.session_state.notifications.insert(0, new_note)

def render_notification_board():
    """UI component to display the logs"""
    st.markdown("### 🔔 Customer Notification Log")
    
    if 'notifications' not in st.session_state or not st.session_state.notifications:
        st.info("No recent notifications sent.")
        return

    for note in st.session_state.notifications[:10]:  # Show last 10
        color = "#3b82f6"  # Default blue
        icon = "📩"
        
        if note['type'] == 'delivered':
            color = "#22c55e"
            icon = "✅"
        elif note['type'] == 'dispatched':
            color = "#eab308"
            icon = "🚚"

        st.markdown(f"""
        <div style="padding:10px; border-left: 4px solid {color}; background:#0f172a; margin-bottom:8px; border-radius:4px;">
            <div style="font-size:13px; color:#f1f5f9;">{icon} <b>{note['order_id']}</b>: {note['msg']}</div>
            <div style="font-size:10px; color:#64748b;">Sent at {note['time']} via SMS Gateway</div>
        </div>
        """, unsafe_allow_html=True)