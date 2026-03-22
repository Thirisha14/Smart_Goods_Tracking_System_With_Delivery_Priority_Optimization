import streamlit as st
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from notifications import add_notification

# ───────────────────────────────
# PATHS
# ───────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_FOLDER = BASE_DIR / "data"
DATA_FOLDER.mkdir(exist_ok=True)  # Create folder if missing

INPUT_CSV = DATA_FOLDER / "amazon_delivery_with_priority_and_links.csv"
OUTPUT_CSV = DATA_FOLDER / "delivery_simulation_output.csv"

# ───────────────────────────────
# Load & clean CSV
# ───────────────────────────────
def load_csv():
    if not INPUT_CSV.exists():
        st.error(f"CSV file not found at {INPUT_CSV}")
        return pd.DataFrame()
    
    df = pd.read_csv(INPUT_CSV)
    # Strip spaces from column names and string columns
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
    
    # Ensure Status column exists
    if "Status" not in df.columns:
        df["Status"] = "Pending"
    
    return df

# ───────────────────────────────
# Update order status
# ───────────────────────────────
def update_csv_status(order_id):
    df = load_csv()
    if df.empty:
        return "not_found", pd.DataFrame()
    
    # Clean Order_ID column
    df["Order_ID"] = df["Order_ID"].str.replace(r'[\s\u200b]', '', regex=True)
    order_id_clean = order_id.strip().replace(" ", "")
    
    if order_id_clean in df["Order_ID"].values:
        row = df[df["Order_ID"] == order_id_clean]
        if str(row["Status"].values[0]).lower() == "delivered":
            return "already", row
        else:
            df.loc[df["Order_ID"] == order_id_clean, "Status"] = "Delivered"
            df.to_csv(INPUT_CSV, index=False)  # Save updates back to CSV
            return "updated", row
    else:
        return "not_found", pd.DataFrame()

# ───────────────────────────────
# Streamlit QR / manual input page
# ───────────────────────────────
def render_qr_scanner_page():
    st.markdown("## 📷 Agent Delivery Portal")
    st.markdown("Scan QR or enter Order ID manually")
    
    order_id = None
    
    # Camera input
    img_file = st.camera_input("Scan Parcel QR Code")
    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detector = cv2.QRCodeDetector()
            data, bbox, _ = detector.detectAndDecode(gray)
            if data:
                order_id = data.strip()
                st.success(f"📷 QR Detected: {order_id}")
            else:
                st.warning("⚠️ QR not detected. Use manual entry below.")
    
    # Manual fallback
    manual_id = st.text_input("✏️ Enter Order ID manually")
    if manual_id:
        order_id = manual_id.strip()
    
    # Update delivery status
    if order_id:
        status, record = update_csv_status(order_id)
        if status == "updated":
            st.balloons()
            st.success(f"✅ Delivery Updated for {order_id}")
            st.dataframe(record, use_container_width=True)
        elif status == "already":
            st.warning(f"⚠️ Order {order_id} already delivered")
            st.dataframe(record, use_container_width=True)
        else:
            st.error(f"❌ Order {order_id} not found in dataset")

# ───────────────────────────────
# Run page
# ───────────────────────────────
if __name__ == "__main__":
    render_qr_scanner_page()
    
if st.button("✅ Confirm Delivery"):
    if update_csv_status(scanned_id):
        # Trigger the notification!
        add_notification(
            order_id=scanned_id, 
            message="Parcel successfully delivered to customer.",
            status_type="delivered"
        )
        st.success("Status Updated!")