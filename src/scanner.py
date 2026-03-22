import streamlit as st
import pandas as pd
import numpy as np
import cv2
import re
from pathlib import Path
from notifications import add_notification

# ───────────────────────────────
# PATHS
# ───────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FOLDER = BASE_DIR / "data"

# You can use either CSV, but the code now handles the missing column
INPUT_CSV = DATA_FOLDER / "amazon_delivery_with_priority_and_links.csv"

# ───────────────────────────────
# LOGIC FUNCTIONS
# ───────────────────────────────

def load_csv():
    """Loads dataset and ensures the 'Status' column exists."""
    if not INPUT_CSV.exists():
        st.error(f"CSV not found at: {INPUT_CSV}")
        return pd.DataFrame()
    
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()
    
    # FIX: If 'Status' column is missing, create it with 'Pending' as default
    if "Status" not in df.columns:
        df["Status"] = "Pending"
        
    return df

def update_csv_status(order_id):
    """Updates the delivery status using a fuzzy matching approach."""
    df = load_csv()
    if df.empty:
        return "not_found", pd.DataFrame()

    # Normalize User Input: Remove spaces, dashes, special chars, and lowercase it
    clean_search_id = re.sub(r'[^a-zA-Z0-9]', '', str(order_id)).lower()
    
    # Normalize CSV Column for comparison
    df["Match_ID"] = df["Order_ID"].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True).str.lower()
    
    if clean_search_id in df["Match_ID"].values:
        row_idx = df[df["Match_ID"] == clean_search_id].index
        
        # Access the first matching row's status
        current_status = str(df.loc[row_idx[0], "Status"]).strip().lower()
        
        if current_status == "delivered":
            result_row = df.loc[row_idx].drop(columns=["Match_ID"])
            return "already", result_row
        else:
            # Update the status to 'Delivered'
            df.loc[row_idx, "Status"] = "Delivered"
            # Remove helper column before saving
            df.drop(columns=["Match_ID"], inplace=True)
            df.to_csv(INPUT_CSV, index=False)
            
            return "updated", df.loc[row_idx]
    
    return "not_found", pd.DataFrame()

# ───────────────────────────────
# UI RENDER FUNCTION
# ───────────────────────────────

def render_qr_scanner_page():
    st.markdown("## 📷 Agent Delivery Portal")
    st.markdown("<div style='color:#64748b;margin-bottom:20px;'>Fuzzy matching: Spaces, dashes, and capitalization are ignored.</div>", unsafe_allow_html=True)
    
    active_id = None
    
    # 1. Camera Input Section
    img_file = st.camera_input("Scan Parcel QR Code")
    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(gray)
            if data:
                active_id = data.strip()
                st.success(f"🎯 QR Detected: **{active_id}**")

    # 2. Manual Fallback Section
    manual_id = st.text_input("✏️ Manual Order ID Entry", placeholder="Type Order ID here...")
    if manual_id:
        active_id = manual_id.strip()

    # 3. Confirmation Action
    if active_id:
        st.info(f"Ready to process: **{active_id}**")

        if st.button("✅ Confirm Delivery Completion"):
            status, record = update_csv_status(active_id)
            
            if status == "updated":
                add_notification(
                    order_id=active_id, 
                    message="Parcel successfully delivered to customer.",
                    status_type="delivered"
                )
                st.balloons()
                st.success(f"Order {active_id} marked as Delivered!")
                st.dataframe(record, use_container_width=True)
                
            elif status == "already":
                st.warning(f"Order {active_id} has already been delivered.")
                st.dataframe(record, use_container_width=True)
                
            else:
                st.error(f"Error: Order ID '{active_id}' not found.")
                with st.expander("Show valid ID examples from dataset"):
                    sample_df = load_csv()
                    if not sample_df.empty:
                        st.write(sample_df["Order_ID"].head(5).tolist())

if __name__ == "__main__":
    render_qr_scanner_page()