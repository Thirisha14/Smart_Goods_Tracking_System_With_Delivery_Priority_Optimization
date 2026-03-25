import streamlit as st
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from notifications import add_notification

INPUT_CSV = Path("data/delivery_simulation_output.csv")

def update_csv_status(order_id):
    if not INPUT_CSV.exists():
        return "not_found", pd.DataFrame()
    
    df = pd.read_csv(INPUT_CSV)
    order_id_str = str(order_id).strip().lower()
    df['Match_ID'] = df['Order_ID'].astype(str).str.strip().str.lower()

    if order_id_str in df['Match_ID'].values:
        idx = df[df['Match_ID'] == order_id_str].index[0]
        if str(df.at[idx, 'Status']).lower() == 'delivered':
            return "already", df.loc[[idx]]
        
        df.at[idx, 'Status'] = 'Delivered'
        df.drop(columns=['Match_ID'], inplace=True)
        df.to_csv(INPUT_CSV, index=False)
        return "updated", df.loc[[idx]]
    return "not_found", pd.DataFrame()

def render_qr_scanner_page():
    st.header("📷 Agent Delivery Portal")
    
    # Native Streamlit Camera Fix
    img_file = st.camera_input("Scan Parcel QR")
    active_id = st.text_input("Or Enter ID manually", placeholder="e.g. qfgc848777114")

    if img_file:
        # QR Decoding logic
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(frame)
        if data: active_id = data

    if active_id:
        if st.button("✅ Confirm Delivery"):
            status, record = update_csv_status(active_id)
            if status == "updated":
                add_notification(active_id, "Delivered successfully", "delivered")
                st.balloons()
                st.success("Delivery Confirmed!")
            elif status == "already":
                st.warning("Already delivered.")
            else:
                st.error("Order ID not found in simulation database.")