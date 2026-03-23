import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data" / "delivery_simulation_output.csv"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

def generate_delivery_report():
    print("📊 Generating Delivery Performance Report...")

    if not INPUT_CSV.exists():
        print(f"❌ Error: Simulation data not found at {INPUT_CSV}")
        return

    # ─────────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────────
    df = pd.read_csv(INPUT_CSV)

    # Clean column names
    df.columns = df.columns.str.strip()

    # 🔥 IMPORTANT FIX: Ensure correct column name
    if "Priority" in df.columns:
        df.rename(columns={"Priority": "Priority_Level"}, inplace=True)

    if "Priority_Level" not in df.columns:
        print("❌ ERROR: 'Priority_Level' column not found in dataset!")
        return

    # ─────────────────────────────────────────────
    # CORE METRICS
    # ─────────────────────────────────────────────
    total_orders = len(df)
    delivered_df = df[df["Status"] == "Delivered"]
    delivered_count = len(delivered_df)
    pending_count = len(df[df["Status"] == "Pending"])
    delivery_rate = (delivered_count / total_orders) * 100

    # ─────────────────────────────────────────────
    # PRIORITY ANALYSIS (FIXED)
    # ─────────────────────────────────────────────
    priority_stats = (
        df.groupby("Priority_Level")["Status"]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0) * 100
    )

    # ─────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Chart 1: Overall success
    ax1.pie(
        [delivered_count, pending_count],
        labels=['Delivered', 'Pending'],
        autopct='%1.1f%%',
        startangle=140
    )
    ax1.set_title("Overall Delivery Success Rate")

    # Chart 2: Success by Priority
    if "Delivered" in priority_stats.columns:
        priority_stats["Delivered"].plot(kind='bar', ax=ax2)
        ax2.set_title("Delivery Success Rate by Priority")
        ax2.set_ylabel("Percentage (%)")
        ax2.set_ylim(0, 105)
        plt.xticks(rotation=0)

    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = REPORT_DIR / f"delivery_report_{timestamp}.png"

    plt.tight_layout()
    plt.savefig(chart_path)

    # ─────────────────────────────────────────────
    # TEXT REPORT
    # ─────────────────────────────────────────────
    report_text_path = REPORT_DIR / f"summary_{timestamp}.txt"

    with open(report_text_path, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write(f" DELIVERY PERFORMANCE REPORT\n")
        f.write("="*50 + "\n\n")

        f.write(f"Total Orders:      {total_orders}\n")
        f.write(f"Delivered:         {delivered_count} ({delivery_rate:.2f}%)\n")
        f.write(f"Pending:           {pending_count}\n\n")

        f.write("--- Delivery Success by Priority ---\n")
        if "Delivered" in priority_stats.columns:
            f.write(priority_stats["Delivered"].to_string())
        else:
            f.write("No delivery data available")

        f.write("\n\n" + "="*50)

    # ─────────────────────────────────────────────
    # SUCCESS OUTPUT
    # ─────────────────────────────────────────────
    print(f"✅ Report generated successfully!")
    print(f"📊 Chart saved: {chart_path}")
    print(f"📄 Summary saved: {report_text_path}")


if __name__ == "__main__":
    generate_delivery_report()