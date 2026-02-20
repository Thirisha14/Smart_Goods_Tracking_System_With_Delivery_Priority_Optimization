import subprocess
import sys
import time
from pathlib import Path

def run_step(script_path, description):
    print(f"\n>>> STEP: {description}")
    
    # sys.executable ensures it uses your (env) python where pandas is installed
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"--- Errors in {script_path} ---\n{result.stderr}")

def main():
    print("=== DELIVERY OPTIMIZATION PROTOTYPE STARTING ===")
    start_time = time.time()

    # Define paths relative to the root folder
    # We use src/ because that is where your logic lives
    run_step("src/train_model.py", "Training GA model on 80% of data")
    run_step("src/predict_priority.py", "Predicting priority levels for current orders")
    run_step("src/priority_engine.py", "Allocating resources and calculating escalation")

    end_time = time.time()
    print(f"\n=== PROTOTYPE RUN COMPLETE in {end_time - start_time:.2f} seconds ===")

if __name__ == "__main__":
    main()