import subprocess
import sys
import time
from pathlib import Path

# ── PATHS ──────────────────────────────────────────────────────────────────────
SRC_DIR  = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

def run_step(script_name: str, description: str) -> bool:
    """Execute a single script in the pipeline."""
    script_path = SRC_DIR / script_name

    if not script_path.exists():
        print(f"  ⚠️  Script not found: {script_path}")
        return False

    print(f"\n{'='*50}")
    print(f">>> {description}")
    print(f"{'='*50}")

    # Run script with the project root as the working directory
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
        encoding='utf-8' # Ensure subprocess output is handled as UTF-8
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(f"[Details/Warnings]\n{result.stderr}")

    if result.returncode != 0:
        print(f"  ❌  {script_name} failed with exit code {result.returncode}")
        return False

    print(f"  ✅  {script_name} finished successfully.")
    return True

def main():
    print("\n" + " -" * 25)
    print("  DELIVERYIQ — FULL SYSTEM EXECUTION")
    print(" =" * 25)
    start_time = time.time()

    # The 4-Step Pipeline
    steps = [
        ("train_model.py",      "STEP 1/4: Training Genetic Algorithm"),
        ("predict_priority.py", "STEP 2/4: Predicting Delivery Priorities"),
        ("priority_engine.py",  "STEP 3/4: Executing Priority-Based Scheduling"),
        ("generate_report.py",  "STEP 4/4: Generating Final Analysis Reports"),
    ]

    for script, label in steps:
        if not run_step(script, label):
            print(f"\n🛑 System stopped: Error in {script}")
            sys.exit(1)

    duration = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"  FULL PIPELINE COMPLETED IN {duration:.1f}s")
    print(f"  Check the 'reports/' folder for your results.")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()