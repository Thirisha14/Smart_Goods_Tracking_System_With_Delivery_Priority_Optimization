import subprocess
import sys
import time
from pathlib import Path

# ── PATHS ──────────────────────────────────────────────────────────────────────
# FIX: Resolve absolute paths from __file__ so the script works when called
# from any working directory (e.g. from root, from src/, or from the app).
SRC_DIR  = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent


def run_step(script_name: str, description: str) -> bool:
    """Run one pipeline script, print its output, and return True on success."""
    script_path = SRC_DIR / script_name

    if not script_path.exists():
        print(f"  ⚠️  Script not found: {script_path}")
        return False

    print(f"\n{'='*50}")
    print(f">>> {description}")
    print(f"    {script_path}")
    print(f"{'='*50}")

    # FIX: Pass cwd=ROOT_DIR so every script's BASE_DIR = ROOT_DIR resolves
    # correctly regardless of where run_prototype.py is executed from.
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        # Print stderr but don't treat warnings as failures
        print(f"[stderr]\n{result.stderr}")

    if result.returncode != 0:
        print(f"  ❌  {script_name} exited with code {result.returncode}")
        return False

    print(f"  ✅  {script_name} completed successfully")
    return True


def main():
    print("\n" + "=" * 50)
    print("  DELIVERYIQ — OPTIMIZATION PIPELINE")
    print("=" * 50)
    start = time.time()

    steps = [
        ("train_model.py",      "STEP 1/3 — Train Genetic Algorithm model (200 generations, 80/20 split)"),
        ("predict_priority.py", "STEP 2/3 — Predict priority levels for all orders"),
        ("priority_engine.py",  "STEP 3/3 — Allocate fleet & apply escalation logic"),
    ]

    for script, label in steps:
        success = run_step(script, label)
        if not success:
            print("\n⛔  Pipeline stopped due to error above.")
            print("    Fix the issue and re-run.")
            sys.exit(1)

    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"  ✅  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()