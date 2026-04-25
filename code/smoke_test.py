"""
Smoke test for Stage 1.
Runs ONE subset with 100 samples to verify the environment and pipeline.
Expected runtime: 3-5 minutes on RTX 4090.

Usage:
    python smoke_test.py
"""

import subprocess
import sys

cmd = [
    sys.executable,
    "run_stage1_sbar.py",
    "--subsets", "Wikipedia (en)",
    "--n_samples", "100",
    "--output_dir", "results_smoke",
    "--cache_dir", "cache_smoke",
]

print("=" * 60)
print("SMOKE TEST · 1 subset (Wikipedia en) · 100 samples")
print("Expected runtime: 3-5 min on RTX 4090")
print("=" * 60)
print(f"Command: {' '.join(cmd)}")
print()

result = subprocess.run(cmd)

print()
print("=" * 60)
if result.returncode == 0:
    print("[PASS] Smoke test completed successfully.")
    print("Check: results_smoke/sbar_raw.csv")
    print("If numbers look reasonable, run the full thing:")
    print("  python run_stage1_sbar.py --n_samples 5000")
else:
    print("[FAIL] Smoke test failed. Check the error above.")
    print("Common causes:")
    print("  - Missing deps: pip install -r requirements.txt")
    print("  - HF token / network issue")
    print("  - Subset name mismatch in dataset meta")
print("=" * 60)
sys.exit(result.returncode)
