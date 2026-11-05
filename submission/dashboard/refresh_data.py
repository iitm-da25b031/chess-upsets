"""Copies the tables of notebooks 01-04 into the dashboard and rebuilds the tables that depend on the full data.

Usage (from this folder):  python refresh_data.py
Run after notebooks 01-03 and after `python ../viz/make_viz_data.py`.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.dirname(HERE)
DASH = os.path.join(HERE, "data", "dashboard")
MODEL = ["model_coefficients_empirical", "model_coefficients_empirical_verified", "critical_gap_empirical",
         "critical_gap_empirical_verified", "fitted_curves", "calibration_empirical", "curve_bins"]

shutil.copytree(os.path.join(SUB, "viz", "viz_data"), os.path.join(HERE, "data", "viz_data"), dirs_exist_ok=True)
shutil.copytree(os.path.join(SUB, "reports", "model_outputs"), os.path.join(HERE, "data", "model_outputs"), dirs_exist_ok=True)
for name in MODEL:
    shutil.copy2(os.path.join(SUB, "reports", "model_outputs", name + ".csv"), DASH)
subprocess.run([sys.executable, os.path.join(HERE, "prepare_fresh.py")], check=True, cwd=HERE)
print("dashboard data refreshed")
