"""Tabulate cycle-skipping scan summary."""
from __future__ import annotations
import json
from pathlib import Path

p = Path(__file__).resolve().parent / "cycle_skipping_summary.json"
rows = json.loads(p.read_text())

print("{:<32s} {:>12s} {:>14s} {:>20s} {:>9s} {:<10s}".format(
    "loss", "argmin_tau", "basin_hw_ms", "basin_in_half_wl", "monotone", "fc_Hz"))
print("-" * 110)
for r in rows:
    if "error" in r:
        print("{:<32s} {:<60s}".format(r["name"], "ERROR: " + r["error"][:60]))
        continue
    print("{:<32s} {:12.5e} {:14.3f} {:20.3f} {:>9s} {:<10s}".format(
        r["name"], r["argmin_tau_s"], r["basin_half_width_s"]*1000.0,
        r["basin_in_half_wavelengths"],
        "Y" if r["monotone_over_scan"] else "N",
        f'{r["fc_Hz"]:g}',
    ))
