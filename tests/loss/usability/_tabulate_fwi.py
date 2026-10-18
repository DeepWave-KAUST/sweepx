"""Tabulate the FWI summary into a markdown table for the report."""

from __future__ import annotations

import json
from pathlib import Path

p = Path(__file__).resolve().parent / "fwi_layer.json"
if not p.exists():
    print("(fwi_layer.json not produced yet)")
    raise SystemExit

rows = json.loads(p.read_text())
print("{:<10s} {:>12s} {:>14s} {:>10s} {:<24s}".format(
    "loss", "init_rmse", "final_rmse", "speedup", "comment"))
print("-" * 80)
for r in rows:
    if "error" in r:
        print("{:<10s} {:<60s}".format(r["name"], "ERROR: " + r["error"][:60]))
        continue
    speed = r["init_rmse_m_per_s"] / max(r["final_rmse_m_per_s"], 1e-12)
    monotone = all(b - a < 1e-3 for a, b in zip(r["rmse_curve"], r["rmse_curve"][1:]))
    note = "monotone RMSE" if monotone else "RMSE non-monotone"
    print("{:<10s} {:>12.2f} {:>14.2f} {:>10.2f}x {:<24s}".format(
        r["name"], r["init_rmse_m_per_s"], r["final_rmse_m_per_s"], speed, note))


# Also dump as markdown for direct paste into REPORT.md
print("\n\n# markdown table")
print("| Loss | init RMSE (m/s) | final RMSE (m/s) | speedup | comment |")
print("|---|---:|---:|---:|---|")
for r in rows:
    if "error" in r:
        print(f"| {r['name']} | — | — | — | ERROR: {r['error'][:40]} |")
        continue
    speed = r["init_rmse_m_per_s"] / max(r["final_rmse_m_per_s"], 1e-12)
    monotone = all(b - a < 1e-3 for a, b in zip(r["rmse_curve"], r["rmse_curve"][1:]))
    note = "monotone RMSE" if monotone else "RMSE non-monotone"
    print(f"| {r['name']} | {r['init_rmse_m_per_s']:.1f} | {r['final_rmse_m_per_s']:.1f} | "
          f"{speed:.2f}× | {note} |")
