"""Tabulate the sanity_report.json into a human-readable table."""
from __future__ import annotations
import json
import sys
from pathlib import Path

p = Path(__file__).resolve().parent / "sanity_report.json"
rows = json.loads(p.read_text())

header = "{:<32s} {:>12s} {:>12s} {:>12s} {:>4s} {:<32s}".format(
    "name", "value", "grad_norm", "zero_resid", "OK", "notes")
print(header)
print("-" * len(header))
for r in rows:
    if "error" in r:
        print("{:<32s} {:>12s} {:>12s} {:>12s} {:>4s} {:<32s}".format(
            r["name"], "ERR", "-", "-", "NO", r["error"][:30]))
        continue
    ok = (r.get("finite", False) and r.get("grad_finite", False)
          and r.get("grad_nonzero", False) and r.get("zero_residual_small", False))
    notes = []
    if r.get("value", 0) < 0:
        notes.append("NEG")
    if not r.get("zero_residual_small", True):
        notes.append("zr={:.2e}".format(r["zero_residual_value"]))
    if not r.get("grad_nonzero", False):
        notes.append("g=0")
    print("{:<32s} {:12.3e} {:12.3e} {:12.3e} {:>4s} {:<32s}".format(
        r["name"], r.get("value", float("nan")),
        r.get("grad_norm", float("nan")),
        r.get("zero_residual_value", float("nan")),
        "OK" if ok else "!!", ",".join(notes)))
