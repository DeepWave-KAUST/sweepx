"""Run every validation script in this folder, in order.

Each script writes ``figs/<name>.png`` and ``figs/<name>.json``.

Usage:
    $ python run_all.py [--only 04 09]   # run only matching scripts
"""
from __future__ import annotations

import argparse
import glob
import os
import runpy
import time


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="match by 2-digit prefix (e.g. 04 09 19)")
    args = parser.parse_args()

    scripts = sorted(
        p for p in glob.glob(os.path.join(HERE, "[0-9][0-9]_*.py"))
    )
    if args.only:
        scripts = [p for p in scripts
                   if any(os.path.basename(p).startswith(prefix) for prefix in args.only)]

    print(f"will run {len(scripts)} scripts")
    for path in scripts:
        name = os.path.basename(path)
        print(f"\n=== {name} ===")
        t0 = time.time()
        try:
            runpy.run_path(path, run_name="__main__")
        except Exception as exc:  # noqa: BLE001
            print(f"!! {name} FAILED: {type(exc).__name__}: {exc}")
            continue
        print(f"   {name} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
