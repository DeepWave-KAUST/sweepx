"""sweep-tasks quickstart: build a tiny FWI task in Python and dispatch it.

Uses an in-memory Pydantic spec instead of YAML so the example is
self-contained. For YAML-driven usage, see `sweep-tasks new <type> -o
task.yaml` followed by `sweep-tasks run task.yaml`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from sweep_tasks import TaskRunner, dump_task, load_task, new_template


def main() -> None:
    # Emit a canonical FWI template, fill in the bits that need real values.
    tpl = new_template("introspect", equation="Acoustic", backend="eager")
    print("Default introspect template:")
    print(tpl)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out_yaml = tmp / "introspect.yaml"
        # `new_template` returns a dict; convert to YAML via yaml_io for round-trip.
        # Easiest path: write the dict, then re-load through the schema.
        import yaml as _yaml
        out_yaml.write_text(_yaml.safe_dump(tpl, sort_keys=False))

        spec = load_task(out_yaml)
        result = TaskRunner().run(spec)
        print(f"\nintrospect task -> state={result.status.state}")
        print(f"summary keys: {list((result.status.summary or {}).keys())}")
        # Round-trip the spec back out so you can see what `dump_task` writes.
        dump_task(spec, tmp / "roundtrip.yaml")
        print(f"\nroundtrip YAML head:\n{(tmp / 'roundtrip.yaml').read_text()[:300]}")


if __name__ == "__main__":
    main()
