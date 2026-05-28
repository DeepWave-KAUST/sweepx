"""Allow `python -m sweep_agent ...` as an alternative to the `sweep-agent` script."""

import sys

from sweep_agent.cli import main

if __name__ == "__main__":
    sys.exit(main())
