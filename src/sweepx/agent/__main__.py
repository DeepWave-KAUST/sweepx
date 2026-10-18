"""Allow `python -m sweepx.agent ...` as an alternative to the `sweep-agent` script."""

import sys

from sweepx.agent.cli import main

if __name__ == "__main__":
    sys.exit(main())
