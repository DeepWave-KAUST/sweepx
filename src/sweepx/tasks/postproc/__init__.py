"""Post-processing utilities applied to imaging products.

These helpers are independent of the FWI/RTM runner and operate on
already-saved 2-D numpy arrays so they can be re-run cheaply with
different parameters when the user is iterating on display / cleanup.
"""

from . import filter_image

__all__ = ["filter_image"]
