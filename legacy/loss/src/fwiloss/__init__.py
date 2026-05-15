"""fwiloss - a PyTorch library of FWI misfit functions.

The package exposes every loss as a ``torch.nn.Module`` so it can be plugged
into any FWI training loop. The canonical tensor shape is

    (nshots, nt, nreceivers, nchannel)

Time axis is ``-3``. Per-trace operations are computed on that axis after
flattening the leading dimensions to ``(nshots * nreceivers * nchannel, nt)``.

Public API
----------
``BaseFWILoss``  - common base class with reduction / masking / shape handling.

Each loss is also re-exported at the top level once it has been implemented,
e.g. ``from fwiloss import L2Loss``.
"""

from __future__ import annotations

from .base import BaseFWILoss
from .l2 import L2Loss, l2_loss

__all__ = [
    "BaseFWILoss",
    "L2Loss",
    "l2_loss",
]

__version__ = "0.1.0"
