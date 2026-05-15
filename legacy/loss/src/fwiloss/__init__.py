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
from .correlation import (
    GlobalCorrelationLoss,
    TraceNormalizedL2Loss,
    global_correlation_loss,
    trace_normalized_l2_loss,
)
from .huber import HuberLoss, PseudoHuberLoss, huber_loss, pseudo_huber_loss
from .hybrid_l1l2 import HybridL1L2Loss, hybrid_l1l2_loss
from .l1 import L1Loss, l1_loss
from .l2 import L2Loss, l2_loss
from .robust import (
    CauchyLoss,
    GemanMcClureLoss,
    TukeyLoss,
    cauchy_loss,
    geman_mcclure_loss,
    tukey_loss,
)
from .student_t import StudentTLoss, student_t_loss

__all__ = [
    "BaseFWILoss",
    "CauchyLoss",
    "GemanMcClureLoss",
    "GlobalCorrelationLoss",
    "HuberLoss",
    "HybridL1L2Loss",
    "L1Loss",
    "L2Loss",
    "PseudoHuberLoss",
    "StudentTLoss",
    "TraceNormalizedL2Loss",
    "TukeyLoss",
    "cauchy_loss",
    "geman_mcclure_loss",
    "global_correlation_loss",
    "huber_loss",
    "hybrid_l1l2_loss",
    "l1_loss",
    "l2_loss",
    "pseudo_huber_loss",
    "student_t_loss",
    "trace_normalized_l2_loss",
    "tukey_loss",
]

__version__ = "0.1.0"
