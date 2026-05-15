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

from .awi import AWILoss, awi_loss
from .base import BaseFWILoss
from .correlation import (
    GlobalCorrelationLoss,
    TraceNormalizedL2Loss,
    global_correlation_loss,
    trace_normalized_l2_loss,
)
from .deconvolution import DeconvolutionLoss, deconvolution_loss
from .envelope import EnvelopeLoss, envelope_loss
from .frequency import (
    FrequencyAmplitudeLoss,
    FrequencyDomainL2Loss,
    FrequencyPhaseLoss,
    LaplaceL2Loss,
    LogarithmicShinMinLoss,
    frequency_amplitude_loss,
    frequency_domain_l2_loss,
    frequency_phase_loss,
    laplace_l2_loss,
    shin_min_log_loss,
)
from .inst_phase import (
    EnvelopePhaseLoss,
    InstantaneousPhaseLoss,
    envelope_phase_loss,
    instantaneous_phase_loss,
)
from .nim import NIMLoss, nim_loss
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
from .sinkhorn import SinkhornLoss, sinkhorn_loss
from .student_t import StudentTLoss, student_t_loss
from .traveltime import (
    CrossCorrelationTraveltimeLoss,
    cross_correlation_traveltime_loss,
)
from .w1 import Wasserstein1Loss, w1_loss
from .w2 import Wasserstein2Loss, w2_loss

__all__ = [
    "AWILoss",
    "BaseFWILoss",
    "CauchyLoss",
    "CrossCorrelationTraveltimeLoss",
    "DeconvolutionLoss",
    "EnvelopeLoss",
    "EnvelopePhaseLoss",
    "FrequencyAmplitudeLoss",
    "FrequencyDomainL2Loss",
    "FrequencyPhaseLoss",
    "GemanMcClureLoss",
    "GlobalCorrelationLoss",
    "HuberLoss",
    "HybridL1L2Loss",
    "InstantaneousPhaseLoss",
    "L1Loss",
    "L2Loss",
    "LaplaceL2Loss",
    "LogarithmicShinMinLoss",
    "NIMLoss",
    "PseudoHuberLoss",
    "SinkhornLoss",
    "StudentTLoss",
    "TraceNormalizedL2Loss",
    "TukeyLoss",
    "Wasserstein1Loss",
    "Wasserstein2Loss",
    "awi_loss",
    "cauchy_loss",
    "cross_correlation_traveltime_loss",
    "deconvolution_loss",
    "envelope_loss",
    "envelope_phase_loss",
    "frequency_amplitude_loss",
    "frequency_domain_l2_loss",
    "frequency_phase_loss",
    "geman_mcclure_loss",
    "global_correlation_loss",
    "huber_loss",
    "hybrid_l1l2_loss",
    "instantaneous_phase_loss",
    "l1_loss",
    "l2_loss",
    "laplace_l2_loss",
    "nim_loss",
    "pseudo_huber_loss",
    "shin_min_log_loss",
    "sinkhorn_loss",
    "student_t_loss",
    "trace_normalized_l2_loss",
    "tukey_loss",
    "w1_loss",
    "w2_loss",
]

__version__ = "0.1.0"
