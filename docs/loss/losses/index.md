# Loss functions — overview

`sweepx.loss` groups the FWI misfit literature into the following families:

| Family | Implemented |
|---|---|
| **A.  Data-domain Lp norms / robust M-estimators** | L2, L1, Huber, pseudo-Huber, hybrid L1/L2, Cauchy, Tukey, Geman–McClure, Student-t |
| **B.  Correlation / amplitude-normalised** | trace-normalised L2, global correlation (NCC), local similarity |
| **C.  Envelope / phase (Hilbert)** | envelope, log-envelope ratio, instantaneous phase, combined envelope+phase, exponentiated phase, time-frequency phase |
| **D.  Frequency / Laplace domain** | Pratt frequency-domain L2, Shin-Min log misfit, phase-only & amplitude-only, Laplace |
| **E.  Travel-time / warping** | cross-correlation travel-time, soft-DTW |
| **F.  Convolution / matching-filter** | AWI, deconvolution-based, OT of the matching filter (OTMF) |
| **G.  Normalized integration method (NIM)** | NIM / CDF L2 |
| **H.  Optimal transport and divergences** | W1 / KR-norm, W2 via CDF, Sinkhorn, GSOT, Jensen–Shannon divergence |

Each subsection on the left-hand sidebar gives the formula, the canonical
reference (with DOI) and the tests we ship.

If you came here looking for a one-page summary with every formula, see
the [**Report**](../report.md).
