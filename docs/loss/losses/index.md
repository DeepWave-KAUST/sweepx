# Loss functions — overview

`sweep_loss` groups the FWI misfit literature into the following families:

| Family | Examples implemented |
|---|---|
| **A.  Data-domain Lp norms / robust M-estimators** | L2, L1, Huber, pseudo-Huber, hybrid L1/L2, Cauchy, Tukey, Geman–McClure |
| **B.  Correlation / amplitude-normalised** | trace-normalised L2, global correlation (NCC), local similarity (planned) |
| **C.  Envelope / instantaneous-phase (Hilbert)** | envelope, log-envelope ratio, instantaneous phase, combined envelope+phase (planned) |
| **D.  Frequency / Laplace domain** | Pratt frequency-domain L2, Shin-Min log misfit, phase-only & amplitude-only, Laplace, Laplace–Fourier (planned) |
| **E.  Travel-time / picking** | cross-correlation travel-time, instantaneous travel-time, dynamic image warping (planned) |
| **F.  Convolution / matching-filter** | AWI, deconvolution-based, matching-filter penalty (planned) |
| **G.  Normalized integration method (NIM)** | NIM / CDF L2 (planned) |
| **H.  Optimal transport** | W1 / KR-norm, W2 via CDF, Sinkhorn, GSOT (planned) |

Each subsection on the left-hand sidebar gives the formula, the canonical
reference (with DOI) and the tests we ship.

If you came here looking for a one-page summary with every formula, see
the [**Report**](../report.md).
