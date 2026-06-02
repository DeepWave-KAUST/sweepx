# `sweep_loss` — formula report & bibliography

This document gathers, in one place, the **mathematical definition and
DOI-linked source paper** of every misfit (loss) implemented in the
`sweep_loss` package.  It is also the canonical landing page for the
mkdocs site (`docs/report.md`).

Tensor layout throughout:

$$
d_{\mathrm s}, d_{\mathrm o} \in \mathbb R^{n_s\times n_t\times n_r\times n_c}
\quad(\text{shots} \times \text{time} \times \text{receivers} \times \text{channels})
$$

with residual $r = d_{\mathrm s} - d_{\mathrm o}$.

| Family | Misfits implemented |
|--------|--------------------|
| **A.** Data-domain Lp / robust M-estimators | L2, L1, Huber, pseudo-Huber, hybrid L1/L2, Cauchy, Tukey, Geman-McClure, Student-t |
| **B.** Correlation / amplitude-normalised   | trace-normalised L2, global correlation (NCC) |
| **C.** Travel-time / picking                | cross-correlation travel-time |
| **D.** Envelope & phase                     | envelope (p=1,2 / log / squared), instantaneous-phase, **exponentiated phase**, envelope+phase combined, **time-frequency phase/envelope** |
| **E.** Frequency / Laplace domain           | freq-L2 (Pratt), phase-only, amplitude-only, Shin-Min log, Laplace L2 |
| **F.** Convolution / matching filter        | Adaptive Waveform Inversion (Warner-Guasch), Luo-Sava deconvolution, **OT of Matching Filter (OTMF)** |
| **G.** Density-domain divergences           | NIM (Liu / Donno), **Jensen-Shannon divergence (Yan 2024)** |
| **H.** Optimal transport                    | W1 (Métivier), W2 via CDF (Engquist-Froese / Yang), Sinkhorn divergence, GSOT (graph-space) |
| **I.** Dynamic warping                      | Soft-DTW |
| **J.** Local attribute                      | local similarity (Fomel) |

A total of **34 PyTorch loss classes** implementing **≈ 38 named misfits**
across families A-J (the variants of `EnvelopeLoss`, `FrequencyXxxLoss`,
`SinkhornLoss(debiased=...)`, `OTMFLoss(order=1|2)` count as separate
misfits in the literature).

> All references below carry a DOI link. Sections D-H reference the same
> "anti-cycle-skipping property" — these are misfits that remain monotone
> with respect to a time shift on a Ricker, well past the half-wavelength
> shift where the plain L2 misfit oscillates.  This is exercised in the
> corresponding test files (`test_<family>.py`).

---

## A. Data-domain Lp norms and robust M-estimators

### A.1 L2 / least-squares — `L2Loss`

$$
\mathcal J_{L_2} = \tfrac12 \sum r^2.
$$

* Lailly, P. (1983). *The seismic inverse problem as a sequence of
  before-stack migrations.* In: *Conference on Inverse Scattering*,
  SIAM, 206-220.  *(no DOI)*
* Tarantola, A. (1984). *Inversion of seismic reflection data in the
  acoustic approximation.* **Geophysics** 49 (8), 1259-1266.
  doi:[10.1190/1.1441754](https://doi.org/10.1190/1.1441754)
* Virieux, J. & Operto, S. (2009). *An overview of full-waveform
  inversion in exploration geophysics.* **Geophysics** 74 (6), WCC1-
  WCC26.
  doi:[10.1190/1.3238367](https://doi.org/10.1190/1.3238367)

### A.2 L1 — `L1Loss`

$$
\mathcal J_{L_1} = \sum |r|.
$$

* Crase, E., Pica, A., Noble, M., McDonald, J. & Tarantola, A. (1990).
  *Robust elastic nonlinear waveform inversion: application to real
  data.* **Geophysics** 55 (5), 527-538.
  doi:[10.1190/1.1442864](https://doi.org/10.1190/1.1442864)
* Brossier, R., Operto, S. & Virieux, J. (2010). *Which data residual
  norm for robust elastic frequency-domain full waveform inversion?*
  **Geophysics** 75 (3), R37-R46.
  doi:[10.1190/1.3379323](https://doi.org/10.1190/1.3379323)

### A.3 Huber — `HuberLoss`

$$
h_\delta(r) = \begin{cases}\tfrac12 r^2 & |r|\le\delta\\ \delta(|r|-\tfrac12\delta) & |r|>\delta\end{cases}.
$$

* Huber, P. J. (1964). *Robust estimation of a location parameter.*
  Ann. Math. Stat. 35 (1), 73-101.
  doi:[10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732)
* Guitton, A. & Symes, W. W. (2003). *Robust inversion of seismic data
  using the Huber norm.* **Geophysics** 68 (4), 1310-1319.
  doi:[10.1190/1.1598124](https://doi.org/10.1190/1.1598124)

### A.4 Pseudo-Huber / Hybrid L1/L2 — `PseudoHuberLoss`, `HybridL1L2Loss`

$$
\tilde h_\delta(r) = \delta^2\bigl(\sqrt{1+(r/\delta)^2}-1\bigr).
$$

* Bube, K. P. & Langan, R. T. (1997). *Hybrid L1/L2 minimisation with
  applications to tomography.* **Geophysics** 62 (4), 1183-1195.
  doi:[10.1190/1.1444219](https://doi.org/10.1190/1.1444219)
* Ha, T., Chung, W. & Shin, C. (2009). *Waveform inversion using a
  back-propagation algorithm and a Huber function norm.* **Geophysics**
  74 (3), R15-R24.
  doi:[10.1190/1.3112572](https://doi.org/10.1190/1.3112572)
* Charbonnier, P., Blanc-Feraud, L., Aubert, G. & Barlaud, M. (1997).
  *Deterministic edge-preserving regularisation in computed imaging.*
  IEEE Trans. Image Process. 6 (2), 298-311.
  doi:[10.1109/83.551699](https://doi.org/10.1109/83.551699)

### A.5 Cauchy / Lorentzian — `CauchyLoss`

$$
\rho^{\mathrm C}_c(r) = \tfrac{c^2}{2}\log(1+(r/c)^2).
$$

* Black, M. J. & Anandan, P. (1996). *The robust estimation of multiple
  motions.* CVIU 63 (1), 75-104.
  doi:[10.1006/cviu.1996.0006](https://doi.org/10.1006/cviu.1996.0006)
* Crase, E., Pica, A., Noble, M., McDonald, J. & Tarantola, A. (1990).
  *Robust elastic nonlinear waveform inversion: application to real
  data.* **Geophysics** 55 (5), 527-538.
  doi:[10.1190/1.1442864](https://doi.org/10.1190/1.1442864)

### A.6 Tukey biweight — `TukeyLoss`

$$
\rho^{\mathrm T}_c(r) = \begin{cases}\tfrac{c^2}{6}\bigl(1-(1-(r/c)^2)^3\bigr) & |r|\le c\\ c^2/6 & |r|>c\end{cases}.
$$

* Beaton, A. E. & Tukey, J. W. (1974). *The fitting of power series,
  meaning polynomials, illustrated on band-spectroscopic data.*
  Technometrics 16, 147-185.
  doi:[10.1080/00401706.1974.10489171](https://doi.org/10.1080/00401706.1974.10489171)
* Bube, K. P. & Nemeth, T. (2007). *Fast line searches for the robust
  solution of linear systems in the hybrid l1/l2 and Huber norms.*
  **Geophysics** 72 (2), A13-A17.
  doi:[10.1190/1.2431639](https://doi.org/10.1190/1.2431639)

### A.7 Geman-McClure — `GemanMcClureLoss`

$$
\rho^{\mathrm{GM}}_c(r) = c^2 \frac{(r/c)^2}{1+(r/c)^2}.
$$

* Geman, S. & McClure, D. E. (1985). *Bayesian image analysis: an
  application to single photon emission tomography.*  Proc. Stat. Comp.
  Sect., ASA, 12-18.  *(no DOI; see
  [permanent record](https://www.dam.brown.edu/people/geman/Homepage/Society%20publications/85GemanMcClure.pdf))*

### A.8 Student-t negative log-likelihood — `StudentTLoss`

$$
\rho^{\nu,\sigma}(r) = \tfrac{\nu+1}{2}\log\bigl(1+\tfrac{1}{\nu}(r/\sigma)^2\bigr).
$$

Limits: $\nu = 1$ is Cauchy; $\nu\to\infty$ is Gaussian / L2.

* Aravkin, A. Y., van Leeuwen, T. & Herrmann, F. J. (2011). *Robust FWI
  using Student-t distribution.* SEG Tech. Progr. Expanded Abstracts,
  pp. 2669-2673.
  doi:[10.1190/1.3627747](https://doi.org/10.1190/1.3627747)
* Aravkin, A., Burke, J. V. & Friedlander, M. P. (2013). *Variational
  properties of value functions.* SIAM J. Optim. 23 (3), 1689-1717.
  doi:[10.1137/120899157](https://doi.org/10.1137/120899157)

---

## B. Correlation / amplitude-normalised

### B.1 Global correlation (NCC) — `GlobalCorrelationLoss`

Per trace with unit-norm $\hat d = d/\|d\|_2$:

$$
\mathcal J_{\mathrm{NCC}} = \sum_{\mathrm{trace}}(1-\langle\hat d_{\mathrm s},\hat d_{\mathrm o}\rangle).
$$

* Choi, Y. & Alkhalifah, T. (2012). *Application of multi-source
  waveform inversion to marine streamer data using the global
  correlation norm.* **Geophysical Prospecting** 60 (4), 748-758.
  doi:[10.1111/j.1365-2478.2012.01079.x](https://doi.org/10.1111/j.1365-2478.2012.01079.x)
* Routh, P., Krebs, J., Lazaratos, S., et al. (2011). SEG Tech. Progr.
  Expanded Abstracts, 2433-2438.
  doi:[10.1190/1.3627697](https://doi.org/10.1190/1.3627697)

### B.2 Trace-normalised L2 — `TraceNormalizedL2Loss`

$$
\mathcal J_{\mathrm{nL2}} = \tfrac12\sum_{\mathrm{trace}}\sum_t (\hat d_{\mathrm s}-\hat d_{\mathrm o})^2 = \mathcal J_{\mathrm{NCC}}.
$$

*Same DOIs as B.1.*

---

## C. Travel-time / picking

### C.1 Cross-correlation travel-time — `CrossCorrelationTraveltimeLoss`

Per trace, with $\tau^\star = \arg\max_\tau \int d_{\mathrm s}(t) d_{\mathrm o}(t+\tau)\,dt$:

$$
\mathcal J_{\mathrm{CCT}} = \tfrac12 \sum_{\text{trace}} (\tau^\star)^2.
$$

The smooth, differentiable surrogate replaces $\arg\max$ with a
correlation-weighted centroid (van Leeuwen & Mulder 2010 form).

* Luo, Y. & Schuster, G. T. (1991). *Wave-equation travel-time
  inversion.* **Geophysics** 56 (5), 645-653.
  doi:[10.1190/1.1443081](https://doi.org/10.1190/1.1443081)
* van Leeuwen, T. & Mulder, W. A. (2010). *A correlation-based misfit
  criterion for wave-equation traveltime tomography.* **Geophys. J.
  Int.** 182 (3), 1383-1394.
  doi:[10.1111/j.1365-246X.2010.04681.x](https://doi.org/10.1111/j.1365-246X.2010.04681.x)
* Marquering, H., Dahlen, F. A. & Nolet, G. (1999). *Three-dimensional
  sensitivity kernels for finite-frequency traveltimes.* **Geophys. J.
  Int.** 137 (3), 805-815.
  doi:[10.1046/j.1365-246x.1999.00837.x](https://doi.org/10.1046/j.1365-246x.1999.00837.x)
* Wang, S., Song, P., Tan, J., Xia, D., Zhao, B. & Mao, S. (2024). *Differentiable Traveltime Misfit for Wave-Equation
  Tomography.* 85th EAGE Annual Conference & Exhibition, Oslo, Norway,
  Expanded Abstracts. doi:[10.3997/2214-4609.202410170](https://doi.org/10.3997/2214-4609.202410170). Replaces the
  power-of-cross-correlation centroid (van Leeuwen-Mulder 2010, our
  default) with a softmax-of-cross-correlation centroid; same family of
  smooth-argmax surrogates.

---

## D. Envelope and instantaneous-phase misfits

Let $a(t)=d(t)+i\mathcal H[d](t) = E(t) e^{i\phi(t)}$ be the analytic
signal.

### D.1 Envelope misfits — `EnvelopeLoss`

* $\mathcal J_E^{(p)} = \tfrac{1}{p}\sum |E_{\mathrm s}-E_{\mathrm o}|^p$ for $p\in\{1,2\}$  (Wu 2014)
* $\mathcal J_E^{\log} = \tfrac12\sum(\log E_{\mathrm s}-\log E_{\mathrm o})^2$  (Bozdağ 2011)
* $\mathcal J_{E^2} = \tfrac12\sum(E_{\mathrm s}^2-E_{\mathrm o}^2)^2$  (Chi 2014)

* Bozdağ, E., Trampert, J. & Tromp, J. (2011). *Misfit functions for
  full waveform inversion based on instantaneous phase and envelope
  measurements.* **Geophys. J. Int.** 185 (2), 845-870.
  doi:[10.1111/j.1365-246X.2011.04970.x](https://doi.org/10.1111/j.1365-246X.2011.04970.x)
* Wu, R.-S., Luo, J. & Wu, B. (2014). *Seismic envelope inversion and
  modulation signal model.* **Geophysics** 79 (3), WA13-WA24.
  doi:[10.1190/geo2013-0294.1](https://doi.org/10.1190/geo2013-0294.1)
* Chi, B., Dong, L. & Liu, Y. (2014). *Full waveform inversion method using
  envelope objective function without low frequency data.*  **J. Appl. Geophys.** 109, 36-46.
  doi:[10.1016/j.jappgeo.2014.07.010](https://doi.org/10.1016/j.jappgeo.2014.07.010)

### D.2 Instantaneous-phase misfit — `InstantaneousPhaseLoss`

$$
\mathcal J_\phi = \tfrac12 \sum [w_\phi(t)\Delta\phi(t)]^2,
\qquad \Delta\phi = \arctan2(\sin(\phi_{\mathrm s}-\phi_{\mathrm o}),\cos(\phi_{\mathrm s}-\phi_{\mathrm o})).
$$

* Bozdağ et al. (2011) — same DOI as D.1.
* Fichtner, A., Kennett, B. L. N., Igel, H. & Bunge, H.-P. (2008).
  *Theoretical background for continental- and global-scale full-waveform
  inversion in the time-frequency domain.* **Geophys. J. Int.** 175 (2),
  665-685.
  doi:[10.1111/j.1365-246X.2008.03923.x](https://doi.org/10.1111/j.1365-246X.2008.03923.x)

### D.3 Exponentiated phase — `ExponentiatedPhaseLoss`

Normalised analytic signal $\tilde s(t) = a(t)/E_s(t) = e^{i\phi_s(t)}$:

$$
\chi_{\mathrm{EP}} = \tfrac{1}{2}\sum\int_0^T |\Re\tilde s-\Re\tilde d|^2 + |\Im\tilde s-\Im\tilde d|^2 \,dt.
$$

* Yuan, Y. O., Bozdağ, E., Ciardelli, C., Gao, F. & Simons, F. J. (2020).
  *The exponentiated phase measurement, and objective-function
  hybridisation for adjoint waveform tomography.* **Geophys. J. Int.**
  221 (2), 1145-1164.
  doi:[10.1093/gji/ggaa063](https://doi.org/10.1093/gji/ggaa063)
* Gao, F., Yuan, Y. O., Ciardelli, C., Simons, F. J., Bozdağ, E. &
  Tromp, J. (2023). *Review of misfit functions for adjoint full waveform
  inversion in seismology.* **Geophys. J. Int.** 235 (3), 2794-2820.
  doi:[10.1093/gji/ggad372](https://doi.org/10.1093/gji/ggad372)

### D.4 Envelope + phase combined — `EnvelopePhaseLoss`

$$
\mathcal J_{E+\phi} = (1-\alpha)\,\mathcal J_E + \alpha\,\mathcal J_\phi.
$$

* Yuan, Y. O., Simons, F. J. & Tromp, J. (2016). *Double-difference
  adjoint seismic tomography.* **Geophys. J. Int.** 206 (3), 1599-1618.
  doi:[10.1093/gji/ggw233](https://doi.org/10.1093/gji/ggw233)

### D.5 Time-Frequency phase / envelope — `TimeFrequencyPhaseLoss`

Gabor STFT $G_s, G_o$ with magnitude $A$ and phase $\phi$. Envelope
term $\tfrac{1}{2}\sum_{t,\omega}(|G_s|-|G_o|)^2$; phase term uses the
unit-circle residual $|e^{i\phi_s}-e^{i\phi_o}|^2$ weighted by $|G_o|$.
Mixed by $\alpha$.

* Fichtner, A., Kennett, B. L. N., Igel, H. & Bunge, H.-P. (2008).
  *Theoretical background for continental- and global-scale full-waveform
  inversion in the time-frequency domain.* **Geophys. J. Int.** 175 (2),
  665-685.
  doi:[10.1111/j.1365-246X.2008.03923.x](https://doi.org/10.1111/j.1365-246X.2008.03923.x)
* Kristeková, M., Kristek, J. & Moczo, P. (2009). *Time-frequency misfit
  and goodness-of-fit criteria for quantitative comparison of time
  signals.* **Geophys. J. Int.** 178 (2), 813-825.
  doi:[10.1111/j.1365-246X.2009.04177.x](https://doi.org/10.1111/j.1365-246X.2009.04177.x)
* Kristeková, M., Kristek, J., Moczo, P. & Day, S. M. (2006). *Misfit
  criteria for quantitative comparison of seismograms.* **Bull. Seismol.
  Soc. Am.** 96 (5), 1836-1850.
  doi:[10.1785/0120060012](https://doi.org/10.1785/0120060012)

---

## E. Frequency / Laplace domain

### E.1 Frequency-domain L2 — `FrequencyDomainL2Loss`

$$
\mathcal J_\Omega = \tfrac12 \sum_{\omega\in\Omega} |D_{\mathrm s}-D_{\mathrm o}|^2.
$$

* Pratt, R. G., Shin, C. & Hicks, G. J. (1998). *Gauss-Newton and full
  Newton methods in frequency-space seismic waveform inversion.*
  **Geophys. J. Int.** 133 (2), 341-362.
  doi:[10.1046/j.1365-246X.1998.00498.x](https://doi.org/10.1046/j.1365-246X.1998.00498.x)
* Bunks, C., Saleck, F. M., Zaleski, S. & Chavent, G. (1995).
  *Multiscale seismic waveform inversion.* **Geophysics** 60 (5),
  1457-1473.
  doi:[10.1190/1.1443880](https://doi.org/10.1190/1.1443880)

### E.2 Phase-only — `FrequencyPhaseLoss`

$$
\mathcal J_\phi^{(\Omega)} = \tfrac12 \sum_{\omega\in\Omega} |\Delta\Phi(\omega)|^2.
$$

* Bednar, J. B., Shin, C. & Pyun, S. (2007). *Comparison of waveform
  inversion, part 2: phase approach.* **Geophys. Prospect.** 55 (4),
  465-475.
  doi:[10.1111/j.1365-2478.2007.00618.x](https://doi.org/10.1111/j.1365-2478.2007.00618.x)

### E.3 Amplitude-only — `FrequencyAmplitudeLoss`

$$
\mathcal J_A = \tfrac12 \sum_{\omega\in\Omega} (|D_{\mathrm s}|-|D_{\mathrm o}|)^2.
$$

* Shin, C. & Min, D.-J. (2006). *Waveform inversion using a logarithmic
  wavefield.* **Geophysics** 71 (3), R31-R42.
  doi:[10.1190/1.2194523](https://doi.org/10.1190/1.2194523)

### E.4 Shin-Min log misfit — `LogarithmicShinMinLoss`

$$
\mathcal J_{\log} = \tfrac12 \sum_{\omega\in\Omega} |\log D_{\mathrm s}-\log D_{\mathrm o}|^2
                 = \tfrac12 \sum (\log|D_{\mathrm s}|-\log|D_{\mathrm o}|)^2 + \tfrac12\sum|\Delta\Phi|^2.
$$

* Shin & Min (2006) — same DOI as E.3.
* Choi, Y. & Alkhalifah, T. (2013). *Frequency-domain waveform
  inversion using the phase derivative.* **Geophys. J. Int.** 195 (3),
  1904-1916.
  doi:[10.1093/gji/ggt351](https://doi.org/10.1093/gji/ggt351)

### E.5 Laplace L2 — `LaplaceL2Loss`

$$
\mathcal J_{\mathrm{Laplace}} = \tfrac12 \sum_t (e^{-s t}(d_{\mathrm s} - d_{\mathrm o}))^2.
$$

* Shin, C. & Cha, Y. H. (2008). *Waveform inversion in the Laplace
  domain.* **Geophys. J. Int.** 173 (3), 922-931.
  doi:[10.1111/j.1365-246X.2008.03768.x](https://doi.org/10.1111/j.1365-246X.2008.03768.x)
* Shin, C. & Cha, Y. H. (2009). *Waveform inversion in the
  Laplace-Fourier domain.* **Geophys. J. Int.** 177 (3), 1067-1079.
  doi:[10.1111/j.1365-246X.2009.04102.x](https://doi.org/10.1111/j.1365-246X.2009.04102.x)

---

## F. Convolution / matching filter

### F.1 Adaptive Waveform Inversion (AWI) — `AWILoss`

For each trace, with Wiener filter
$w = \arg\min \|d_{\mathrm s}\ast w - d_{\mathrm o}\|^2 + \epsilon\|w\|^2$:

$$
\mathcal J_{\mathrm{AWI}} = \tfrac12 \sum_{\mathrm{trace}} \frac{\sum_\tau (T(\tau) w(\tau))^2}{\sum_\tau w(\tau)^2},\quad T(\tau) = \tau\cdot\Delta t.
$$

* Warner, M. & Guasch, L. (2016). *Adaptive waveform inversion:
  theory.* **Geophysics** 81 (6), R429-R445.
  doi:[10.1190/geo2015-0387.1](https://doi.org/10.1190/geo2015-0387.1)
* Warner, M. & Guasch, L. (2014). SEG Tech. Progr. Expanded Abstracts,
  1089-1093.
  doi:[10.1190/segam2014-0371.1](https://doi.org/10.1190/segam2014-0371.1)
* Guasch, L., Warner, M. & Ravaut, C. (2019). *Adaptive waveform
  inversion: practice.* **Geophysics** 84 (3), R447-R461.
  doi:[10.1190/geo2018-0377.1](https://doi.org/10.1190/geo2018-0377.1)

### F.2 Optimal Transport of Matching Filter — `OTMFLoss`

Compute the Wiener matching filter $w(\tau)$ as in AWI, preprocess to a
density $\hat w$, then measure Wasserstein distance to the Dirac at zero
lag:

$$
W_2^2(\hat w, \delta_0) = \int \tau^2\,\hat w(\tau)\,d\tau, \qquad W_1(\hat w, \delta_0) = \int |\tau|\,\hat w(\tau)\,d\tau.
$$

* Sun, B. & Alkhalifah, T. (2019). *Adaptive traveltime inversion.*
  **Geophysics** 84 (4), U13-U29.
  doi:[10.1190/geo2018-0595.1](https://doi.org/10.1190/geo2018-0595.1)
* Sun, B. & Alkhalifah, T. (2019). *The application of an optimal
  transport to a preconditioned data matching function for robust
  waveform inversion.* **Geophysics** 84 (6), R923-R945.
  doi:[10.1190/geo2018-0413.1](https://doi.org/10.1190/geo2018-0413.1)
* Sun, B. & Alkhalifah, T. (2019). *Stereo optimal transport of the
  matching filter.*  SEG Tech. Progr. Expanded Abstracts, pp. 1505-1509.
  doi:[10.1190/segam2019-3199662.1](https://doi.org/10.1190/segam2019-3199662.1)

### F.3 Deconvolution-based — `DeconvolutionLoss`

$$
\Psi(\tau) = \mathcal F^{-1}\!\Bigl(\tfrac{D_{\mathrm o}}{D_{\mathrm s}+\epsilon}\Bigr),\quad
\mathcal J_{\mathrm{Decon}} = \tfrac12 \sum \tau^2 \Psi(\tau)^2.
$$

* Luo, S. & Sava, P. (2011). *A deconvolution-based objective function
  for wave-equation inversion.* SEG Tech. Progr. Expanded Abstracts,
  2788-2792.
  doi:[10.1190/1.3627773](https://doi.org/10.1190/1.3627773)
* Choi, Y. & Alkhalifah, T. (2018). *Time-domain full-waveform
  inversion of exponentially damped wavefield using the
  deconvolution-based objective function.* **Geophysics** 83 (2),
  R77-R88.
  doi:[10.1190/geo2017-0057.1](https://doi.org/10.1190/geo2017-0057.1)

---

## G. Density-domain divergences

### G.1 NIM (CDF distance) — `NIMLoss`

$$
f = \sigma(d),\quad F(t) = \tfrac{\int_0^t f}{\int_0^T f},\quad \mathcal J_{\mathrm{NIM}} = \tfrac12 \sum \int_0^T (F_{\mathrm s}-F_{\mathrm o})^2\, dt.
$$

* Donno, D., Chauris, H. & Calandra, H. (2013). 75th EAGE Conf.
  doi:[10.3997/2214-4609.20130411](https://doi.org/10.3997/2214-4609.20130411)

### G.2 Jensen-Shannon divergence — `JensenShannonLoss`

$$
\mathrm{JSD}(p, q) = \tfrac12\mathrm{KL}(p\|m) + \tfrac12\mathrm{KL}(q\|m),\quad m = \tfrac12(p+q).
$$

Symmetric, bounded by $\log 2$.

* Yan, Y., Chen, X., Li, J., et al. (2024). *Multiparameter
  shallow-seismic waveform inversion based on the Jensen-Shannon
  divergence.* **Geophys. J. Int.** 238 (1), 132-155.
  doi:[10.1093/gji/ggae143](https://doi.org/10.1093/gji/ggae143)
* Endres, D. M. & Schindelin, J. E. (2003). *A new metric for
  probability distributions.* **IEEE T. Inf. Theory** 49 (7), 1858-1860.
  doi:[10.1109/TIT.2003.813506](https://doi.org/10.1109/TIT.2003.813506)
* Lin, J. (1991). *Divergence measures based on the Shannon entropy.*
  **IEEE T. Inf. Theory** 37 (1), 145-151.
  doi:[10.1109/18.61115](https://doi.org/10.1109/18.61115)

---

## H. Optimal transport

### H.1 1-Wasserstein — `Wasserstein1Loss`

$$
W_1(f, g) = \int_0^T |F(t)-G(t)|\, dt.
$$

* Métivier, L., Brossier, R., Mérigot, Q., Oudet, E. & Virieux, J.
  (2016). *Measuring the misfit between seismograms using an optimal
  transport distance: application to full waveform inversion.*
  **Geophys. J. Int.** 205 (1), 345-377.
  doi:[10.1093/gji/ggw014](https://doi.org/10.1093/gji/ggw014)
* Engquist, B. & Froese, B. D. (2014). *Application of the Wasserstein
  metric to seismic signals.* **Commun. Math. Sci.** 12 (5), 979-988.
  doi:[10.4310/CMS.2014.v12.n5.a7](https://doi.org/10.4310/CMS.2014.v12.n5.a7)

### H.2 2-Wasserstein via CDF transport — `Wasserstein2Loss`

$$
W_2^2(f, g) = \int_0^1 (F^{-1}(z)-G^{-1}(z))^2\, dz.
$$

* Engquist, Froese & Yang (2016).  *Optimal transport for seismic full
  waveform inversion.*  **Commun. Math. Sci.** 14 (8), 2309-2330.
  doi:[10.4310/CMS.2016.v14.n8.a9](https://doi.org/10.4310/CMS.2016.v14.n8.a9)
* Yang, Y., Engquist, B., Sun, J. & Hamfeldt, B. F. (2018).
  *Application of optimal transport and the quadratic Wasserstein
  metric to full-waveform inversion.* **Geophysics** 83 (1), R43-R62.
  doi:[10.1190/geo2016-0663.1](https://doi.org/10.1190/geo2016-0663.1)

### H.3 Sinkhorn / entropic OT — `SinkhornLoss`

$$
\mathrm{OT}_\varepsilon(p, q) = \min_{\Pi \in \Pi(p,q)}\sum_{ij} C_{ij}\Pi_{ij} + \varepsilon \sum_{ij} \Pi_{ij}(\log\Pi_{ij}-1),
$$
$$
S_\varepsilon(p, q) = \mathrm{OT}_\varepsilon(p,q) - \tfrac12 \mathrm{OT}_\varepsilon(p,p) - \tfrac12 \mathrm{OT}_\varepsilon(q,q).
$$

* Cuturi, M. (2013). *Sinkhorn distances: lightspeed computation of
  optimal transport.* **NeurIPS** 26.
  arXiv:[1306.0895](https://arxiv.org/abs/1306.0895)
* Feydy, J., Séjourné, T., Vialard, F.-X., Amari, S., Trouvé, A. &
  Peyré, G. (2019). *Interpolating between optimal transport and MMD
  using Sinkhorn divergences.*  AISTATS 22, 2681-2690.
  arXiv:[1810.08278](https://arxiv.org/abs/1810.08278)
* Schmitzer, B. (2019). *Stabilised sparse scaling algorithms for
  entropy regularised transport problems.* **SIAM J. Sci. Comput.** 41
  (3), A1443-A1481.
  doi:[10.1137/16M1106018](https://doi.org/10.1137/16M1106018)

### H.4 Graph-space OT — `GSOTLoss`

Per-trace assignment on $(t, d)$ with ground cost
$c(\cdot, \cdot) = \eta(t-t')^2 + (d-d')^2$ solved by Hungarian.

* Métivier, L., Allain, A., Brossier, R., Mérigot, Q., Oudet, E. &
  Virieux, J. (2018). *Optimal transport for mitigating cycle skipping
  in FWI: a graph-space transform approach.* **Geophysics** 83 (5),
  R515-R540.
  doi:[10.1190/geo2017-0807.1](https://doi.org/10.1190/geo2017-0807.1)
* Jonker, R. & Volgenant, A. (1987). *A shortest augmenting path
  algorithm for dense and sparse linear assignment problems.*
  **Computing** 38 (4), 325-340.
  doi:[10.1007/BF02278710](https://doi.org/10.1007/BF02278710)

---

## I. Dynamic warping

### I.1 Soft-DTW — `SoftDTWLoss`

$$
\mathrm{sDTW}_\gamma(d_{\mathrm s}, d_{\mathrm o}) = -\gamma\log\sum_{A\in\mathcal A} \exp\!\bigl(-\tfrac{1}{\gamma}\langle A, \Delta\rangle\bigr),\quad \Delta_{ij} = (d_{\mathrm s,i}-d_{\mathrm o,j})^2.
$$

* Cuturi, M. & Blondel, M. (2017). *Soft-DTW: a differentiable loss
  function for time-series.* **ICML** 70, 894-903.
  arXiv:[1703.01541](https://arxiv.org/abs/1703.01541)
* Hale, D. (2013). *Dynamic warping of seismic images.* **Geophysics**
  78 (2), S105-S115.
  doi:[10.1190/geo2012-0327.1](https://doi.org/10.1190/geo2012-0327.1)
* Ma, Y. & Hale, D. (2013). *Wave-equation reflection traveltime
  inversion with dynamic warping and full-waveform inversion.*
  **Geophysics** 78 (6), R223-R233.
  doi:[10.1190/geo2013-0004.1](https://doi.org/10.1190/geo2013-0004.1)

---

## J. Local-similarity attribute

### J.1 Local similarity — `LocalSimilarityLoss`

$$
\gamma_\sigma(\tau) = \frac{\sum_t w_\sigma(t-\tau) d_{\mathrm s}(t) d_{\mathrm o}(t)}{\sqrt{\sum_t w_\sigma(t-\tau) d_{\mathrm s}^2 \cdot \sum_t w_\sigma(t-\tau) d_{\mathrm o}^2}},
\quad
\mathcal J_{\mathrm{LS}} = \tfrac12 \sum w_\sigma^E(\tau)\,(1-\gamma_\sigma(\tau))^2.
$$

* Fomel, S. (2007). *Local seismic attributes.* **Geophysics** 72 (3),
  A29-A33.
  doi:[10.1190/1.2437573](https://doi.org/10.1190/1.2437573)

---

## Misfits NOT implemented (and why)

These FWI objectives **couple the misfit with the forward problem** or
require *additional* inputs (extended wavefields, model perturbations,
…) that don't fit into a pure `loss(syn, obs)` interface.  They are
listed here for completeness:

* Symes, W. W. & Carazzone, J. J. (1991). *Velocity inversion by
  differential semblance optimisation.* **Geophysics** 56 (5), 654-663.
  doi:[10.1190/1.1443082](https://doi.org/10.1190/1.1443082)
* van Leeuwen, T. & Herrmann, F. J. (2013). *Mitigating local minima in
  full-waveform inversion by expanding the search space.* **Geophys.
  J. Int.** 195 (1), 661-667.
  doi:[10.1093/gji/ggt258](https://doi.org/10.1093/gji/ggt258)
* Chauris, H. & Plessix, R.-E. (2012). *Investigating the differential
  waveform inversion.*  74th EAGE Conf.
  doi:[10.3997/2214-4609.20149790](https://doi.org/10.3997/2214-4609.20149790)

These can be added later by exposing the additional inputs (extended
gather offsets, the wavefield, the model gradient, …) as constructor
arguments — but the consensus is to treat them as separate inversion
*frameworks* rather than `loss(syn, obs)` losses.

---

## Plugging `sweep_loss` into the `sweep` propagator

The misfit API matches the one already used in
`geophyai/examples/FWI/2d/acoustic/torch/_fwi_marmousi_common.py`,
which currently computes the L2 misfit inline as

```python
loss_sum = (syn - obs_batch).pow(2).sum()
(loss_sum / normalization_elements).backward()
```

Replacing it with any `sweep_loss` misfit is a two-line change:

```python
from sweep_loss import HuberLoss          # or any other misfit in this report
loss_fn = HuberLoss(delta=0.5, reduction="sum")

# inside the training loop:
loss_value = loss_fn(syn, obs_batch)   # canonical (ns, nt, nr, nc)
loss_value.backward()
```

For misfits that need extra parameters (`dt`, `epsilon`, `freq_band`,
`gamma`, ...), see the per-loss docs under `docs/losses/`.

---

## How to cite

> Wang, S. (2026). *sweep-loss: a PyTorch library of misfit functions for
> Full Waveform Inversion.*  https://github.com/DeepWave-KAUST/sweep-loss

Per-loss citations: see the DOI links above.
