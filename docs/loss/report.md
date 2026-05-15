# Report — implemented FWI misfits

This page is the formula-by-formula summary of every loss that ships in
`fwiloss`, with the **DOI for each source paper** so you can jump straight
to the literature.  Notation: throughout we let
$d_{\mathrm s} = d_{\mathrm{syn}}$, $d_{\mathrm o} = d_{\mathrm{obs}}$,
$r = d_{\mathrm s} - d_{\mathrm o}$.

The canonical tensor layout is `(nshots, nt, nreceivers, nchannel)`
(see [Data convention](getting_started/data_convention.md)).  Sums are
over all indices unless otherwise noted; many of the losses below are
first defined per-trace and then summed over the trailing axes.

> **This file is also exported in the repo root as `REPORT.md`** so it is
> easy to share without spinning up mkdocs.

## A. Data-domain Lp norms / robust M-estimators

### A.1 L2 / least-squares

$$\mathcal J_{L_2} = \tfrac{1}{2}\sum r^2.$$

Lailly (1983); Tarantola, A. (1984), **Geophysics** 49 (8), 1259-1266.
doi:[10.1190/1.1441754](https://doi.org/10.1190/1.1441754) — see also the
review by Virieux & Operto (2009),
doi:[10.1190/1.3238367](https://doi.org/10.1190/1.3238367).

### A.2 L1

$$\mathcal J_{L_1} = \sum |r|.$$

Crase et al. (1990), **Geophysics** 55 (5), 527-538.
doi:[10.1190/1.1442864](https://doi.org/10.1190/1.1442864);
Brossier, Operto & Virieux (2010), **Geophysics** 75 (3), R37-R46.
doi:[10.1190/1.3379323](https://doi.org/10.1190/1.3379323).

### A.3 Huber norm

$$
h_\delta(r) = \begin{cases}\tfrac{1}{2}r^2 & |r|\le\delta\\ \delta(|r|-\tfrac{1}{2}\delta) & |r|>\delta\end{cases}.
$$

Huber, P. J. (1964), Ann. Math. Stat. 35 (1), 73-101.
doi:[10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732);
Guitton & Symes (2003), **Geophysics** 68 (4), 1310-1319.
doi:[10.1190/1.1598124](https://doi.org/10.1190/1.1598124).

### A.4 Pseudo-Huber / Hybrid L1/L2 (Bube–Langan)

$$
\tilde h_\delta(r) = \delta^2\bigl(\sqrt{1+(r/\delta)^2}-1\bigr).
$$

Bube & Langan (1997), **Geophysics** 62 (4), 1183-1195.
doi:[10.1190/1.1444219](https://doi.org/10.1190/1.1444219);
Ha, Chung & Shin (2009), **Geophysics** 74 (3), R15-R24.
doi:[10.1190/1.3112572](https://doi.org/10.1190/1.3112572);
Charbonnier et al. (1997), IEEE T-IP 6 (2), 298-311.
doi:[10.1109/83.551699](https://doi.org/10.1109/83.551699).

### A.5 Cauchy / Lorentzian

$$
\rho^{\mathrm{C}}_c(r) = \tfrac{c^2}{2}\log\bigl(1+(r/c)^2\bigr).
$$

Crase et al. (1990), **Geophysics** 55 (5), 527-538.
doi:[10.1190/1.1442864](https://doi.org/10.1190/1.1442864);
Black & Anandan (1996), CVIU 63 (1), 75-104.
doi:[10.1006/cviu.1996.0006](https://doi.org/10.1006/cviu.1996.0006).

### A.6 Tukey biweight

$$
\rho^{\mathrm{T}}_c(r) = \begin{cases}\tfrac{c^2}{6}\bigl[1-(1-(r/c)^2)^3\bigr] & |r|\le c\\ c^2/6 & |r|>c\end{cases}.
$$

Beaton & Tukey (1974), **Technometrics** 16, 147-185.
doi:[10.1080/00401706.1974.10489171](https://doi.org/10.1080/00401706.1974.10489171);
Bube & Nemeth (2007), **Geophysics** 72 (2), A13-A17.
doi:[10.1190/1.2431639](https://doi.org/10.1190/1.2431639).

### A.7 Geman–McClure

$$
\rho^{\mathrm{GM}}_c(r) = c^2\frac{(r/c)^2}{1+(r/c)^2}.
$$

Geman & McClure (1985) (no DOI; see
[permanent record](https://www.dam.brown.edu/people/geman/Homepage/Society%20publications/85GemanMcClure.pdf)).

### A.8 Student-t negative log-likelihood *(planned next)*

$$
\rho^{\nu,\sigma}(r) = \tfrac{\nu+1}{2}\log\bigl(1+\tfrac{1}{\nu}(r/\sigma)^2\bigr).
$$

Aravkin, van Leeuwen & Herrmann (2012), SEG Expanded Abstracts.
doi:[10.1190/segam2012-1010.1](https://doi.org/10.1190/segam2012-1010.1).

---

Sections **B–H** below are placeholders that are filled in as each family
is implemented.  Every paper here will carry a DOI when the formula is
added.

## B. Correlation / amplitude-normalised — *implementation pending*

## C. Envelope / instantaneous phase — *implementation pending*

## D. Frequency / Laplace domain — *implementation pending*

## E. Travel-time / picking — *implementation pending*

## F. Convolution / matching filter — *implementation pending*

## G. Normalized integration method — *implementation pending*

## H. Optimal transport — *implementation pending*

## I. Extended / wavefield-coupled (NOT implemented)

These misfits do not have a clean `loss(syn, obs)` interface because they
need the full PDE / extended wavefield as an additional input.  We list
them for completeness in the [References](references.md) page.
