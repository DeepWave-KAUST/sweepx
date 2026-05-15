# References

This page collects the canonical references for every algorithm in
`sweep-opt`. Each algorithm page also links its own primary reference at the
bottom; they are mirrored here for a single bibliographic snapshot.

## General optimization texts

- Nocedal, J., and Wright, S. J. (2006). _Numerical Optimization_, 2nd ed.,
  Springer. [DOI:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)

## Per-algorithm references

### Tier-2

- **Truncated Newton survey**: Nash, S. G. (2000). "A survey of
  truncated-Newton methods." _Journal of Computational and Applied
  Mathematics_ 124(1-2), 45-59.
  [DOI:10.1016/S0377-0427(00)00426-X](https://doi.org/10.1016/S0377-0427(00)00426-X).
- **Eisenstat-Walker forcing**: Eisenstat, S. C., and Walker, H. F. (1996).
  "Choosing the forcing terms in an inexact Newton method." _SIAM J. Sci.
  Comput._ 17(1), 16-32.
  [DOI:10.1137/0917003](https://doi.org/10.1137/0917003).
- **Pearlmutter HVP trick**: Pearlmutter, B. A. (1994). "Fast exact
  multiplication by the Hessian." _Neural Computation_ 6(1), 147-160.
  [DOI:10.1162/neco.1994.6.1.147](https://doi.org/10.1162/neco.1994.6.1.147).
- **Hestenes-Stiefel CG**: Hestenes, M. R., and Stiefel, E. (1952). "Methods
  of conjugate gradients for solving linear systems." _J. Res. NBS_ 49(6),
  409-436.
  [DOI:10.6028/jres.049.044](https://doi.org/10.6028/jres.049.044).
- **Steihaug truncated CG**: Steihaug, T. (1983). "The conjugate gradient
  method and trust regions in large scale optimization." _SIAM J. Numer.
  Anal._ 20(3), 626-637.
  [DOI:10.1137/0720042](https://doi.org/10.1137/0720042).
- **Truncated Newton for FWI**: Metivier, L., Brossier, R., Virieux, J., and
  Operto, S. (2013). "Full waveform inversion and the truncated Newton
  method." _SIAM J. Sci. Comput._ 35(2), B401-B437.
  [DOI:10.1137/120877854](https://doi.org/10.1137/120877854).
- **Levenberg-Marquardt damping**: Levenberg, K. (1944).
  "A method for the solution of certain non-linear problems in least
  squares." _Quarterly of Applied Mathematics_ 2(2), 164-168.
  [DOI:10.1090/qam/10666](https://doi.org/10.1090/qam/10666);
  Marquardt, D. W. (1963).
  [DOI:10.1137/0111030](https://doi.org/10.1137/0111030).
- **Adaptive Subspace Trust-Region (ASTR)**: Wang, Y., and Rao, Y. (2015).
  "Reflection seismic waveform tomography in transversely isotropic media."
  _Geophysical Journal International_ 202(2), 1362-1373.
  [DOI:10.1093/gji/ggv226](https://doi.org/10.1093/gji/ggv226).
- **L-BFGS-B (active-set)**: Byrd, R. H., Lu, P., Nocedal, J., and Zhu, C.
  (1995). "A limited memory algorithm for bound constrained optimization."
  _SIAM J. Sci. Comput._ 16(5), 1190-1208.
  [DOI:10.1137/0916069](https://doi.org/10.1137/0916069).
- **L-BFGS-B (Fortran)**: Zhu, C., Byrd, R. H., Lu, P., and Nocedal, J.
  (1997). "Algorithm 778: L-BFGS-B." _ACM Trans. Math. Soft._ 23(4),
  550-560.
  [DOI:10.1145/279232.279236](https://doi.org/10.1145/279232.279236).

### Tier-3

- **Adam**: Kingma, D. P., and Ba, J. (2015). "Adam: A method for
  stochastic optimization."
  [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).
- **AdamW**: Loshchilov, I., and Hutter, F. (2019). "Decoupled weight
  decay regularization." ICLR.
  [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
- **RAdam**: Liu, L., et al. (2019). "On the variance of the adaptive
  learning rate and beyond."
  [arXiv:1908.03265](https://arxiv.org/abs/1908.03265).
- **NAdam**: Dozat, T. (2016). "Incorporating Nesterov momentum into
  Adam."
  [openreview](https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ).
- **AMSGrad**: Reddi, S. J., et al. (2018). "On the convergence of Adam
  and beyond."
  [arXiv:1904.09237](https://arxiv.org/abs/1904.09237).
- **Lion**: Chen, X., et al. (2023). "Symbolic discovery of optimization
  algorithms."
  [arXiv:2302.06675](https://arxiv.org/abs/2302.06675).
- **SAM**: Foret, P., et al. (2021). "Sharpness-aware minimization for
  efficiently improving generalization." ICLR.
  [arXiv:2010.01412](https://arxiv.org/abs/2010.01412).
- **Shampoo**: Gupta, V., Koren, T., and Singer, Y. (2018). "Shampoo:
  Preconditioned stochastic tensor optimization." ICML.
  [PMLR](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf).
- **SOAP**: Vyas, N., et al. (2024). "SOAP: Improving and stabilizing
  Shampoo using Adam."
  [arXiv:2409.11321](https://arxiv.org/abs/2409.11321).
- **Deep Image Prior**: Lempitsky, V., Vedaldi, A., and Ulyanov, D. (2018).
  "Deep Image Prior." CVPR.
  [DOI:10.1109/CVPR.2018.00984](https://doi.org/10.1109/CVPR.2018.00984).
- **DIP-FWI (geophysics)**: He, T., et al. (2024). "Deep image prior
  full-waveform inversion." _JGR: Machine Learning and Computation_ 1,
  e2024JH000227.
  [DOI:10.1029/2024JH000227](https://doi.org/10.1029/2024JH000227).

### Tier-1

- **Steepest Descent** + **strong-Wolfe line search**: Nocedal & Wright (2006),
  _Numerical Optimization_, Algorithms 3.5-3.6. See above for DOI.
- **Pseudo-Hessian preconditioner**: Plessix, R.-E., and Mulder, W. A. (2004).
  "Frequency-domain finite-difference amplitude-preserving migration."
  _Geophysical Journal International_ 157(3), 975-987.
  [DOI:10.1111/j.1365-246X.2004.02289.x](https://doi.org/10.1111/j.1365-246X.2004.02289.x).
- **Nonlinear CG (PR+) restart**: Powell, M. J. D. (1977). "Restart procedures for
  the conjugate gradient method." _Mathematical Programming_ 12, 241-254.
  [DOI:10.1007/BF01593790](https://doi.org/10.1007/BF01593790).
- **Polak-Ribiere formula**: Polak, E., and Ribiere, G. (1969). "Note sur la
  convergence de methodes de directions conjuguees." _RIRO_ 3(R1), 35-43.
  [DOI:10.1051/m2an/196903R100351](https://doi.org/10.1051/m2an/196903R100351).
- **L-BFGS limited-memory storage**: Nocedal, J. (1980). "Updating quasi-Newton
  matrices with limited storage." _Mathematics of Computation_ 35(151),
  773-782.
  [DOI:10.1090/S0025-5718-1980-0572855-7](https://doi.org/10.1090/S0025-5718-1980-0572855-7).
- **L-BFGS for large-scale optimization**: Liu, D. C., and Nocedal, J. (1989).
  "On the limited memory BFGS method for large scale optimization."
  _Mathematical Programming_ 45, 503-528.
  [DOI:10.1007/BF01589116](https://doi.org/10.1007/BF01589116).

(Additional algorithms appear here as later commits land.)

### Topic indices

- [Optimizers](optimizers/index.md)
- [Preconditioners](preconditioners/index.md)
- [Parameterization](parameterization/index.md)
- [Marmousi report](report.md)
