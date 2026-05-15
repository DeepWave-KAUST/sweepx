# References

This page collects the canonical references for every algorithm in
`sweep-opt`. Each algorithm page also links its own primary reference at the
bottom; they are mirrored here for a single bibliographic snapshot.

## General optimization texts

- Nocedal, J., and Wright, S. J. (2006). _Numerical Optimization_, 2nd ed.,
  Springer. [DOI:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)

## Per-algorithm references

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

(Additional algorithms appear here as later commits land.)

### Topic indices

- [Optimizers](optimizers/index.md)
- [Preconditioners](preconditioners/index.md)
- [Parameterization](parameterization/index.md)
- [Marmousi report](report.md)
