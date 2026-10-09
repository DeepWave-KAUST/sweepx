# sweep-nn examples

| Example | What it shows | Needs |
|---|---|---|
| [`quickstart.py`](quickstart.py) | The API on a toy problem: a SIREN fitted to a velocity model directly, no wave equation | CPU is enough |
| [`pseudo_hessian_overthrust.ipynb`](pseudo_hessian_overthrust.ipynb) | Implicit FWI on Overthrust, and the pseudo-Hessian (illumination) preconditioner | CUDA GPU, `sweep-solver` |
| [`hash_encoding_marmousi.ipynb`](hash_encoding_marmousi.ipynb) | Implicit FWI on Marmousi, and the multiresolution hash encoding | CUDA GPU, `sweep-solver` |

## The two implicit-FWI notebooks

They reproduce the two synthetic examples of *Accelerating High Resolution Implicit Full
Waveform Inversion* (Shaowen Wang and Tariq Alkhalifah, *Geophysics*). Each runs three
inversions from the same smoothed starting model, conventional FWI, implicit FWI with a
SIREN, and implicit FWI with the method of the example, and plots the paper's figures.
The velocity is `sweep_nn.VelocityINR`, `vp = vp_init + vp_std · net(z, x)`; the wave
equation is sweep's compiled acoustic solver.

```bash
pip install sweep-nn sweep-solver jupyterlab
cd docs/examples                     # the notebooks import ifwi_models.py from here
jupyter lab
```

Set `IFWI_EPOCHS=60` (Overthrust) or `IFWI_EPOCHS=40` (Marmousi) in the environment for a
quick check instead of the full run.

Final model RMSE (m/s), conventional FWI / implicit FWI / implicit FWI + method:

| | this directory | paper ([ifwi-pub](https://github.com/DeepWave-KAUST/ifwi-pub)) | wall time, RTX 6000 Ada |
|---|---|---|---|
| Overthrust, + pseudo-Hessian | 513 / 718 / 200 | 513 / 615 / 189 | 7 min |
| Marmousi, + hash encoding | 298 / 380 / 311 | 298 / 371 / 310 | 2 min |

The networks are the paper's: the same widths, depths, sine frequencies and hash-grid
configuration, and the same parameter counts. They are not the same random draws:
`VelocityINR` initializes its layers in a different order, and it samples coordinates on
`[0, 1)` where the paper used `[0, 1]`. So the numbers land close to the paper's rather
than on them. [ifwi-pub](https://github.com/DeepWave-KAUST/ifwi-pub) is the stand-alone
package that reproduces the paper bit for bit.

## Velocity models

[`ifwi_models.py`](ifwi_models.py) holds the four models the notebooks use, embedded
(float32, zlib + base85), copied unchanged from ifwi-pub. They are derived arrays:
decimated, cropped and, for the starting models, smoothed.

| Model | Derived from | Terms | Cite |
|---|---|---|---|
| `overthrust`, 187 × 401 at 25 m | a 2-D slice of the SEG/EAGE 3-D Overthrust model | CC-BY-4.0 | Aminzadeh, Burkhard, Kunz, Nicoletis & Rocca (1995), *The Leading Edge* 14, 125-128 |
| `marmousi`, 141 × 341 at 25 m | an 8.5 × 3.5 km subset of the Marmousi2 P-wave velocity | public academic open data (AGL / University of Houston) | Martin, Wiley & Marfurt (2006), [10.1190/1.2172306](https://doi.org/10.1190/1.2172306) |

For the original Marmousi2 model, go to the official source rather than to this file.

## Citation

The two methods were first presented as EAGE abstracts:

* *Implicit full waveform inversion with energy-weighted gradient*,
  [10.3997/2214-4609.202510069](https://doi.org/10.3997/2214-4609.202510069) (pseudo-Hessian)
* *Multiresolution hash encoding for high resolution implicit full waveform inversion*,
  [10.3997/2214-4609.202510109](https://doi.org/10.3997/2214-4609.202510109) (hash encoding)
