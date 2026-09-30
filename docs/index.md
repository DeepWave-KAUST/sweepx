---
template: home.html
hide:
  - navigation
  - toc
---

<section class="sweep-features-3">
  <a class="sweep-features-3__card" href="solver/">
    <div class="sweep-features-3__tag">PACKAGE · import sweep</div>
    <h3 class="sweep-features-3__title">sweep-solver</h3>
    <p class="sweep-features-3__desc">
      The differentiable wave-equation <strong>engine</strong>. Nine equation
      families — acoustic, elastic, VTI, TTI, VRZ, SEM — on torch / JAX / native-CUDA
      backends, with FWI, LSRTM and RTM building blocks.
    </p>
    <span class="sweep-features-3__link">Open the solver docs <span aria-hidden="true">→</span></span>
  </a>
  <a class="sweep-features-3__card" href="agent/">
    <div class="sweep-features-3__tag">PACKAGE · import sweep_agent</div>
    <h3 class="sweep-features-3__title">sweep-agent</h3>
    <p class="sweep-features-3__desc">
      The natural-language <strong>control layer</strong>. Say what you want and a
      <em>local</em> LLM (Ollama / vLLM) turns it into a validated <code>sweep</code>
      run — forward modelling, benchmark models, plots.
    </p>
    <span class="sweep-features-3__link">Open the agent docs <span aria-hidden="true">→</span></span>
  </a>
  <a class="sweep-features-3__card" href="https://github.com/DeepWave-KAUST">
    <div class="sweep-features-3__tag">COMPANIONS · GROWING</div>
    <h3 class="sweep-features-3__title">…and the family</h3>
    <p class="sweep-features-3__desc">
      <code>sweep-tasks</code> (production FWI/LSRTM runner), <code>sweep-loss</code>,
      <code>sweep-nn</code>, <code>sweep-tomo</code> — each an independent package,
      added to <code>sweepx</code> as it ships.
    </p>
    <span class="sweep-features-3__link">Browse the org <span aria-hidden="true">→</span></span>
  </a>
</section>

<!-- What's new: six cards, rotate on release. See docs-template/README.md. -->
<section class="sweep-new">
<div class="sweep-new__eyebrow">WHAT'S NEW</div>
<h2 class="sweep-new__title">Recently shipped.</h2>
<p class="sweep-new__lede">The latest user-visible additions across the stack. The badge says where each one lives: a released version you get from <code>pip install</code>, or <span class="sweep-new__badge sweep-new__badge--dev">dev</span> for what is merged but not yet on PyPI — these docs are built from the development branch.</p>
<div class="sweep-new__grid">

  <a class="sweep-new__card" href="solver/getting-started/installation/">
    <div class="sweep-new__hd"><span class="sweep-new__badge">v0.3.0</span><span class="sweep-new__date">2026-09</span></div>
    <div class="sweep-new__name">Prebuilt CUDA core</div>
    <div class="sweep-new__desc">The wheel carries the compiled kernels, so <code>impl='c'</code> runs straight after <code>pip install</code> — no nvcc, no first-use build, one wheel for any PyTorch version.</div>
  </a>

  <a class="sweep-new__card" href="solver/user-guide/equations/">
    <div class="sweep-new__hd"><span class="sweep-new__badge">v0.3.0</span><span class="sweep-new__date">2026-08</span></div>
    <div class="sweep-new__name">Elastic TTI in 3-D</div>
    <div class="sweep-new__desc">A tilted symmetry axis in three dimensions, where the azimuth finally matters — plus the Oh-2020 displacement form on <code>(v<sub>h</sub>, &eta;)</code>, held stable by a multiaxial PML. CUDA gradients for every parameter.</div>
  </a>

  <a class="sweep-new__card" href="solver/user-guide/parallel/">
    <div class="sweep-new__hd"><span class="sweep-new__badge">v0.2.0</span><span class="sweep-new__date">2026-08</span></div>
    <div class="sweep-new__name">Domain decomposition</div>
    <div class="sweep-new__desc">A model too big for one GPU now splits across several — one tile per rank, a halo per step, and a gradient that is <em>bit-identical</em> to the single-GPU one.</div>
  </a>

  <a class="sweep-new__card" href="solver/user-guide/propagators/#memory-saving-features">
    <div class="sweep-new__hd"><span class="sweep-new__badge">v0.2.0</span><span class="sweep-new__date">2026-08</span></div>
    <div class="sweep-new__name">Pick how the gradient is stored</div>
    <div class="sweep-new__desc">Full, boundary or checkpoint — one three-way choice, same on both backends. Plus thinner CPML buffers and a truncated reverse pass for steady-state objectives.</div>
  </a>

  <a class="sweep-new__card" href="agent/">
    <div class="sweep-new__hd"><span class="sweep-new__badge">agent 0.0.3</span><span class="sweep-new__date">2026-08</span></div>
    <div class="sweep-new__name">Natural-language control</div>
    <div class="sweep-new__desc">Natural-language control on a plain <code>pip install</code>: forward modelling (acoustic <em>and</em> elastic) plus benchmark models, no <code>sweep-tasks</code> needed.</div>
  </a>

  <a class="sweep-new__card" href="solver/notebooks/24_wavefield_per_edge_free_surface/">
    <div class="sweep-new__hd"><span class="sweep-new__badge">v0.2.0</span><span class="sweep-new__date">2026-08</span></div>
    <div class="sweep-new__name">Per-edge free surface</div>
    <div class="sweep-new__desc">Turn the free surface on for any subset of faces, elastic included — with fp16 boundary storage and a body-force <code>rho</code> gradient fix.</div>
  </a>

</div>
</section>

<section class="sweep-onefile" markdown>
<div class="sweep-onefile__inner" markdown>
<div class="sweep-onefile__eyebrow">ONE INSTALL · MANY PACKAGES</div>
<h2 class="sweep-onefile__title">One install.<br><span class="sweep-onefile__dim">Then <code>import sweep</code>.</span></h2>
<p class="sweep-onefile__lede"><code>sweepx</code> is the umbrella — it carries no code of its own, it just pulls in the engine and its published companions. Install <code>sweepx</code>, but <strong>import <code>sweep</code></strong> (same pattern as <code>scikit-learn</code> → <code>sklearn</code>).</p>

```python
# one install — pulls the engine + the agent:
#     pip install sweepx

import sweep                                       # ← sweep-solver, the engine
from sweep.propagator.torch import PropTorch       # torch / JAX / native-CUDA
from sweep.equations import Acoustic, ElasticTTI   # nine equation families

# …or drive it all in plain language — sweep-agent:
#     sweep-agent chat
#     >>> load Marmousi and run a forward — show the shot gather
```

<a class="sweep-cta sweep-cta--ghost-dark" href="getting-started/">Install &amp; get started →</a>
</div>
</section>

<section class="sweep-stack">
<div class="sweep-stack__eyebrow">THE FAMILY</div>
<h2 class="sweep-stack__title">Every piece, installable on its own.</h2>
<p class="sweep-stack__lede">The same layout as the PyLops family — <code>sweepx</code> bundles the published packages; you can also <code>pip install</code> any single one.</p>
<div class="sweep-stack__grid sweep-stack__grid--3">
  <div class="sweep-stack__card">
    <div class="sweep-stack__head"><span class="sweep-stack__dot" style="background:#1AA690"></span><span class="sweep-stack__role">PUBLISHED</span></div>
    <div class="sweep-stack__name">sweep-solver</div>
    <div class="sweep-stack__version">import sweep</div>
    <div class="sweep-stack__desc">Wave-equation engine — equations, propagators, FWI/LSRTM/RTM.</div>
  </div>
  <div class="sweep-stack__card">
    <div class="sweep-stack__head"><span class="sweep-stack__dot" style="background:#1AA690"></span><span class="sweep-stack__role">PUBLISHED</span></div>
    <div class="sweep-stack__name">sweep-agent</div>
    <div class="sweep-stack__version">import sweep_agent</div>
    <div class="sweep-stack__desc">Natural-language control via a local LLM.</div>
  </div>
  <div class="sweep-stack__card">
    <div class="sweep-stack__head"><span class="sweep-stack__dot sweep-stack__dot--sq" style="background:#ED8B2E"></span><span class="sweep-stack__role">COMPANION</span></div>
    <div class="sweep-stack__name">sweep-tasks</div>
    <div class="sweep-stack__version">coming</div>
    <div class="sweep-stack__desc">Production FWI/LSRTM runner — specs, YAML, multi-GPU, IO.</div>
  </div>
  <div class="sweep-stack__card">
    <div class="sweep-stack__head"><span class="sweep-stack__dot sweep-stack__dot--sq" style="background:#ED8B2E"></span><span class="sweep-stack__role">COMPANION</span></div>
    <div class="sweep-stack__name">sweep-loss</div>
    <div class="sweep-stack__version">coming</div>
    <div class="sweep-stack__desc">Misfit / loss functions for inversion.</div>
  </div>
  <div class="sweep-stack__card">
    <div class="sweep-stack__head"><span class="sweep-stack__dot sweep-stack__dot--sq" style="background:#ED8B2E"></span><span class="sweep-stack__role">COMPANION</span></div>
    <div class="sweep-stack__name">sweep-nn</div>
    <div class="sweep-stack__version">coming</div>
    <div class="sweep-stack__desc">Neural reparameterizations — INR / hash / SIREN encoders.</div>
  </div>
  <div class="sweep-stack__card">
    <div class="sweep-stack__head"><span class="sweep-stack__dot sweep-stack__dot--sq" style="background:#ED8B2E"></span><span class="sweep-stack__role">COMPANION</span></div>
    <div class="sweep-stack__name">sweep-tomo</div>
    <div class="sweep-stack__version">coming</div>
    <div class="sweep-stack__desc">First-arrival traveltime tomography — eikonal + SIRT / FATT.</div>
  </div>
</div>
</section>
