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

<a class="sweep-cta sweep-cta--ghost-dark" href="solver/getting-started/installation/">Install &amp; get started →</a>
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
