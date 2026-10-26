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
    <div class="sweep-features-3__tag">PACKAGE · import sweepx.agent</div>
    <h3 class="sweep-features-3__title">sweepx.agent</h3>
    <p class="sweep-features-3__desc">
      The natural-language <strong>control layer</strong>. Say what you want and a
      <em>local</em> LLM (Ollama / vLLM) turns it into a validated <code>sweep</code>
      run — forward modelling, benchmark models, plots.
    </p>
    <span class="sweep-features-3__link">Open the agent docs <span aria-hidden="true">→</span></span>
  </a>
  <a class="sweep-features-3__card" href="sweepx/">
    <div class="sweep-features-3__tag">TOOLKIT · import sweepx</div>
    <h3 class="sweep-features-3__title">…and the toolkit</h3>
    <p class="sweep-features-3__desc">
      <code>sweepx.tasks</code> (production FWI/LSRTM runner),
      <code>sweepx.loss</code>, <code>sweepx.io</code>, <code>sweepx.nn</code> and
      <code>sweepx.opt</code>, all in the one <code>sweepx</code> package; tomography follows.
    </p>
    <span class="sweep-features-3__link">Open the toolkit docs <span aria-hidden="true">→</span></span>
  </a>
</section>

<!-- What's new: six cards, rotate on release. See docs-template/README.md. -->
<section class="sweep-new">
<div class="sweep-new__eyebrow">WHAT'S NEW</div>
<h2 class="sweep-new__title">Recently shipped.</h2>
<p class="sweep-new__lede">The latest user-visible additions across the stack. The badge says where each one lives: a released version you get from <code>pip install</code>, or <span class="sweep-new__badge sweep-new__badge--dev">dev</span> for what is merged but not yet on PyPI — these docs are built from the development branch.</p>
<div class="sweep-new__grid">
  <a class="sweep-new__card" href="solver/notebooks/40_wavefield_poroelastic/">
    <div class="sweep-new__hd"><span class="sweep-new__badge">v0.3.6</span><span class="sweep-new__date">2026-10</span></div>
    <div class="sweep-new__name">Poroelastic waves</div>
    <div class="sweep-new__desc"><code>Biot</code>: 2-D low-frequency Biot waves, fast P, slow P and S in one wavefield, with gradients. The viscous drag is integrated exactly, so any permeability runs at the fast-P time step, and <code>biot_from_rock</code> builds the coefficients from rock properties.</div>
  </a>

  <a class="sweep-new__card" href="solver/notebooks/15_wavefield_topography/">
    <div class="sweep-new__hd"><span class="sweep-new__badge">v0.3.6</span><span class="sweep-new__date">2026-10</span></div>
    <div class="sweep-new__name">Elastic topography, rebuilt</div>
    <div class="sweep-new__desc"><code>ElasticCurvilinear</code> now follows Bartana, Xu &amp; Rickett: collocated fields and a self-adjoint operator that stays stable on every hill tried, where the old scheme blew up. Sources, receivers, snapshots and models all live on the physical grid. It replaces the APM free surface, now removed.</div>
  </a>

  <a class="sweep-new__card" href="solver/notebooks/34_lsrtm_qp_vti_tti/">
    <div class="sweep-new__hd"><span class="sweep-new__badge">v0.3.6</span><span class="sweep-new__date">2026-10</span></div>
    <div class="sweep-new__name">Anisotropic qP on the CUDA core</div>
    <div class="sweep-new__desc">The pure-qP <code>AcousticVTI</code> and <code>AcousticTTI</code> of Liang et al. (2023) run on <code>impl='c'</code> with full-storage and boundary-saving gradients, and gain Born / LSRTM forms, <code>AcousticVTILSRTM</code> and <code>AcousticTTILSRTM</code>, on both backends.</div>
  </a>

  <a class="sweep-new__card" href="tasks/datasets/viking/">
    <div class="sweep-new__hd"><span class="sweep-new__badge">sweepx 0.4.1</span><span class="sweep-new__date">2026-10</span></div>
    <div class="sweep-new__name">Viking, wavelet-free</div>
    <div class="sweep-new__desc">The Viking Graben walkthrough now inverts all 1001 shots at once with frequency-selection FWI: download, index, plan, six coefficient shards and one <code>sweep-tasks run</code>. No source wavelet, and under an hour on one V100.</div>
  </a>

  <a class="sweep-new__card" href="getting-started/">
    <div class="sweep-new__hd"><span class="sweep-new__badge">sweepx 0.4.0</span><span class="sweep-new__date">2026-10</span></div>
    <div class="sweep-new__name">One package for the toolkit</div>
    <div class="sweep-new__desc">The task runner, the agent and the io, loss, nn and opt libraries now ship as one package, <code>sweepx</code>, from one repository. Imports change from <code>sweep_tasks</code> to <code>sweepx.tasks</code>, and likewise for the rest; the <code>sweep-tasks</code> and <code>sweep-agent</code> commands stay as they were.</div>
  </a>

  <a class="sweep-new__card" href="opt/">
    <div class="sweep-new__hd"><span class="sweep-new__badge">sweepx 0.4.0</span><span class="sweep-new__date">2026-10</span></div>
    <div class="sweep-new__name">Optimizers for FWI</div>
    <div class="sweep-new__desc"><code>sweepx.opt</code> joins the toolkit: L-BFGS, truncated Newton and Gauss-Newton, trust region, Adam, Shampoo and SOAP behind the <code>torch.optim</code> interface, with pseudo-Hessian and depth preconditioners and bound parameterizations.</div>
  </a>

</div>
</section>

<section class="sweep-onefile" markdown>
<div class="sweep-onefile__inner" markdown>
<div class="sweep-onefile__eyebrow">ONE INSTALL · TWO PACKAGES</div>
<h2 class="sweep-onefile__title">One install.<br><span class="sweep-onefile__dim">Then <code>import sweep</code>.</span></h2>
<p class="sweep-onefile__lede"><code>sweepx</code> is the toolkit — the task runner, the agent and the libraries they use — and it pulls in the engine, <code>sweep-solver</code>, as a dependency. Install <code>sweepx</code>; <strong>import the engine as <code>sweep</code></strong> (same pattern as <code>scikit-learn</code> → <code>sklearn</code>) and the toolkit as <code>sweepx.&lt;module&gt;</code>.</p>

```python
# one install — the engine and the whole toolkit:
#     pip install sweepx

import sweep                                       # ← sweep-solver, the engine
from sweep.propagator.torch import PropTorch       # torch / JAX / native-CUDA
from sweep.equations import Acoustic, ElasticTTI   # nine equation families

from sweepx.loss import EnvelopeLoss               # ← sweepx, the toolkit
from sweepx.tasks import load_task, TaskRunner     # YAML task runner

# …or drive it all in plain language — sweepx.agent:
#     sweep-agent chat
#     >>> load Marmousi and run a forward — show the shot gather
```

<a class="sweep-cta sweep-cta--ghost-dark" href="getting-started/">Install &amp; get started →</a>
</div>
</section>

<section class="sweep-stack">
<div class="sweep-stack__eyebrow">THE FAMILY</div>
<h2 class="sweep-stack__title">The engine, and the toolkit on top.</h2>
<p class="sweep-stack__lede">The engine is its own package, <code>sweep-solver</code>, so it installs on its own. Everything built on it ships together in <code>sweepx</code>, one module per job, layered <code>agent</code> → <code>tasks</code> → <code>io</code> / <code>loss</code> / <code>nn</code> / <code>opt</code>: a module imports only from the layers below it.</p>
<div class="sweep-stack__grid sweep-stack__grid--3">
  <div class="sweep-stack__card">
    <div class="sweep-stack__head"><span class="sweep-stack__dot" style="background:#1AA690"></span><span class="sweep-stack__role">PUBLISHED</span></div>
    <div class="sweep-stack__name">sweep-solver</div>
    <div class="sweep-stack__version">import sweep · <a href="solver/">docs</a></div>
    <div class="sweep-stack__desc">Wave-equation engine — equations, propagators, FWI/LSRTM/RTM.</div>
  </div>
  <div class="sweep-stack__card">
    <div class="sweep-stack__head"><span class="sweep-stack__dot" style="background:#1AA690"></span><span class="sweep-stack__role">IN SWEEPX</span></div>
    <div class="sweep-stack__name">sweepx.agent</div>
    <div class="sweep-stack__version">in sweepx · <a href="agent/">docs</a></div>
    <div class="sweep-stack__desc">Natural-language control via a local LLM.</div>
  </div>
  <div class="sweep-stack__card">
    <div class="sweep-stack__head"><span class="sweep-stack__dot" style="background:#1AA690"></span><span class="sweep-stack__role">IN SWEEPX</span></div>
    <div class="sweep-stack__name">sweepx.tasks</div>
    <div class="sweep-stack__version">in sweepx · <a href="tasks/">docs</a></div>
    <div class="sweep-stack__desc">Production FWI/LSRTM runner — specs, YAML, multi-GPU, IO.</div>
  </div>
  <div class="sweep-stack__card">
    <div class="sweep-stack__head"><span class="sweep-stack__dot" style="background:#1AA690"></span><span class="sweep-stack__role">IN SWEEPX</span></div>
    <div class="sweep-stack__name">sweepx.loss</div>
    <div class="sweep-stack__version">in sweepx · <a href="loss/">docs</a></div>
    <div class="sweep-stack__desc">Misfit / loss functions for inversion.</div>
  </div>
  <div class="sweep-stack__card">
    <div class="sweep-stack__head"><span class="sweep-stack__dot" style="background:#1AA690"></span><span class="sweep-stack__role">IN SWEEPX</span></div>
    <div class="sweep-stack__name">sweepx.nn</div>
    <div class="sweep-stack__version">in sweepx · <a href="nn/">docs</a></div>
    <div class="sweep-stack__desc">Neural reparameterizations and priors — INR / hash / SIREN, TV, diffusion.</div>
  </div>
  <div class="sweep-stack__card">
    <div class="sweep-stack__head"><span class="sweep-stack__dot" style="background:#1AA690"></span><span class="sweep-stack__role">IN SWEEPX</span></div>
    <div class="sweep-stack__name">sweepx.io</div>
    <div class="sweep-stack__version">in sweepx · <a href="io/">docs</a></div>
    <div class="sweep-stack__desc">Seismic I/O — SEG-Y index, data plans, prefetched reads, geometry.</div>
  </div>
  <div class="sweep-stack__card">
    <div class="sweep-stack__head"><span class="sweep-stack__dot" style="background:#1AA690"></span><span class="sweep-stack__role">IN SWEEPX</span></div>
    <div class="sweep-stack__name">sweepx.opt</div>
    <div class="sweep-stack__version">in sweepx · <a href="opt/">docs</a></div>
    <div class="sweep-stack__desc">FWI optimizers — L-BFGS, truncated Newton / Gauss-Newton, Adam, Shampoo / SOAP, preconditioners, bounds.</div>
  </div>
  <div class="sweep-stack__card">
    <div class="sweep-stack__head"><span class="sweep-stack__dot sweep-stack__dot--sq" style="background:#ED8B2E"></span><span class="sweep-stack__role">COMING</span></div>
    <div class="sweep-stack__name">sweepx.tomo</div>
    <div class="sweep-stack__version">in sweepx, later</div>
    <div class="sweep-stack__desc">First-arrival traveltime tomography — eikonal + SIRT / FATT.</div>
  </div>
</div>
</section>
