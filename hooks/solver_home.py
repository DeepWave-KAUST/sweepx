"""Replace an imported package's marketing homepage with a plain overview.

sweep-solver's `docs/index.md` is the landing page of its OWN standalone site:
it declares `template: home.html` and renders the SWEEP hero, feature cards, and
notebook gallery. Inside the sweepx umbrella that duplicates the sweepx Home tab,
and its relative `figures/…` sources don't resolve at the imported depth — so the
Solver tab opened as a broken second homepage.

The umbrella owns the landing page, so that one file becomes a short overview and
its nav entry is retitled from "Home" (which collides with the umbrella's own Home
tab) to "Overview". Keeping this here instead of editing the sweep repo means
sweep-solver's site is unchanged when built standalone.
"""

from __future__ import annotations

# src_uri -> replacement markdown, at the path multirepo produces once the
# package's docs/* is hoisted to its section root.
_OVERVIEWS: dict[str, str] = {
    "solver/index.md": """# sweep-solver

The differentiable, GPU **wave-equation engine** at the core of the sweep
ecosystem — forward modelling, RTM, and FWI / LSRTM, with equations for
acoustic, elastic, VTI/TTI, VRZ and spectral-element (SEM) physics, and
torch / JAX / native-CUDA backends.

Installed with `pip install sweepx` (or `pip install sweep-solver`) → `import sweep`.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: __[Getting started](getting-started/installation.md)__

    ---

    Install, run your first forward model, take your first gradient.

-   :material-book-open-variant-outline: __[User guide](user-guide/equations.md)__

    ---

    Equations, propagators, backends, boundary-saving, memory options.

-   :material-notebook-outline: __[Examples](examples/index.md)__

    ---

    Runnable notebooks — modelling, wavefields, FWI, RTM.

-   :material-api: __[API reference](api/index.md)__

    ---

    Every equation, propagator, and operator, documented from source.

</div>
""",
}


def on_nav(nav, config, files):
    """Retitle a replaced landing page.

    A package's own nav calls its landing page "Home" — right for its standalone
    site, wrong under the umbrella, where Home is the sweepx tab. Runs before
    pages are read, and `Page.read_source` only derives a title when none is
    set, so this sticks.
    """
    for page in nav.pages:
        if page.file.src_uri in _OVERVIEWS:
            page.title = "Overview"
    return nav


def on_page_markdown(markdown: str, page, config, files) -> str:
    """Swap the body of an imported package's landing page."""
    replacement = _OVERVIEWS.get(page.file.src_uri)
    if replacement is None:
        return markdown
    # Drop the hero template + sidebar hiding so it renders as a normal page,
    # and let the H1 above supply the title.
    page.meta.pop("template", None)
    page.meta.pop("hide", None)
    return replacement
