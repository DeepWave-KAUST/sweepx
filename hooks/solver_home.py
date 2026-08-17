"""Replace an imported package's marketing homepage with a plain overview.

sweep-solver's own `docs/index.md` is its standalone landing page: it declares
`template: home.html` and renders the SWEEP hero, feature cards, and notebook
gallery. Inside the sweepx umbrella that duplicates the sweepx Home tab, and its
relative `figures/...` sources don't resolve at the imported depth — so the
Solver tab opened as a broken second homepage.

The umbrella owns the landing page, so here we swap that one file for a short
overview + grid cards. Everything else the package ships is untouched.

Keeping this in the umbrella (instead of editing the sweep repo) means
sweep-solver's own site keeps its hero when built standalone.
"""

from __future__ import annotations

# Section index pages to replace: src_uri -> (title, markdown body).
# `solver/docs/index.md` is the path multirepo produces for the Solver import
# (keep_docs_dir=true preserves the repo's docs/ level).
_OVERVIEWS: dict[str, str] = {
    "solver/docs/index.md": """# sweep-solver

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


def on_page_markdown(markdown: str, page, config, files) -> str:
    """Swap the body of an imported package's landing page."""
    replacement = _OVERVIEWS.get(page.file.src_uri)
    if replacement is None:
        return markdown
    # Drop the custom hero template so the page renders as normal content, and
    # let the H1 above supply the page title.
    page.meta.pop("template", None)
    page.meta.pop("hide", None)
    return replacement
