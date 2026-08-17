"""Umbrella-side fixups for docs imported from the package repos.

Two things, both because an imported package's `docs/` was written to be the
root of its OWN standalone site:

1. `on_config` — **repair dangling symlinks.** sweep-solver keeps its notebooks
   in `examples/notebooks/` and exposes them to mkdocs as `docs/notebooks`, a
   relative symlink. mkdocs-multirepo hoists `docs/*` to the section root, so
   that link ends up pointing one level *outside* the section and mkdocs dies
   with FileNotFoundError. We import `examples/notebooks/*` too (see the
   `!import` line in mkdocs.yml) and re-point the link at the copy that landed
   inside the section — which keeps the URLs clean (`/solver/notebooks/…`
   instead of the `/solver/docs/…` that `keep_docs_dir` would force).

2. `on_page_markdown` — **replace a package's marketing homepage.** sweep-solver's
   `docs/index.md` declares `template: home.html` and renders the SWEEP hero,
   feature cards, and notebook gallery. Inside the umbrella that duplicates the
   sweepx Home tab, and its relative `figures/…` sources don't resolve at the
   imported depth. The umbrella owns the landing page, so that one file becomes
   a short overview instead.

Keeping both here (instead of editing the sweep repo) means sweep-solver's own
site is unchanged when built standalone.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.sweepx")


# ── 1. dangling symlinks in imported docs ────────────────────────────────────
def on_config(config):
    """Satisfy dangling symlinks in the freshly imported tree.

    Ordering makes this work: mkdocs-multirepo clones into its `temp_dir` during
    *its* `on_config`, mkdocs collects files later (multirepo's `on_files`), and
    hooks run after plugins — so this sits exactly in between, when the tree is
    on disk but not yet read.

    A hoisted `docs/notebooks -> ../examples/notebooks` now points just outside
    its section. Rather than rewrite the link, we create what it asks for: a
    link at the target path pointing to the copy `extra_imports` placed inside
    the section. mkdocs then walks it as an ordinary directory.
    """
    temp_dir = _multirepo_temp_dir(config)
    if temp_dir is None or not temp_dir.is_dir():
        return config

    for dirpath, dirnames, filenames in os.walk(temp_dir, followlinks=False):
        for name in list(dirnames) + list(filenames):
            link = Path(dirpath) / name
            if not link.is_symlink() or link.exists():
                continue  # not a symlink, or it already resolves

            target = (link.parent / os.readlink(link)).resolve()
            section = _section_root(link, temp_dir)
            real = next(
                (
                    p
                    for p in section.rglob(name)
                    if p.is_dir() and not p.is_symlink() and p.resolve() != link
                ),
                None,
            )
            if real is None:
                log.warning(
                    "sweepx: %s dangles and nothing matching was imported — add it to "
                    "the !import's extra_imports",
                    link.relative_to(temp_dir),
                )
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(real)
            log.info(
                "sweepx: satisfied %s via %s",
                link.relative_to(temp_dir),
                real.relative_to(temp_dir),
            )
    return config


def _multirepo_temp_dir(config) -> Path | None:
    """Where mkdocs-multirepo cloned the imported repos."""
    plugin = config.get("plugins", {}).get("multirepo")
    if plugin is None:
        return None
    # The plugin stores it on itself; fall back to its documented default.
    temp_dir = getattr(plugin, "temp_dir", None)
    if temp_dir:
        return Path(temp_dir)
    name = plugin.config.get("temp_dir", "temp_dir")
    return Path(config["docs_dir"]).parent / name


def _section_root(path: Path, root: Path) -> Path:
    """The top-level imported section (e.g. `<temp>/solver`) containing `path`."""
    rel = path.relative_to(root)
    return root / rel.parts[0] if len(rel.parts) > 1 else root


# ── 2. an imported package's landing page ────────────────────────────────────
# src_uri -> replacement markdown. `solver/index.md` is what multirepo produces
# for the Solver import once docs/* is hoisted to the section root.
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
