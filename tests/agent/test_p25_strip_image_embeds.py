"""P25: strip local-path markdown image embeds from the final reply.

The figures are shown inline (from the tool result). A model that ALSO writes
`![](/tmp/x.png)` into its reply would render a broken image icon in gradio,
because gradio only serves files through its /file= proxy — a raw local path
404s. We strip those embeds (keeping the surrounding prose)."""

from __future__ import annotations

from sweep_agent.webui import _strip_local_image_embeds


def test_strips_local_image_and_tidies_bullet():
    txt = (
        "The synthetic FWI run was successful. Here are the results:\n\n"
        "- **Inverted vs True Model Comparison**: ![](/tmp/fwi_run/output/model_comparison.png)\n"
        "- **Convergence Curve**: ![](/tmp/fwi_run/output/convergence.png)\n\n"
        "The inverted model is saved at `/tmp/fwi_run/output/inverted_vp.npy`."
    )
    out = _strip_local_image_embeds(txt)
    assert "![" not in out                       # no image embeds remain
    assert ".png)" not in out
    assert "run was successful" in out           # prose kept
    assert "inverted_vp.npy" in out              # path reference kept
    assert "\n\n\n" not in out                   # no triple blank lines


def test_keeps_remote_images():
    txt = "See ![chart](https://example.com/a.png) online."
    assert _strip_local_image_embeds(txt) == txt  # http(s) images untouched


def test_plain_text_unchanged():
    txt = "FWI converged; loss dropped from 0.13 to 0.005 over 40 epochs."
    assert _strip_local_image_embeds(txt) == txt


def test_handles_relative_and_spaced_paths():
    assert _strip_local_image_embeds("a ![x]( output/g.gif ) b").strip() == "a  b".strip() or \
        "g.gif" not in _strip_local_image_embeds("a ![x]( output/g.gif ) b")


def test_empty():
    assert _strip_local_image_embeds("") == ""
    assert _strip_local_image_embeds(None) is None
