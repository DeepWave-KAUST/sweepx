"""Convert REPORT.md (with embedded ./figs/*.png) → REPORT.pdf.

Approach:
1. markdown → HTML with the `tables`/`fenced_code`/`attr_list` extensions
2. wrap in a print-friendly CSS template (A4 portrait, table borders, image
   widths capped to the page, page-break-inside avoid for figures)
3. headless Chrome's --print-to-pdf renders it identically to "Print to PDF"
   in a real browser
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import markdown


HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "REPORT.md")
HTML = os.path.join(HERE, "REPORT.html")
PDF = os.path.join(HERE, "REPORT.pdf")


CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
html, body { font-family: "DejaVu Sans", "Noto Sans CJK SC", "Source Han Sans SC",
                          "WenQuanYi Zen Hei", "Microsoft YaHei", sans-serif;
             font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 22pt; margin-top: 0; }
h2 { font-size: 15pt; border-bottom: 1px solid #ccc; padding-bottom: 4px;
     page-break-after: avoid; margin-top: 22pt; }
h3 { font-size: 12.5pt; page-break-after: avoid; margin-top: 16pt; }
h4 { font-size: 11pt; }
p, ul, ol { margin: 6pt 0; }
code, pre { font-family: "DejaVu Sans Mono", "Liberation Mono", monospace;
            font-size: 9.5pt; background: #f4f4f4; padding: 1px 3px; border-radius: 2px; }
pre { padding: 8px; overflow-x: auto; page-break-inside: avoid; }
blockquote { margin: 6pt 0; padding: 4pt 10pt; border-left: 3px solid #888;
             color: #444; background: #fafafa; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 9.5pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 3pt 6pt; vertical-align: top; }
th { background: #efefef; font-weight: 600; }
img { max-width: 100%; height: auto; display: block; margin: 8pt auto;
      page-break-inside: avoid; }
a { color: #1750a4; text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border: none; border-top: 1px solid #ddd; margin: 14pt 0; }
"""


HTML_TEMPLATE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>sweep_loss formula-level validation report</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def convert():
    if not os.path.exists(MD):
        sys.exit(f"missing {MD}")

    with open(MD, "r", encoding="utf-8") as fh:
        text = fh.read()

    body = markdown.markdown(
        text,
        # NB. no `sane_lists` — REPORT.md doesn't put a blank line before each
        # bullet list, and the default markdown lexer is lenient about that.
        extensions=["tables", "fenced_code", "attr_list", "toc", "nl2br"],
        output_format="html5",
    )

    with open(HTML, "w", encoding="utf-8") as fh:
        fh.write(HTML_TEMPLATE.format(css=CSS, body=body))
    print(f"wrote {HTML}")

    chrome = shutil.which("google-chrome") or shutil.which("chromium") \
        or shutil.which("chromium-browser")
    if chrome is None:
        sys.exit("no chrome / chromium in PATH; install one and retry")

    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF}",
        f"file://{HTML}",
    ]
    print(f"calling: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"chrome failed (rc={res.returncode}):\n{res.stderr}")
    print(f"wrote {PDF} ({os.path.getsize(PDF) / 1024:.0f} KB)")


if __name__ == "__main__":
    convert()
