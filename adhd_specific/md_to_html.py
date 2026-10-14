"""RESULTS.md -> self-contained RESULTS.html (figures inlined as base64, math via MathJax).

    ~/miniforge/bin/python md_to_html.py [RESULTS.md] [RESULTS.html]

Adapted from gus_analysis/md_to_html.py. Additions: $...$ / $$...$$ are shielded from
Markdown and typeset by MathJax (loaded from jsdelivr), and every <img src="results/*.png">
is embedded so the HTML can be opened or sent as a single file.
"""
import base64
import os
import re
import sys

import markdown

sys.path.insert(0, "<PROJECT_ROOT>/gus_analysis")
from md_to_html import CSS as _GUS_CSS  # noqa: E402

CSS = _GUS_CSS + """
body { max-width: 980px; }
figure { margin: 1.6rem 0; }
figure img, p > img { max-width: 100%; height: auto; display: block; margin: 0 auto;
                      border: 1px solid var(--line); border-radius: 6px; }
figcaption, .cap { font-size: 13.5px; color: var(--muted); margin-top: .5rem; line-height: 1.5; }
.MathJax { font-size: 1.02em !important; }
mjx-container[display="true"] { overflow-x: auto; overflow-y: hidden; }
details { margin: .8rem 0; } summary { cursor: pointer; color: var(--accent); }
"""

MATHJAX = """<script>window.MathJax = { tex: { inlineMath: [['\\\\(', '\\\\)']],
displayMath: [['\\\\[', '\\\\]']] }, options: { skipHtmlTags: ['script','style','textarea','pre','code'] } };</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>"""


def shield_math(text):
    store = []

    def keep(s):
        store.append(s)
        return f"MATHPH{len(store) - 1}X"

    text = re.sub(r"\$\$(.+?)\$\$", lambda m: keep("\\[" + m.group(1) + "\\]"), text, flags=re.S)
    text = re.sub(r"(?<![\\$])\$(?!\s)([^$\n]+?)(?<!\s)\$", lambda m: keep("\\(" + m.group(1) + "\\)"), text)
    return text, store


def unshield(html, store):
    import html as h
    return re.sub(r"MATHPH(\d+)X", lambda m: h.escape(store[int(m.group(1))], quote=False), html)


def inline_images(html, base):
    def rep(m):
        p = os.path.join(base, m.group(1))
        if not os.path.exists(p):
            return m.group(0)
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        return f'src="data:image/png;base64,{b64}"'
    return re.sub(r'src="([^"]+\.png)"', rep, html)


def convert(src, dst):
    text = open(src, encoding="utf-8").read()
    shielded, store = shield_math(text)
    body = markdown.markdown(shielded, extensions=["tables", "fenced_code", "toc", "sane_lists",
                                                   "attr_list", "md_in_html"], output_format="html5")
    body = inline_images(unshield(body, store), os.path.dirname(os.path.abspath(src)))
    title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("# ")),
                 os.path.basename(src))
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
{MATHJAX}</head>
<body>
{body}
<div class="footer">Generated from <code>{os.path.basename(src)}</code> · BBC ASD revision ·
ADHD analyses. Figures embedded; aggregate results only.</div>
</body></html>
"""
    open(dst, "w", encoding="utf-8").write(html)
    print(f"wrote {dst} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "RESULTS.md")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".html"
    convert(src, dst)
