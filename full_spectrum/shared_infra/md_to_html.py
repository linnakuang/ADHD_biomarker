"""Convert a Markdown report to a styled, self-contained HTML file.

Usage: python md_to_html.py INPUT.md [OUTPUT.html]
Uses python-markdown with tables/toc/fenced-code/sane-lists. Figure/table paths in the
source stay relative (results/...), so keep the HTML next to the .md so links resolve.
"""
import os
import sys
import markdown

CSS = """
:root { --fg:#1a1a1a; --muted:#555; --line:#e2e2e2; --accent:#2b6cb0; --bg:#fff;
        --code-bg:#f5f6f8; --th-bg:#f0f4f8; }
* { box-sizing: border-box; }
body { max-width: 900px; margin: 0 auto; padding: 2.5rem 1.5rem 5rem;
       font: 16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
       color: var(--fg); background: var(--bg); }
h1 { font-size: 1.9rem; border-bottom: 3px solid var(--accent); padding-bottom: .4rem;
     margin-top: 0; }
h2 { font-size: 1.4rem; margin-top: 2.4rem; border-bottom: 1px solid var(--line);
     padding-bottom: .3rem; }
h3 { font-size: 1.15rem; margin-top: 1.8rem; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code { background: var(--code-bg); padding: .12em .35em; border-radius: 4px;
       font: 13.5px/1.5 "SF Mono",Menlo,Consolas,monospace; }
pre { background: var(--code-bg); padding: 1rem 1.1rem; border-radius: 8px; overflow-x: auto;
      border: 1px solid var(--line); }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 1.1rem 0; font-size: 14.5px;
        display: block; overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: .5rem .7rem; text-align: left;
         vertical-align: top; }
th { background: var(--th-bg); font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
blockquote { border-left: 4px solid var(--accent); margin: 1rem 0; padding: .3rem 1rem;
             color: var(--muted); background: #f8fafc; }
hr { border: none; border-top: 1px solid var(--line); margin: 2.2rem 0; }
strong { font-weight: 650; }
.footer { margin-top: 4rem; padding-top: 1rem; border-top: 1px solid var(--line);
          color: var(--muted); font-size: 13px; }
"""

def convert(src, dst):
    text = open(src, encoding="utf-8").read()
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
        output_format="html5",
    )
    title = next((l.lstrip("# ").strip() for l in text.splitlines()
                  if l.startswith("# ")), os.path.basename(src))
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style></head>
<body>
{body}
<div class="footer">Generated from <code>{os.path.basename(src)}</code> ·
BBC ASD revision · spectrum analyses.</div>
</body></html>
"""
    open(dst, "w", encoding="utf-8").write(html)
    print(f"wrote {dst} ({len(html)} bytes)")

if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".html"
    convert(src, dst)
