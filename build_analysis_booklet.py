#!/usr/bin/env python3
"""
build_analysis_booklet.py

Scans the current directory recursively for Jupyter notebooks (*.ipynb),
extracts embedded figures, preceding markdown, code cells, and helper
function definitions from all Python files.

Output:
analysis_booklet/
    analysis_report.md
    figures/
"""


import ast, base64, re
from pathlib import Path
import nbformat

ROOT=Path(".")
OUT=ROOT/"analysis_booklet"
FIGS=OUT/"figures"
SKIP = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".ipynb_checkpoints",
    "analysis_booklet",
    ".mypy_cache",
    ".pytest_cache",
    ".virtual_documents"
}
OUT.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)

def calls(src):
    out=[]
    try:
        t=ast.parse(src)
        for n in ast.walk(t):
            if isinstance(n,ast.Call):
                if isinstance(n.func,ast.Attribute):
                    out.append(ast.unparse(n))
                elif isinstance(n.func,ast.Name):
                    out.append(ast.unparse(n))
    except: pass
    # unique preserve order
    seen=[];res=[]
    for x in out:
        if x not in seen:
            seen.append(x);res.append(x)
    return res

report=[]
fig=1
nbs=0
for nbpath in ROOT.rglob("*.ipynb"):
    if any(p in SKIP for p in nbpath.parts): continue
    nbs+=1
    nb=nbformat.read(nbpath,as_version=4)
    prev_md=""
    for i,cell in enumerate(nb.cells):
        if cell.cell_type=="markdown":
            prev_md=cell.source.strip(); continue
        if cell.cell_type!="code": continue

        title=""
        for line in cell.source.splitlines():
            m=re.match(r"\s*#\s*(Fig(?:ure)?\.?\s*\d*[:\-]?\s*.*)",line,re.I)
            if m:
                title=m.group(1).strip(); break

        fns=calls(cell.source)
        for out in getattr(cell,"outputs",[]):
            data=out.get("data",{})
            if "image/png" not in data: continue
            img=f"Figure_{fig:03d}.png"
            (FIGS/img).write_bytes(base64.b64decode(data["image/png"]))
            report.append(f"# Figure {fig}\n")
            report.append(f"**Notebook:** `{nbpath}`")
            if title: report.append(f"\n**Title:** {title}")
            if prev_md:
                report.append("\n\n## Markdown Context\n")
                report.append(prev_md)
            if fns:
                report.append("\n\n## Function Calls\n")
                for c in fns:
                    report.append(f"- `{c}`")
            report.append(f"\n\n## Image\n\n![](figures/{img})\n\n---\n")
            fig+=1

summary=f"""# Analysis Summary

- Notebooks scanned: {nbs}
- Figures extracted: {fig-1}

This report intentionally omits implementation details and helper functions.
"""
(OUT/"analysis_report.md").write_text(summary+"\n".join(report),encoding="utf8")
print("Done.")