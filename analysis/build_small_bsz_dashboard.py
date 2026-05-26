#!/usr/bin/env python3
from __future__ import annotations
import csv, html, json, math
from collections import defaultdict
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "figures" / "small_bsz_dashboard"
OUT.mkdir(parents=True, exist_ok=True)
MAIN_CSV = ROOT / "figures" / "current_same_driver_v26" / "raw_rows.csv"
CATALOG_CSV = ROOT / "results" / "sweep_catalog" / "all_sweeps.csv"

COLORS = {
    "streaming_identity": "#1f77b4", "identity": "#1f77b4",
    "streaming_lite": "#2ca02c", "top_aware_muon": "#d62728",
    "native_muon": "#9467bd", "muon": "#9467bd", "soap": "#ff7f0e",
    "shampoo": "#8c564b", "adamw": "#7f7f7f", "lite_chi2_rs01": "#2ca02c", "top1pm_a05": "#d62728",
}
MARKERS = {"streaming_identity":"circle", "identity":"circle", "streaming_lite":"square", "top_aware_muon":"tri", "native_muon":"diamond", "muon":"diamond", "soap":"square"}

def read_csv(path):
    if not path.exists(): return []
    with path.open(newline="") as f: return list(csv.DictReader(f))

def flt(x, default=math.nan):
    try:
        if x is None or x == "": return default
        return float(x)
    except Exception: return default

def integer(x, default=0):
    try:
        if x is None or x == "": return default
        return int(float(x))
    except Exception: return default

def fmt_batch(n):
    n=int(n)
    if n>=1048576:
        v=n/1048576; return f"{int(v)}M" if abs(v-round(v))<1e-9 else f"{v:g}M"
    if n>=1024:
        v=n/1024; return f"{int(v)}K" if abs(v-round(v))<1e-9 else f"{v:g}K"
    return str(n)

def norm_main(r):
    mp={"identity":"streaming_identity","lite_chi2_rs01":"streaming_lite","top1pm_a05":"top_aware_muon"}
    m=mp.get(r.get("method",""), r.get("method",""))
    return {**r, "method_norm":m, "variant":r.get("method_label",""), "score_f":flt(r.get("score")), "lr_f":flt(r.get("lr")), "batch_i":integer(r.get("batch")), "batch_label":r.get("batch_label") or fmt_batch(integer(r.get("batch"))), "source":"v26_streaming_small", "family":"same_driver_streaming_ddp"}

def norm_cat(r):
    return {**r, "method_norm":r.get("method",""), "variant":r.get("variant",""), "score_f":flt(r.get("score")), "lr_f":flt(r.get("lr")), "batch_i":integer(r.get("batch")), "batch_label":r.get("batch_label") or fmt_batch(integer(r.get("batch")))}

def valid(rows): return [r for r in rows if math.isfinite(r["score_f"]) and math.isfinite(r["lr_f"]) and r["batch_i"]>0]
main_rows=valid([norm_main(r) for r in read_csv(MAIN_CSV)])
catalog_rows=valid([norm_cat(r) for r in read_csv(CATALOG_CSV)])
catalog_full=[r for r in catalog_rows if (not r.get("status") or r.get("status")=="complete") and r.get("quality","") in {"","full_or_historical"}]

def esc(s): return html.escape(str(s), quote=True)
def color(m): return COLORS.get(m, "#333333")
def log2(x): return math.log(x,2)
def nice_lr(x): return f"{x:g}"
def nice_score(x): return f"{x:.6f}" if abs(x)<10 else f"{x:.4g}"

def shape_svg(kind,x,y,c,sz=4.2,stroke=None,fill=None):
    stroke=stroke or c; fill=fill if fill is not None else c
    if kind=="square": return f'<rect x="{x-sz:.1f}" y="{y-sz:.1f}" width="{2*sz:.1f}" height="{2*sz:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
    if kind=="tri": return f'<polygon points="{x:.1f},{y-sz*1.2:.1f} {x-sz*1.2:.1f},{y+sz:.1f} {x+sz*1.2:.1f},{y+sz:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
    if kind=="diamond": return f'<polygon points="{x:.1f},{y-sz*1.25:.1f} {x-sz*1.25:.1f},{y:.1f} {x:.1f},{y+sz*1.25:.1f} {x+sz*1.25:.1f},{y:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{sz:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'

def best_by_batch_method(rows):
    best={}
    for r in rows:
        k=(r["batch_i"], r.get("batch_label") or fmt_batch(r["batch_i"]), r["method_norm"], r.get("variant",""), r.get("source",""), r.get("family",""))
        if k not in best or r["score_f"]<best[k]["score_f"]: best[k]=r
    return list(best.values())

def winners_by_batch(rows):
    by=defaultdict(list)
    for r in best_by_batch_method(rows): by[r["batch_i"]].append(r)
    return [min(by[b], key=lambda x:x["score_f"]) for b in sorted(by)]

def svg_header(w,h):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<rect width="100%" height="100%" fill="white"/>
<style>text{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;fill:#202124}} .small{{font-size:11px;fill:#5f6368}} .title{{font-size:18px;font-weight:700}} .subtitle{{font-size:12px;fill:#5f6368}} .axis{{stroke:#555;stroke-width:1}} .grid{{stroke:#ddd;stroke-width:.7}} .legend{{font-size:12px}}</style>
'''

def plot_lr_loss_facets(rows, title, out, subtitle=""):
    if not rows: return False
    batches=sorted({r["batch_i"] for r in rows})
    cols=min(3,len(batches)); panel_w=420; panel_h=320; top=75; legend_h=90
    w=cols*panel_w+40; h=top+math.ceil(len(batches)/cols)*panel_h+legend_h
    parts=[svg_header(w,h), f'<text x="24" y="28" class="title">{esc(title)}</text>']
    if subtitle: parts.append(f'<text x="24" y="48" class="subtitle">{esc(subtitle)}</text>')
    legend_methods=[]
    for idx,b in enumerate(batches):
        row=idx//cols; col=idx%cols; x0=30+col*panel_w; y0=top+row*panel_h; lm=58
        gx=x0+lm; gy=y0+20; plot_w=330; plot_h=235
        sub=[r for r in rows if r["batch_i"]==b]
        lrs=[log2(r["lr_f"]) for r in sub]; scores=[r["score_f"] for r in sub]
        xmin,xmax=min(lrs),max(lrs); ymin,ymax=min(scores),max(scores)
        if xmax==xmin: xmax=xmin+1
        pad=(ymax-ymin)*0.08 or 0.001; ymin-=pad; ymax+=pad
        def sx(lr): return gx+(log2(lr)-xmin)/(xmax-xmin)*plot_w
        def sy(sc): return gy+plot_h-(sc-ymin)/(ymax-ymin)*plot_h
        parts.append(f'<g><text x="{gx}" y="{y0+6}" font-size="14" font-weight="700">batch={fmt_batch(b)}</text>')
        ticks=sorted({r["lr_f"] for r in sub})
        for t in ticks:
            x=sx(t); parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{gy}" y2="{gy+plot_h}" class="grid"/><text x="{x:.1f}" y="{gy+plot_h+16}" class="small" text-anchor="middle">{nice_lr(t)}</text>')
        for i in range(5):
            val=ymin+(ymax-ymin)*i/4; y=sy(val)
            parts.append(f'<line x1="{gx}" x2="{gx+plot_w}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{gx-7}" y="{y+3:.1f}" class="small" text-anchor="end">{nice_score(val)}</text>')
        parts.append(f'<line x1="{gx}" x2="{gx}" y1="{gy}" y2="{gy+plot_h}" class="axis"/><line x1="{gx}" x2="{gx+plot_w}" y1="{gy+plot_h}" y2="{gy+plot_h}" class="axis"/>')
        for m in sorted({r["method_norm"] for r in sub}):
            pts=sorted([r for r in sub if r["method_norm"]==m], key=lambda r:r["lr_f"])
            c=color(m); kind=MARKERS.get(m,"circle")
            coords=[(sx(r["lr_f"]), sy(r["score_f"])) for r in pts]
            if len(coords)>1: parts.append('<polyline fill="none" stroke="{}" stroke-width="2" points="{}"/>'.format(c, ' '.join(f'{x:.1f},{y:.1f}' for x,y in coords)))
            for r,(x,y) in zip(pts,coords): parts.append(shape_svg(kind,x,y,c,4.2))
            br=min(pts,key=lambda r:r["score_f"]); parts.append(shape_svg(kind,sx(br["lr_f"]),sy(br["score_f"]),c,7.0,fill="white"))
            if m not in legend_methods: legend_methods.append(m)
        parts.append('</g>')
    lx=30; ly=h-62
    parts.append(f'<text x="{lx}" y="{ly-12}" class="subtitle">Legend: open marker = per-method optimum</text>')
    x=lx
    for m in legend_methods:
        if x> w-190: x=lx; ly+=22
        parts.append(shape_svg(MARKERS.get(m,"circle"),x+8,ly,color(m),5)); parts.append(f'<text x="{x+20}" y="{ly+4}" class="legend">{esc(m)}</text>'); x += 165
    parts.append('</svg>'); out.write_text('\n'.join(parts)); return True

def plot_optimal(rows, out_score, out_lr, title):
    if not rows: return False
    best=best_by_batch_method(rows); batches=sorted({r["batch_i"] for r in best}); methods=sorted({r["method_norm"] for r in best})
    for metric,out,ylabel,logy in [("score_f",out_score,"best final loss / BPB",False),("lr_f",out_lr,"optimal learning rate",True)]:
        w=max(900,130*len(batches)); h=520; gx=80; gy=80; pw=w-170; ph=320
        vals=[r[metric] for r in best if math.isfinite(r[metric])]
        if logy: vals=[log2(v) for v in vals if v>0]
        ymin,ymax=min(vals),max(vals); pad=(ymax-ymin)*0.08 or .01; ymin-=pad; ymax+=pad
        bx=[log2(b) for b in batches]; xmin,xmax=min(bx)-.1,max(bx)+.1
        def sx(b): return gx+(log2(b)-xmin)/(xmax-xmin)*pw
        def sy(v):
            vv=log2(v) if logy else v
            return gy+ph-(vv-ymin)/(ymax-ymin)*ph
        parts=[svg_header(w,h), f'<text x="24" y="32" class="title">{esc(title)}: {esc(ylabel)}</text>']
        for b in batches:
            x=sx(b); parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{gy}" y2="{gy+ph}" class="grid"/><text x="{x:.1f}" y="{gy+ph+18}" class="small" text-anchor="middle">{fmt_batch(b)}</text>')
        for i in range(5):
            yy=ymin+(ymax-ymin)*i/4; label=nice_lr(2**yy) if logy else nice_score(yy); y=gy+ph-(yy-ymin)/(ymax-ymin)*ph
            parts.append(f'<line x1="{gx}" x2="{gx+pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{gx-8}" y="{y+3:.1f}" class="small" text-anchor="end">{label}</text>')
        parts.append(f'<line x1="{gx}" x2="{gx}" y1="{gy}" y2="{gy+ph}" class="axis"/><line x1="{gx}" x2="{gx+pw}" y1="{gy+ph}" y2="{gy+ph}" class="axis"/>')
        by={(r["batch_i"],r["method_norm"]):r for r in best}; legend=[]
        for m in methods:
            pts=[]
            for b in batches:
                r=by.get((b,m))
                if r and math.isfinite(r[metric]): pts.append((b,r))
            if not pts: continue
            c=color(m); kind=MARKERS.get(m,"circle"); coords=[(sx(b),sy(r[metric])) for b,r in pts]
            if len(coords)>1: parts.append('<polyline fill="none" stroke="{}" stroke-width="2" points="{}"/>'.format(c, ' '.join(f'{x:.1f},{y:.1f}' for x,y in coords)))
            for x,y in coords: parts.append(shape_svg(kind,x,y,c,4.8))
            legend.append(m)
        lx=70; ly=h-75; x=lx
        for m in legend:
            if x>w-180: x=lx; ly+=22
            parts.append(shape_svg(MARKERS.get(m,"circle"),x+8,ly,color(m),5)); parts.append(f'<text x="{x+20}" y="{ly+4}" class="legend">{esc(m)}</text>'); x+=155
        parts.append('</svg>'); out.write_text('\n'.join(parts))
    return True

def write_best_table(rows, path):
    best=best_by_batch_method(rows); identity={}
    for r in best:
        if "identity" in r["method_norm"] and (r["batch_i"] not in identity or r["score_f"]<identity[r["batch_i"]]["score_f"]): identity[r["batch_i"]]=r
    winners={(r["batch_i"],r["method_norm"]):True for r in winners_by_batch(rows)}
    out=[]
    for r in sorted(best,key=lambda x:(x["batch_i"],x["score_f"],x["method_norm"])):
        base=identity.get(r["batch_i"]); delta=r["score_f"]-base["score_f"] if base else math.nan
        out.append({"batch_label":r.get("batch_label") or fmt_batch(r["batch_i"]),"batch":r["batch_i"],"winner_in_batch":"yes" if winners.get((r["batch_i"],r["method_norm"])) else "","method":r["method_norm"],"variant":r.get("variant",""),"source":r.get("source",""),"family":r.get("family",""),"optimal_lr":f'{r["lr_f"]:.8g}',"best_score":f'{r["score_f"]:.12g}',"delta_vs_identity_best":"" if not math.isfinite(delta) else f'{delta:+.12g}',"quality":r.get("quality",""),"seed":r.get("seed",""),"path":r.get("path","")})
    with path.open('w',newline='') as f:
        fields=list(out[0].keys()) if out else []
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(out)
    return out

def table_html(rows, cols, limit=None):
    rows=rows[:limit] if limit else rows
    s=['<table><thead><tr>'+''.join(f'<th>{esc(c)}</th>' for c in cols)+'</tr></thead><tbody>']
    for r in rows: s.append('<tr>'+''.join(f'<td>{esc(r.get(c,""))}</td>' for c in cols)+'</tr>')
    s.append('</tbody></table>'); return '\n'.join(s)

def read_display(path): return list(csv.DictReader(path.open())) if path.exists() else []

plot_lr_loss_facets(main_rows, "Small-bsz sweep: LR vs final loss", OUT/"lr_loss_by_batch_v26_small.svg", "same-driver streaming DDP; open marker indicates per-method optimum")
plot_lr_loss_facets(catalog_full, "Catalog full/historical sweeps: LR vs final loss", OUT/"lr_loss_by_batch_catalog_full.svg", "broader context from sweep_catalog/all_sweeps.csv")
plot_optimal(main_rows, OUT/"optimal_score_by_batch_v26_small.svg", OUT/"optimal_lr_by_batch_v26_small.svg", "Small-bsz v26")
plot_optimal(catalog_full, OUT/"optimal_score_by_batch_catalog_full.svg", OUT/"optimal_lr_by_batch_catalog_full.svg", "Catalog full/historical")
write_best_table(main_rows, OUT/"optimal_summary_v26_small.csv")
write_best_table(catalog_full, OUT/"optimal_summary_catalog_full.csv")
winners=winners_by_batch(main_rows)
with (OUT/"winner_by_batch_v26_small.csv").open('w',newline='') as f:
    fields=["batch_label","batch","winner_method","variant","optimal_lr","best_score","path"]
    w=csv.DictWriter(f,fields); w.writeheader()
    for r in winners: w.writerow({"batch_label":r.get("batch_label") or fmt_batch(r["batch_i"]),"batch":r["batch_i"],"winner_method":r["method_norm"],"variant":r.get("variant",""),"optimal_lr":f'{r["lr_f"]:.8g}',"best_score":f'{r["score_f"]:.12g}',"path":r.get("path","")})

missing=sum(1 for r in main_rows if not (ROOT/(r.get("path") or "")).exists())
main_disp=read_display(OUT/"optimal_summary_v26_small.csv"); cat_disp=read_display(OUT/"optimal_summary_catalog_full.csv"); win_disp=read_display(OUT/"winner_by_batch_v26_small.csv")
html_doc=f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Sigma-search small-bsz dashboard</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;margin:28px;color:#202124}} h1{{margin-bottom:.2em}} .card{{border:1px solid #ddd;border-radius:12px;padding:16px;margin:18px 0;box-shadow:0 1px 3px rgba(0,0,0,.05)}} .note{{background:#fff8e1;border-left:4px solid #f4b400;padding:10px 14px;margin:16px 0}} img{{max-width:100%;height:auto;border:1px solid #eee;border-radius:8px;background:white}} table{{border-collapse:collapse;font-size:13px;width:100%;margin:10px 0}} th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left}} th{{background:#f6f8fa}} code{{background:#f6f8fa;padding:2px 4px;border-radius:4px}} .small{{color:#666;font-size:13px}}
</style></head><body>
<h1>Sigma-search small-bsz sweep dashboard</h1><p class=\"small\">Generated under <code>{esc(OUT)}</code>. Lower final loss/BPB is better.</p>
<div class=\"note\"><b>Training curve note:</b> summary CSVs are present, but raw <code>result.json</code> artifacts referenced by <code>figures/current_same_driver_v26/raw_rows.csv</code> are missing in this clean workdir ({missing}/{len(main_rows)} missing). Exact optimal-LR <i>training loss curves</i> need those raw artifacts restored under <code>search_evals/v26_small_bsz_streaming_fair_20260430/**/result.json</code>, then rerun <code>python3 analysis/build_small_bsz_dashboard.py</code>. The dashboard below fully covers LR-loss curves and optimal result tables from the available summaries.</div>
<div class=\"card\"><h2>1. Different bsz: LR-loss curves (small-bsz v26)</h2><p>Each facet is one batch size; open markers indicate per-method optimal LR.</p><img src=\"lr_loss_by_batch_v26_small.svg\"></div>
<div class=\"card\"><h2>2. Optimal results across batch sizes (small-bsz v26)</h2><img src=\"optimal_score_by_batch_v26_small.svg\"><img src=\"optimal_lr_by_batch_v26_small.svg\"></div>
<div class=\"card\"><h2>3. Winner by batch</h2>{table_html(win_disp,["batch_label","batch","winner_method","variant","optimal_lr","best_score","path"])}</div>
<div class=\"card\"><h2>4. Different optimizer optimal result summary</h2><p>CSV: <code>optimal_summary_v26_small.csv</code></p>{table_html(main_disp,["batch_label","winner_in_batch","method","variant","optimal_lr","best_score","delta_vs_identity_best","seed","path"])}</div>
<div class=\"card\"><h2>5. Catalog-wide LR-loss curves (full/historical)</h2><p>Broader comparison from <code>results/sweep_catalog/all_sweeps.csv</code>.</p><img src=\"lr_loss_by_batch_catalog_full.svg\"></div>
<div class=\"card\"><h2>6. Catalog-wide optimal summary</h2><img src=\"optimal_score_by_batch_catalog_full.svg\"><img src=\"optimal_lr_by_batch_catalog_full.svg\"><p>CSV: <code>optimal_summary_catalog_full.csv</code></p>{table_html(cat_disp,["batch_label","winner_in_batch","method","variant","source","optimal_lr","best_score","delta_vs_identity_best","quality"],120)}</div>
</body></html>"""
(OUT/"dashboard.html").write_text(html_doc)
print(json.dumps({"out_dir":str(OUT),"dashboard":str(OUT/"dashboard.html"),"main_rows":len(main_rows),"catalog_full_rows":len(catalog_full),"raw_result_json_missing":missing,"winners":[{"batch":r.get("batch_label") or fmt_batch(r["batch_i"]),"method":r["method_norm"],"lr":r["lr_f"],"score":r["score_f"]} for r in winners]},indent=2))
