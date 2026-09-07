"""Aggregate benchmark CSVs into Table A (same CPU) and Table B (H200 vs CPU).

Usage: python make_tables.py results/*.csv
Each CSV starts with a '# key=value ...' header line, then 'sim,device,envs,variant,frames_per_s'.
"""
import csv, glob, re, sys
from collections import defaultdict

rows = []  # (tag, sim, device, envs, variant, fps, hostinfo)
for path in sys.argv[1:] or glob.glob("results/*.csv"):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    hdr = dict(re.findall(r"(\w+)=(\S+)", lines[0])) if lines and lines[0].startswith("#") else {}
    body = [l for l in lines if not l.startswith("#")]
    for r in csv.DictReader(body):
        rows.append((hdr.get("tag", ""), r["sim"], r["device"], int(r["envs"]), r["variant"], float(r["frames_per_s"]), hdr))

def fmt(x):
    return f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.1f}k" if x >= 1e4 else f"{x:,.0f}"

def series(tag, sim, device, variant):
    return sorted([(e, f) for t, s, d, e, v, f, _ in rows if t == tag and s == sim and d == device and v == variant])

for tag in sorted({t for t, *_ in rows}):
    print(f"\n### tag = {tag}")
    hosts = {h.get("host") for t, *_, h in rows if t == tag}
    print(f"host(s): {', '.join(sorted(map(str, hosts)))}")
    for sim, dev, var in [("numpy", "cpu", "env_step"), ("jax", "cpu", "env_step"), ("jax", "cpu", "step_only"), ("jax", "gpu", "env_step"), ("jax", "gpu", "step_only")]:
        s = series(tag, sim, dev, var)
        if not s: continue
        peak = max(s, key=lambda x: x[1])
        print(f"- {sim} on {dev} [{var}]: " + ", ".join(f"{e}→{fmt(f)}" for e, f in s) + f"   | peak {fmt(peak[1])} @ {peak[0]} envs")

# Table A: same CPU, env_step, numpy vs jax (per tag that has both)
for tag in sorted({t for t, *_ in rows}):
    a, b = series(tag, "numpy", "cpu", "env_step"), series(tag, "jax", "cpu", "env_step")
    if a and b:
        pa, pb = max(a, key=lambda x: x[1]), max(b, key=lambda x: x[1])
        print(f"\nTABLE A ({tag}): NumPy peak {fmt(pa[1])} @ {pa[0]} procs; JAX-CPU peak {fmt(pb[1])} @ {pb[0]} envs; ratio {pb[1]/pa[1]:,.0f}x")
        # matched parallelism where available
        for e, f in a:
            m = dict(b).get(e)
            if m: print(f"   matched {e:>4} parallel: numpy {fmt(f)}  jax {fmt(m)}  ratio {m/f:,.0f}x")
# Table B: gpu env_step vs numpy cpu peak (any tag)
g = [(t, e, f) for t, s, d, e, v, f, _ in rows if s == "jax" and d == "gpu" and v == "env_step"]
n = [(t, e, f) for t, s, d, e, v, f, _ in rows if s == "numpy" and d == "cpu" and v == "env_step"]
if g and n:
    tg, eg, fg = max(g, key=lambda x: x[2]); tn, en, fn = max(n, key=lambda x: x[2])
    print(f"\nTABLE B: JAX H200 peak {fmt(fg)} @ {eg} envs ({tg}) vs NumPy CPU peak {fmt(fn)} @ {en} procs ({tn}); ratio {fg/fn:,.0f}x")
