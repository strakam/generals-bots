"""Figure + LaTeX table of simulator throughput vs. parallel environments.

Usage:
  python plot_throughput.py --numpy-tag amd-threads64-shared --jaxcpu-tag amd-threads64-shared \
                            --gpu-tag h200-shared --out ../../figures/sim_throughput.pdf --tex results/sim_throughput_table.tex
Frame = one environment advanced by one game turn. For NumPy one process = one environment.
"""
import argparse, csv, glob, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        hdr = dict(re.findall(r"(\w+)=(\S+)", lines[0])) if lines and lines[0].startswith("#") else {}
        for r in csv.DictReader([l for l in lines if not l.startswith("#")]):
            rows.append(dict(tag=hdr.get("tag", ""), host=hdr.get("host", ""), sim=r["sim"], device=r["device"],
                             envs=int(r["envs"]), variant=r["variant"], fps=float(r["frames_per_s"])))
    return rows

def series(rows, tag, sim, device, variant="env_step"):
    """`tag` may be a comma-separated list of repetition tags; the median per env count is returned."""
    import statistics
    tags = set(tag.split(","))
    by_env = {}
    for r in rows:
        if r["tag"] in tags and r["sim"] == sim and r["device"] == device and r["variant"] == variant:
            by_env.setdefault(r["envs"], []).append(r["fps"])
    return sorted((e, statistics.median(v)) for e, v in by_env.items())

def fmt(x):
    return f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k" if x >= 1e4 else f"{x/1e3:.1f}k" if x >= 1e3 else f"{x:.0f}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--numpy-tag", required=True); ap.add_argument("--jaxcpu-tag", required=True); ap.add_argument("--gpu-tag", required=True)
    ap.add_argument("--out", default="sim_throughput.pdf"); ap.add_argument("--tex", default=None)
    ap.add_argument("--cpu-label", default="CPU (64 threads)"); ap.add_argument("--gpu-label", default="one H200 GPU")
    a = ap.parse_args()
    rows = load(glob.glob("results/*.csv"))
    S = {
        "NumPy, " + a.cpu_label: series(rows, a.numpy_tag, "numpy", "cpu"),
        "JAX, " + a.cpu_label: series(rows, a.jaxcpu_tag, "jax", "cpu"),
        "JAX, " + a.gpu_label: series(rows, a.gpu_tag, "jax", "gpu"),
    }
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    styles = [dict(marker="s", color="#7f7f7f"), dict(marker="o", color="#0ea5e9"), dict(marker="^", color="#ef4444")]
    for (label, s), st in zip(S.items(), styles):
        if not s: continue
        xs, ys = zip(*s)
        ax.plot(xs, ys, label=label, lw=1.2, ms=3.5, **st)
        px, py = max(s, key=lambda t: t[1])
        ax.annotate(fmt(py), (px, py), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7, color=st["color"])
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Parallel environments"); ax.set_ylabel("Frames per second")
    ax.grid(True, which="major", lw=0.4, alpha=0.5); ax.grid(True, which="minor", lw=0.2, alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(a.out); print("wrote", a.out)

    if a.tex:
        cols = sorted({e for s in S.values() for e, _ in s})
        lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
                 "Parallel environments & " + " & ".join(f"${c:,}$".replace(",", "{,}") for c in cols) + r" \\", r"\midrule"]
        for label, s in S.items():
            d = dict(s)
            lines.append(label + " & " + " & ".join(fmt(d[c]) if c in d else "--" for c in cols) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(a.tex, "w").write("\n".join(lines) + "\n"); print("wrote", a.tex)
