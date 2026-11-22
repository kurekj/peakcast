"""Draw the figures of the paper from the recorded measurements.

    python benchmarks/make_figures.py OUTPUT_DIRECTORY

Needs matplotlib. Reads benchmarks/results (grid.csv, truth.csv, summary.json, live/*.jsonl).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay  # noqa: E402

import peakcast  # noqa: E402

RESULTS = os.environ.get("PEAKCAST_RESULTS", os.path.join(HERE, "results"))
GIB = replay.GIB
# categorical colours in fixed order (blue, orange, aqua, yellow, magenta), dark ink for text
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd6"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 1.6,
})


def figure_laws(out: str) -> None:
    """Measured peaks, probes, fitted law and bound for four workloads at a 24 GiB budget."""
    first = replay.main_device(RESULTS)
    grid = replay.load_grid(os.path.join(RESULTS, first))
    truth = replay.load_truth(os.path.join(RESULTS, first))
    cal = peakcast.default_calibration()
    wanted = [("resnet50|adamw|0", "ResNet-50, AdamW"), ("hf_bert_base|adamw|0", "BERT-base, AdamW"),
              ("vgg16|adamw|0", "VGG-16, AdamW"), ("hf_gpt2_medium|adamw|1", "GPT-2 medium, AdamW, AMP"),
              ("hf_bert_large|adamw|0", "BERT-large, AdamW"), ("hf_gpt2_small|adamw|1", "GPT-2 small, AdamW, AMP"),
              ("convnext_tiny|adamw|0", "ConvNeXt-T, AdamW"), ("vit_b_16|adamw|0", "ViT-B/16, AdamW")]
    cases = [c for c in wanted if c[0] in grid][:4]
    cap = (24 if first == "l40s" else 16) * GIB
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), sharey=True)
    for ax, (key, title) in zip(axes, cases, strict=True):
        w = grid[key]
        rows = w.rows[w.axis_default]
        law, _, _ = replay.replay_law(w, cap, truth=truth, calibration=cal)
        pts = sorted((b, r["peak_alloc"] / GIB) for b, r in rows.items() if r["status"] == "ok")
        probed = {p.batch for p in law.probes}
        ax.plot([b for b, _ in pts if b not in probed], [y for b, y in pts if b not in probed], "o", ms=3.5,
                mfc="white", mec=MUTED, mew=0.9, label="measured")
        ax.plot([b for b, _ in pts if b in probed], [y for b, y in pts if b in probed], "o", ms=4.5,
                color=BLUE, label="probes")
        xs = np.geomspace(1, pts[-1][0], 200)
        ax.plot(xs, [law.expected(x) / GIB for x in xs], color=BLUE, label="fitted law")
        top = max(probed)
        ub = []
        for x in xs:
            if x >= top:
                try:
                    ub.append((x, law.upper(x, 0.01) / GIB))
                except KeyError:  # beyond the calibrated range the bound is not defined
                    break
        ax.plot([x for x, _ in ub], [y for _, y in ub], color=ORANGE, ls="--", label="bound, 1% risk")
        ax.axhline(cap / GIB, color=MUTED, lw=0.8, ls=":")
        best = law.max_batch(cap, 0.01)
        lo = truth.get((key, w.axis_default, cap), (None, None))[0]
        ax.axvline(best, color=ORANGE, lw=0.9)
        if lo:
            ax.axvline(lo, color=INK, lw=0.9, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_title(f"{title}\nrecommended {best}, fits up to {lo}", fontsize=8)
        ax.set_xlabel("batch size")
    axes[0].set_ylabel("peak allocated memory (GiB)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.savefig(os.path.join(out, "fig_laws.png"))
    plt.close(fig)


def figure_calibration(out: str) -> None:
    with open(os.path.join(RESULTS, "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    first = summary["main"]
    main = summary[first]
    other = [d for d in summary if d not in ("main", first)]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax = axes[0]
    series = [("batch axis, held-out families", main["lofo"], BLUE, "o"),
              ("second axis beyond probes, held-out families", main["two_axes"].get("s", {}).get("lofo", {}),
               ORANGE, "s")]
    for device in other:
        if "transfer_from_main" in summary[device]:
            series.append(("24 GB GPU, calibration of the 48 GB GPU", summary[device]["transfer_from_main"], AQUA, "^"))
    for label, data, colour, marker in series:
        pts = sorted((float(k), v) for k, v in data.items() if "@" not in k and v["n"])
        if not pts:
            continue
        x = [100 * p[0] for p in pts]
        y = [100 * p[1]["violation_rate"] for p in pts]
        lo = [100 * (p[1]["violation_rate"] - p[1]["violation_ci"][0]) for p in pts]
        hi = [100 * (p[1]["violation_ci"][1] - p[1]["violation_rate"]) for p in pts]
        ax.errorbar(x, y, yerr=[lo, hi], color=colour, marker=marker, ms=5, lw=1.2, capsize=2.5, label=label)
    ax.plot([0, 21], [0, 21], color=MUTED, lw=0.9, ls=":")
    ax.text(15.5, 16.8, "rate = risk", color=MUTED, fontsize=7.5, rotation=40)
    ax.set_xlabel("chosen risk (%)")
    ax.set_ylabel("ran out of memory (%)")
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 21)
    ax.legend(fontsize=7.5, loc="upper left")

    ax = axes[1]
    margins = sorted((float(k), v) for k, v in main["margins"].items() if float(k) > 0)
    ax.plot([100 * v["utilization"]["median"] for _, v in margins], [100 * v["violation_rate"] for _, v in margins],
            color=MUTED, marker="o", ms=4, mfc="white", label="fitted law with a fixed margin")
    for m, v in margins:
        if m in (0.05, 0.1, 0.2, 0.3, 0.5):
            ax.annotate(f"{100 * m:.0f}%", (100 * v["utilization"]["median"], 100 * v["violation_rate"]),
                        textcoords="offset points", xytext=(-5, 5), fontsize=7, color=MUTED, ha="right")
    risks = sorted((float(k), v) for k, v in main["lofo"].items() if "@" not in k)
    ax.plot([100 * v["utilization"]["median"] for _, v in risks], [100 * v["violation_rate"] for _, v in risks],
            color=BLUE, marker="o", ms=5, label="peakcast, calibrated bound")
    for r, v in risks:
        ax.annotate(f"{100 * r:g}%", (100 * v["utilization"]["median"], 100 * v["violation_rate"]),
                    textcoords="offset points", xytext=(-3, -12) if r <= 0.01 else (6, -9), fontsize=7,
                    color=BLUE, ha="right" if r <= 0.01 else "left")
    rule = sorted((float(k), v) for k, v in main["autobatch_rule"].items())
    ax.plot([100 * v["utilization"]["median"] for _, v in rule], [100 * v["violation_rate"] for _, v in rule],
            color=ORANGE, marker="s", ms=4.5, label="AutoBatch rule (replayed)")
    for f, v in rule:
        ax.annotate(f"{100 * f:.0f}%", (100 * v["utilization"]["median"], 100 * v["violation_rate"]),
                    textcoords="offset points", xytext=(-4, 6), fontsize=7, color=ORANGE, ha="right")
    none = main["margins"]["0"]
    ax.set_ylim(-1.5, 26)
    ax.text(0.80, 0.62, f"no margin:\n{100 * none['violation_rate']:.0f}% at "
            f"{100 * none['utilization']['median']:.0f}%", transform=ax.transAxes, ha="right", va="center",
            fontsize=7.5, color=MUTED)
    ax.set_xlabel("median batch size, % of the largest that fits")
    ax.set_ylabel("ran out of memory (%)")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_calibration.png"))
    plt.close(fig)


def figure_tools(out: str) -> None:
    """Live comparison: time to an answer and batch size relative to the best tool on that case."""
    rows = []
    for path in glob.glob(os.path.join(RESULTS, "live", "compare", "*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if '"tool"' in line]
    if not rows:
        return
    risks = sorted({r["tool"] for r in rows if r["tool"].startswith("peakcast@")}, key=lambda t: float(t[9:]))[:2]
    order = [*risks, "lightning-power", "lightning-binary", "toma", "torch-max-mem"]
    names = [*(f"peakcast\n{100 * float(t[9:]):g}% risk" for t in risks), "Lightning\npower",
             "Lightning\nbinary search", "toma", "torch-max-mem"]
    best: dict = {}
    for r in rows:
        if r["verify"] == "ok":
            key = (r["model"], r["cap_label"])
            best[key] = max(best.get(key, 0), r["batch"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    for ax, field in zip(axes, ("seconds", "relative"), strict=True):
        data = []
        for tool in order:
            sub = [r for r in rows if r["tool"] == tool and r["cap_label"] != "full"]
            if field == "seconds":
                data.append([r["seconds"] for r in sub])
            else:
                data.append([100 * r["batch"] / best[(r["model"], r["cap_label"])] for r in sub
                             if r["verify"] == "ok" and (r["model"], r["cap_label"]) in best])
        box = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops={"color": INK, "lw": 1.3},
                         flierprops={"ms": 2.5, "mec": MUTED}, whiskerprops={"color": MUTED},
                         capprops={"color": MUTED})
        for patch, colour in zip(box["boxes"], [BLUE, BLUE, ORANGE, ORANGE, AQUA, YELLOW], strict=True):
            patch.set(facecolor=colour, alpha=0.55, edgecolor=MUTED)
        ax.set_xticks(range(1, len(order) + 1), names, fontsize=7.5)
        if field == "seconds":
            ax.set_yscale("log")
            ax.set_ylabel("time to a batch size (s)")
        else:
            ax.set_ylabel("batch size, % of the largest verified one")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_tools.png"))
    plt.close(fig)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    figure_laws(out)
    figure_calibration(out)
    figure_tools(out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
