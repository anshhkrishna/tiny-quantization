"""Regenerates results/headline.png from the committed results/rigor.log -- parses the
mean/std test accuracy per variant that the rigor step already printed (5-seed
uncertainty), rather than re-running training, so the plot always matches exactly
what is checked into the log.

Run with `python src/plot_headline.py` from the project root.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LOG_PATH = RESULTS_DIR / "rigor.log"
OUT_PATH = RESULTS_DIR / "headline.png"

VARIANTS = ("float32", "naive_int4", "per_tensor_int8", "per_channel_int8", "gptq_int4")
LABELS = {
    "float32": "float32",
    "naive_int4": "naive int4",
    "per_tensor_int8": "per-tensor int8",
    "per_channel_int8": "per-channel int8",
    "gptq_int4": "gptq-style int4",
}
COLORS = {
    "float32": "#2a78d6",
    "naive_int4": "#eb6834",
    "per_tensor_int8": "#1baf7a",
    "per_channel_int8": "#eda100",
    "gptq_int4": "#e87ba4",
}

SUMMARY_RE = re.compile(r"^(\S+)\s+mean\s*=\s*(-?\d+\.\d+)%\s+std\s*=\s*(-?\d+\.\d+)%\s+\(n=(\d+)\)$")


def parse_log(path=LOG_PATH):
    """Returns {variant: (mean_pct, std_pct, n)} from the committed rigor log's
    summary lines.
    """
    stats = {}
    for line in path.read_text().splitlines():
        m = SUMMARY_RE.match(line)
        if m:
            name, mean, std, n = m.groups()
            stats[name] = (float(mean), float(std), int(n))
    missing = [v for v in VARIANTS if v not in stats]
    assert not missing, f"missing summary rows for {missing} -- log format changed?"
    return stats


def plot(stats, out_path=OUT_PATH):
    fig, ax = plt.subplots(figsize=(1600 / 150, 900 / 150), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xs = list(range(len(VARIANTS)))
    means = [stats[v][0] for v in VARIANTS]
    stds = [stats[v][1] for v in VARIANTS]
    colors = [COLORS[v] for v in VARIANTS]
    n_seeds = stats[VARIANTS[0]][2]

    ax.bar(xs, means, yerr=stds, capsize=6, color=colors, width=0.6,
           error_kw={"linewidth": 1.5, "ecolor": "#0b0b0b"})
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[v] for v in VARIANTS], fontsize=12)

    float32_mean = stats["float32"][0]
    naive_mean = stats["naive_int4"][0]
    ax.axhline(float32_mean, linestyle="--", linewidth=1.5, color=COLORS["float32"], alpha=0.8,
               label=f"float32 reference ({float32_mean:.2f}%)")
    ax.axhline(naive_mean, linestyle="--", linewidth=1.5, color=COLORS["naive_int4"], alpha=0.8,
               label=f"naive int4 reference ({naive_mean:.2f}%)")

    ax.set_ylim(min(means) - max(stds) - 2, max(means) + max(stds) + 1)
    ax.set_ylabel(f"test accuracy, mean +/- std ({n_seeds} seeds)", fontsize=13)
    ax.set_title("gptq-style int4 recovers part of naive int4's accuracy loss, not all of it", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    plot(parse_log())
