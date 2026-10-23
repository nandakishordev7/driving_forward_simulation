"""
plot_real_comparison.py

Reads the two REAL loss logs produced by training:
    checkpoints/loss_log_baseline.json   <- from train_gaussian_baseline.py
    checkpoints/loss_log.json            <- from train_gaussian.py (with RefineNet)

Produces a 4-panel figure:
    Panel 1 (top-wide) : Baseline val vs Refined D3 val — main comparison
    Panel 2 (mid-left) : Train loss  baseline vs refined
    Panel 3 (mid-right): Per-iteration val loss D0→D3 (refinement breakdown)
    Panel 4 (bottom)   : Per-epoch improvement  Δ = baseline_val - refined_D3_val

Run:
    python plot_real_comparison.py
    python plot_real_comparison.py --baseline checkpoints/loss_log_baseline.json
                                   --refined  checkpoints/loss_log.json
                                   --out      my_plot.png
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# ── Colours ───────────────────────────────────────────────────────────────────
C_BASE  = "#e05c5c"   # red    — baseline
C_D0    = "#888888"   # grey   — D0 (initial, no correction yet)
C_D1    = "#80b4e0"   # light blue
C_D2    = "#4090d0"   # mid blue
C_D3    = "#1a6fbd"   # dark blue — final refined
C_DELTA = "#f0c040"   # yellow — improvement curve
C_TRAIN_BASE   = "#e08080"
C_TRAIN_REFINE = "#60a8d0"
BG      = "#0d0d0d"
AX_BG   = "#141414"
GRID_C  = "#252525"
SPINE_C = "#404040"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors="white", labelsize=8.5)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_C)
    ax.grid(color=GRID_C, linewidth=0.55, linestyle="--")


def _legend(ax, **kw):
    leg = ax.legend(facecolor="#1c1c1c", edgecolor="#555",
                    labelcolor="white", fontsize=8, **kw)
    return leg


def _smooth(arr, w=3):
    """Simple moving average to reduce noise in real training curves."""
    if w <= 1:
        return np.array(arr)
    kernel = np.ones(w) / w
    pad    = np.pad(arr, (w // 2, w // 2), mode="edge")
    return np.convolve(pad, kernel, mode="valid")[:len(arr)]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _align(base, refn):
    """Trim both logs to the shorter run so arrays line up."""
    n = min(len(base["epochs"]), len(refn["epochs"]))
    return n


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_real_comparison(base_log, refn_log,
                         save_path="real_comparison.png",
                         smooth_w=3,
                         show=True):

    n      = _align(base_log, refn_log)
    epochs = base_log["epochs"][:n]

    # raw arrays
    base_train = np.array(base_log["train_loss"][:n])
    base_val   = np.array(base_log["val_loss"][:n])

    refn_train = np.array(refn_log["train_total"][:n])
    d0_val     = np.array(refn_log["refine_d0_val"][:n])
    d1_val     = np.array(refn_log["refine_d1_val"][:n])
    d2_val     = np.array(refn_log["refine_d2_val"][:n])
    d3_val     = np.array(refn_log["refine_d3_val"][:n])

    # smoothed versions for cleaner lines
    s_base_train = _smooth(base_train, smooth_w)
    s_base_val   = _smooth(base_val,   smooth_w)
    s_refn_train = _smooth(refn_train, smooth_w)
    s_d0   = _smooth(d0_val,  smooth_w)
    s_d1   = _smooth(d1_val,  smooth_w)
    s_d2   = _smooth(d2_val,  smooth_w)
    s_d3   = _smooth(d3_val,  smooth_w)

    # improvement signal
    delta        = base_val - d3_val          # positive = refined is better
    s_delta      = _smooth(delta, smooth_w)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    gs  = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.6, 1.2, 0.9],
        hspace=0.52, wspace=0.30,
        left=0.07, right=0.97,
        top=0.91,  bottom=0.07
    )

    ax_main   = fig.add_subplot(gs[0, :])    # row 0, both cols
    ax_train  = fig.add_subplot(gs[1, 0])    # row 1 left
    ax_iter   = fig.add_subplot(gs[1, 1])    # row 1 right
    ax_delta  = fig.add_subplot(gs[2, :])    # row 2, both cols

    for ax in [ax_main, ax_train, ax_iter, ax_delta]:
        _style_ax(ax)

    # ── Panel 1: main val comparison ─────────────────────────────────────────
    ax_main.plot(epochs, base_val,   color=C_BASE, lw=1.2,
                 alpha=0.35, label="_nolegend_")
    ax_main.plot(epochs, s_base_val, color=C_BASE, lw=2.4,
                 label="Baseline val (no refinement)")

    ax_main.plot(epochs, d3_val,   color=C_D3, lw=1.2,
                 alpha=0.35, label="_nolegend_")
    ax_main.plot(epochs, s_d3,     color=C_D3, lw=2.4,
                 label="Refined D₃ val  (K=3 iters)")

    ax_main.fill_between(epochs, s_base_val, s_d3,
                         where=(s_base_val >= s_d3),
                         alpha=0.13, color=C_D2,
                         label="Improvement region")

    # annotate best gap epoch
    gap         = s_base_val - s_d3
    best_epoch_idx = int(np.argmax(gap))
    bv_best     = s_base_val[best_epoch_idx]
    rv_best     = s_d3[best_epoch_idx]
    ep_best     = epochs[best_epoch_idx]

    ax_main.annotate(
        "", xy=(ep_best, rv_best), xytext=(ep_best, bv_best),
        arrowprops=dict(arrowstyle="<->", color=C_DELTA, lw=1.8)
    )
    ax_main.text(ep_best + 0.4, (bv_best + rv_best) / 2,
                 f"max Δ={gap[best_epoch_idx]:.4f}\n@ epoch {ep_best}",
                 color=C_DELTA, fontsize=8, va="center")

    # annotate final epoch gap
    ax_main.annotate(
        f"final Δ={gap[-1]:.4f}",
        xy=(epochs[-1], s_d3[-1]),
        xytext=(epochs[-1] - n * 0.18, s_d3[-1] - (bv_best - rv_best) * 0.35),
        color="white", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="white", lw=1.0)
    )

    ax_main.set_xlabel("Epoch", fontsize=10)
    ax_main.set_ylabel("Photometric Loss", fontsize=10)
    ax_main.set_title(
        "Validation Loss — Baseline vs Iterative Geometry Refinement",
        fontsize=12, pad=8
    )
    _legend(ax_main, loc="upper right")

    # ── Panel 2: train loss ───────────────────────────────────────────────────
    ax_train.plot(epochs, base_train,   color=C_TRAIN_BASE,   lw=1.0, alpha=0.3)
    ax_train.plot(epochs, s_base_train, color=C_TRAIN_BASE,   lw=2.0,
                  label="Baseline train")
    ax_train.plot(epochs, refn_train,   color=C_TRAIN_REFINE, lw=1.0, alpha=0.3)
    ax_train.plot(epochs, s_refn_train, color=C_TRAIN_REFINE, lw=2.0,
                  label="Refined train")

    ax_train.set_xlabel("Epoch", fontsize=9)
    ax_train.set_ylabel("Train Loss", fontsize=9)
    ax_train.set_title("Training Loss Comparison", fontsize=10, pad=6)
    _legend(ax_train, loc="upper right")

    # ── Panel 3: per-iteration val loss breakdown ─────────────────────────────
    iter_lines = [
        ("D₀  (no correction)", s_d0, C_D0, "--"),
        ("D₁  (+1 refinement)", s_d1, C_D1, "-"),
        ("D₂  (+2 refinements)", s_d2, C_D2, "-"),
        ("D₃  (+3 refinements)", s_d3, C_D3, "-"),
    ]
    for label, curve, col, ls in iter_lines:
        ax_iter.plot(epochs, curve, color=col, lw=1.9, ls=ls, label=label)

    # shade between D0 and D3 to show total refinement gain
    ax_iter.fill_between(epochs, s_d0, s_d3,
                         alpha=0.10, color=C_D2)

    ax_iter.set_xlabel("Epoch", fontsize=9)
    ax_iter.set_ylabel("Val Photometric Loss", fontsize=9)
    ax_iter.set_title("Per-Iteration Val Loss  D₀ → D₃", fontsize=10, pad=6)
    _legend(ax_iter, loc="upper right")

    # ── Panel 4: delta (improvement) curve ───────────────────────────────────
    ax_delta.axhline(0, color=SPINE_C, lw=1.0, ls="--")
    ax_delta.plot(epochs, delta,   color=C_DELTA, lw=1.0, alpha=0.3)
    ax_delta.plot(epochs, s_delta, color=C_DELTA, lw=2.2,
                  label="Δ = Baseline val − Refined D₃ val")

    ax_delta.fill_between(epochs, 0, s_delta,
                          where=(s_delta > 0),
                          alpha=0.15, color=C_DELTA,
                          label="Positive = refined is better")
    ax_delta.fill_between(epochs, 0, s_delta,
                          where=(s_delta < 0),
                          alpha=0.15, color=C_BASE,
                          label="Negative = baseline is better (early training)")

    # mark where refinement consistently wins
    first_positive = next(
        (i for i, v in enumerate(s_delta) if v > 0), None
    )
    if first_positive is not None:
        ax_delta.axvline(epochs[first_positive], color="white",
                         lw=1.0, ls=":", alpha=0.6)
        ax_delta.text(epochs[first_positive] + 0.3,
                      s_delta.max() * 0.7,
                      f"Refinement\nconsistently\nwins\n→ epoch {epochs[first_positive]}",
                      color="white", fontsize=7.5, va="top")

    ax_delta.set_xlabel("Epoch", fontsize=9)
    ax_delta.set_ylabel("Loss Reduction  (↑ better)", fontsize=9)
    ax_delta.set_title(
        "Per-Epoch Improvement from Refinement  "
        "Δ = Baseline − Refined",
        fontsize=10, pad=6
    )
    _legend(ax_delta, loc="lower right")

    # ── Stats box ─────────────────────────────────────────────────────────────
    final_base  = base_val[-1]
    final_d3    = d3_val[-1]
    pct_improve = (final_base - final_d3) / final_base * 100
    best_base   = float(np.min(base_val))
    best_d3     = float(np.min(d3_val))

    stats_txt = (
        f"  Final epoch results\n"
        f"  Baseline val   :  {final_base:.4f}\n"
        f"  Refined D₃ val :  {final_d3:.4f}\n"
        f"  Improvement    :  {pct_improve:+.2f}%\n"
        f"  Best baseline  :  {best_base:.4f}\n"
        f"  Best refined   :  {best_d3:.4f}"
    )
    fig.text(0.50, 0.955, stats_txt,
             color="white", fontsize=8.5,
             ha="center", va="top",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="#1e1e1e", edgecolor="#555", alpha=0.9),
             fontfamily="monospace")

    # ── Super-title ───────────────────────────────────────────────────────────
    fig.suptitle(
        "Real Training Results  ·  "
        "Baseline  vs  Iterative Geometry Refinement  "
        "( ΔDₖ = RefineNet(Dₖ, Irender − Igt) )",
        color="white", fontsize=13, y=0.995
    )

    plt.savefig(save_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[plot] Saved → {os.path.abspath(save_path)}")
    if show:
        plt.show()
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="C://Projects//driving_forward//checkpoints//loss_log_baseline.json",
        help="Path to loss_log_baseline.json from train_gaussian_baseline.py"
    )
    parser.add_argument(
        "--refined",
        default="C://Projects//driving_forward//checkpoints//loss_log.json",
        help="Path to loss_log.json from train_gaussian.py (with RefineNet)"
    )
    parser.add_argument(
        "--out",
        default="C://Projects//driving_forward//real_comparison.png"
    )
    parser.add_argument(
        "--smooth", type=int, default=3,
        help="Moving average window for smoothing noisy curves (default 3)"
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    # check files exist
    for p, name in [(args.baseline, "baseline log"),
                    (args.refined,  "refined log")]:
        if not os.path.exists(p):
            print(f"[plot] ERROR: {name} not found at {p}")
            print("       Run train_gaussian_baseline.py and "
                  "train_gaussian.py first.")
            raise SystemExit(1)

    base_log = _load(args.baseline)
    refn_log = _load(args.refined)

    print(f"[plot] Baseline epochs : {len(base_log['epochs'])}")
    print(f"[plot] Refined epochs  : {len(refn_log['epochs'])}")

    plot_real_comparison(
        base_log, refn_log,
        save_path=args.out,
        smooth_w=args.smooth,
        show=not args.no_show
    )
