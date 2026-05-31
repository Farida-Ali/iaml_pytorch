"""
scripts/plot_psnr_curve.py
──────────────────────────
Parse a basicsr training log and plot PSNR over all validation checkpoints.

Usage (run on GPU machine from repo root):
  python scripts/plot_psnr_curve.py \
      --log_dir experiments/train_Retinexformer_LOL_v1_fixed \
      --out     results/A0_fixed_psnr_curve.png \
      --title   "A0_fixed (Retinexformer, clip=1.0)"

Or point directly at a log file:
  python scripts/plot_psnr_curve.py \
      --log_file experiments/train_Retinexformer_LOL_v1_fixed/train_*.log \
      --out      results/A0_fixed_psnr_curve.png
"""
import sys
import os
import re
import argparse
import math
from glob import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── LR schedule (from config) ─────────────────────────────────────────── #

def cosine_cyclic_lr(t, periods, rw, eta_mins, base_lr):
    cumulative = [sum(periods[:i+1]) for i in range(len(periods))]
    for i, p in enumerate(cumulative):
        if t <= p:
            nearest = 0 if i == 0 else cumulative[i-1]
            progress = (t - nearest) / periods[i]
            return (eta_mins[i] + rw[i] * 0.5 * (base_lr - eta_mins[i])
                    * (1 + math.cos(math.pi * progress)))
    return eta_mins[-1]


# ── Log parser ────────────────────────────────────────────────────────── #

# Matches lines like:
#   ... INFO: Validation ValSet [psnr: 23.4567]
#   ... INFO: # psnr: 23.4567
#   ... psnr: 23.4567
VAL_RE  = re.compile(r'psnr[:\s]+([0-9]+\.[0-9]+)', re.IGNORECASE)
ITER_RE = re.compile(r'(?:iter|iteration)[:\s]*([0-9]+)', re.IGNORECASE)


def parse_log(path: str):
    """Return list of (iter, psnr) from one log file."""
    records = []
    with open(path) as f:
        for line in f:
            if 'psnr' not in line.lower():
                continue
            m_psnr = VAL_RE.search(line)
            if not m_psnr:
                continue
            psnr = float(m_psnr.group(1))
            # Try to get iter from same line
            m_iter = ITER_RE.search(line)
            it = int(m_iter.group(1)) if m_iter else None
            records.append((it, psnr))
    return records


def collect_logs(log_dir=None, log_file=None):
    paths = []
    if log_file:
        paths = sorted(glob(log_file))
    elif log_dir:
        paths = sorted(glob(os.path.join(log_dir, '*.log')))
    if not paths:
        raise FileNotFoundError(f'No log files found in {log_dir or log_file}')
    all_records = []
    for p in paths:
        all_records.extend(parse_log(p))
    return all_records


# ── CLI ───────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--log_dir',  help='Experiment directory (scans *.log inside)')
    g.add_argument('--log_file', help='Path or glob to a specific log file')
    p.add_argument('--out',   required=True, help='Output PNG path')
    p.add_argument('--title', default='PSNR convergence', help='Plot title')
    p.add_argument('--ref_psnr', type=float, default=25.16,
                   help='Official reference PSNR for comparison line (default 25.16)')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────── #

def main():
    args = parse_args()
    records = collect_logs(args.log_dir, args.log_file)

    if not records:
        print('ERROR: no PSNR entries found in logs.')
        return 1

    # Sort by iter (None iters go to front)
    records = [(it if it is not None else i * 1000, p)
               for i, (it, p) in enumerate(records)]
    records.sort(key=lambda x: x[0])

    iters = [r[0] for r in records]
    psnrs = [r[1] for r in records]

    print(f'Found {len(records)} PSNR validation points.')
    print(f'First: iter={iters[0]}, PSNR={psnrs[0]:.4f}')
    print(f'Last:  iter={iters[-1]}, PSNR={psnrs[-1]:.4f}')
    print(f'Best:  iter={iters[np.argmax(psnrs)]}, PSNR={max(psnrs):.4f}')
    last10 = psnrs[-min(10, len(psnrs)):]
    print(f'Last {len(last10)} PSNR mean={np.mean(last10):.4f}  std=±{np.std(last10):.4f}')

    # LR curve (A0 default schedule)
    periods  = [46000, 104000]
    rw       = [1, 1]
    eta_mins = [0.0003, 0.000001]
    base_lr  = 2e-4
    lr_curve = [cosine_cyclic_lr(t, periods, rw, eta_mins, base_lr) for t in iters]

    # ── Plot ──────────────────────────────────────────────────────────── #
    iters_k = [t / 1000 for t in iters]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(args.title, fontsize=13, fontweight='bold')

    # PSNR
    ax1.plot(iters_k, psnrs, 'o-', color='#e05c3a', linewidth=1.3,
             markersize=3.5, label='PSNR per checkpoint')
    ax1.axhline(args.ref_psnr, color='#2ca02c', linestyle='--', linewidth=1.5,
                label=f'Official reference: {args.ref_psnr} dB')

    best_iter = iters[np.argmax(psnrs)] / 1000
    best_psnr = max(psnrs)
    ax1.scatter([best_iter], [best_psnr], color='gold', zorder=6, s=80,
                edgecolors='black', linewidths=0.8, label=f'Best: {best_psnr:.2f} dB @ {best_iter:.0f}k')

    ax1.axvspan(0, 46, alpha=0.07, color='blue',   label='Phase 1: LR warmup (2e-4→3e-4)')
    ax1.axvspan(46, max(iters_k), alpha=0.07, color='orange',
                label='Phase 2: cosine anneal (2e-4→1e-6)')
    ax1.axvline(46, color='gray', linestyle='--', linewidth=0.9, alpha=0.6)

    ax1.annotate(f'Last {len(last10)} ckpts\n'
                 f'mean={np.mean(last10):.2f}  std=±{np.std(last10):.2f} dB',
                 xy=(iters_k[-1], np.mean(last10)),
                 xytext=(max(0, iters_k[-1] - 40), np.mean(last10) + 0.5),
                 fontsize=8.5,
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3e0',
                           edgecolor='orange', alpha=0.9))

    ax1.set_ylabel('PSNR (dB)', fontsize=11)
    ax1.legend(fontsize=8.5, loc='lower right')
    ax1.grid(True, alpha=0.3)

    # LR
    ax2.semilogy(iters_k, lr_curve, 'b-', linewidth=1.8)
    ax2.axvspan(0, 46, alpha=0.07, color='blue')
    ax2.axvspan(46, max(iters_k), alpha=0.07, color='orange')
    ax2.axvline(46, color='gray', linestyle='--', linewidth=0.9, alpha=0.6)
    ax2.scatter([46], [3e-4], color='red', zorder=5, s=50,
                label='Warmup peak: 3e-4 at iter 46k')
    ax2.set_xlabel('Iteration (×1000)', fontsize=11)
    ax2.set_ylabel('LR (log scale)', fontsize=11)
    ax2.legend(fontsize=8.5)
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'\n[Saved] {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
