"""
TASK 3 — Training instability diagnostics from basicsr log file.

Parses the A4 (or any model) training log, extracts l_pix per iteration,
and produces:
  1. Spike count: iterations where l_pix > 2× running median
  2. Spike periodicity: checks if spikes cluster at specific offsets
     modulo epoch length (default 61 = ceil(485/8))
  3. Tail stats: mean l_pix in the last 10K / 50K iterations
  4. loss_curve_a4.png: full training loss curve with spikes highlighted

Usage (GPU machine):
    python scripts/analyze_training_log.py \
        --log_file  experiments/FD2RT_A4_LOL_v1/train_FD2RT_A4_LOL_v1_*.log \
        --out_dir   results/diag \
        --epoch_len 61

The log_file argument also accepts glob patterns. If multiple files are
found they are concatenated in chronological order (useful if training
was resumed and a new log file was created).
"""

import sys, os, re, json, argparse
from glob import glob
from datetime import datetime, timezone
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))


# ── Parser ─────────────────────────────────────────────────────────────── #

# Example log line (basicsr default):
# 2024-05-01 12:34:56,789 INFO: [FD2RT_A4..][epoch:  0, iter:    500, lr:(2.000e-04,)] ... l_pix: 4.5681e-02
_LOG_RE = re.compile(
    r'iter:\s*(\d+).*?lr:\(([\d.e+\-]+)'
    r'.*?l_pix:\s*([\d.e+\-]+)'
)
_VAL_RE = re.compile(
    r'iter:\s*(\d+).*?psnr:\s*([\d.]+)'
)


def parse_log(log_path):
    """Returns (train_rows, val_rows) where each row is a dict."""
    train_rows, val_rows = [], []
    with open(log_path) as f:
        for line in f:
            m = _LOG_RE.search(line)
            if m:
                train_rows.append({
                    'iter': int(m.group(1)),
                    'lr':   float(m.group(2)),
                    'l_pix': float(m.group(3)),
                })
                continue
            m = _VAL_RE.search(line)
            if m:
                val_rows.append({
                    'iter': int(m.group(1)),
                    'psnr': float(m.group(2)),
                })
    return train_rows, val_rows


def running_median(values, window=200):
    """Fast approximate running median using a sliding window."""
    out = np.empty(len(values))
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = np.median(values[lo:hi])
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--log_file',  required=True,
                   help='Path to .log file (glob ok, e.g. experiments/*/train*.log)')
    p.add_argument('--out_dir',   default='results/diag')
    p.add_argument('--epoch_len', type=int, default=61,
                   help='Iterations per epoch = ceil(n_train / batch_size)')
    p.add_argument('--spike_factor', type=float, default=2.0,
                   help='A loss is a spike if l_pix > factor × running_median')
    p.add_argument('--window', type=int, default=200,
                   help='Window for running-median computation')
    p.add_argument('--no_plot', action='store_true',
                   help='Skip matplotlib plot (for headless envs without display)')
    return p.parse_args()


def main():
    args = parse_args()

    # Collect log files
    paths = sorted(glob(args.log_file))
    if not paths:
        print(f'ERROR: no files match {args.log_file}')
        return 1
    print(f'\nLog files ({len(paths)}):')
    for p in paths:
        print(f'  {p}')

    all_train, all_val = [], []
    for p in paths:
        tr, va = parse_log(p)
        all_train.extend(tr)
        all_val.extend(va)

    # Deduplicate and sort
    seen_i = set()
    train_rows = []
    for r in sorted(all_train, key=lambda x: x['iter']):
        if r['iter'] not in seen_i:
            seen_i.add(r['iter'])
            train_rows.append(r)

    val_rows = sorted(all_val, key=lambda x: x['iter'])

    if not train_rows:
        print('ERROR: no training rows found. Check log format.')
        return 1

    iters  = np.array([r['iter']  for r in train_rows])
    l_pix  = np.array([r['l_pix'] for r in train_rows])
    lrs    = np.array([r['lr']    for r in train_rows])

    print(f'\nTraining rows parsed: {len(train_rows)}')
    print(f'Iter range: {iters[0]} → {iters[-1]}')
    print(f'Validation points: {len(val_rows)}')

    # ── Task 3.1: Gradient clipping (from code / config) ──────────────── #
    print('\n' + '=' * 60)
    print('TASK 3.1 — Gradient clipping')
    print('=' * 60)
    print('  use_grad_clip : True  (from Options/train_FD2RT_A4_LOL_v1.yml)')
    print('  clip value    : 0.01  (hardcoded in basicsr/models/'
          'image_restoration_model.py:194)')
    print('  AMP enabled   : True  (amp_scaler active)')
    print()
    print('  NOTE: clip_norm=0.01 is ~50-100× more aggressive than standard')
    print('        practice (PyTorch default examples use 0.5–5.0). At this')
    print('        value, almost every gradient step is clipped, making the')
    print('        effective step size nearly constant and independent of')
    print('        gradient magnitude. This decouples LR scheduling from')
    print('        actual parameter updates — a likely contributor to the')
    print('        observed oscillation persisting into near-zero LR.')

    # ── Task 3.2: Spike analysis ───────────────────────────────────────── #
    print('\n' + '=' * 60)
    print('TASK 3.2 — Loss spike analysis')
    print('=' * 60)

    rm = running_median(l_pix, window=args.window)
    spike_thresh = args.spike_factor * rm
    spike_mask   = l_pix > spike_thresh
    spike_iters  = iters[spike_mask]
    spike_values = l_pix[spike_mask]
    n_spikes     = int(spike_mask.sum())
    spike_frac   = n_spikes / len(l_pix)

    print(f'  Total iterations   : {len(l_pix)}')
    print(f'  Spike threshold    : l_pix > {args.spike_factor}× running_median')
    print(f'  Spikes detected    : {n_spikes}  ({spike_frac*100:.1f}%)')
    if n_spikes > 0:
        print(f'  Spike l_pix range  : [{spike_values.min():.4f}, '
              f'{spike_values.max():.4f}]  (median={np.median(spike_values):.4f})')

    # Late-training stats (last 50K iters)
    late_mask  = iters >= (iters[-1] - 50_000)
    late_l_pix = l_pix[late_mask]
    late_iters = iters[late_mask]
    late_spikes = spike_mask[late_mask]
    print(f'\n  Last 50K iters:')
    print(f'    l_pix mean ± std : {late_l_pix.mean():.5f} ± {late_l_pix.std():.5f}')
    print(f'    l_pix min / max  : {late_l_pix.min():.5f} / {late_l_pix.max():.5f}')
    print(f'    Spikes           : {late_spikes.sum()}  '
          f'({late_spikes.mean()*100:.1f}% of late iters)')

    # LR at last logged iter
    print(f'\n  LR at last iteration: {lrs[-1]:.3e}')

    # ── Task 3.3: Epoch-length periodicity ────────────────────────────── #
    print('\n' + '=' * 60)
    print(f'TASK 3.3 — Spike periodicity  (epoch_len = {args.epoch_len} iters)')
    print('=' * 60)
    print(f'  Dataset: 485 training images  |  batch 8  |  '
          f'epoch ≈ ceil(485/8) = {args.epoch_len} iters')

    if n_spikes > 0:
        offsets = (spike_iters % args.epoch_len).tolist()
        counter = Counter(offsets)
        top5    = counter.most_common(5)
        print(f'\n  Top-5 most common offset (iter %% {args.epoch_len}) among spikes:')
        for off, cnt in top5:
            pct = cnt / n_spikes * 100
            print(f'    offset {off:>3d}  →  {cnt} spikes  ({pct:.1f}% of all spikes)')

        # Is the most common offset >> random?
        expected_per_offset = n_spikes / args.epoch_len
        max_cnt = top5[0][1]
        ratio   = max_cnt / (expected_per_offset + 1e-8)
        if ratio > 3:
            print(f'\n  *** PATTERN DETECTED: offset {top5[0][0]} appears '
                  f'{ratio:.1f}× above random expectation.')
            print(f'      A specific training image pair is likely causing '
                  f'periodic loss spikes.')
        else:
            print(f'\n  No strong periodic pattern detected '
                  f'(max ratio vs random: {ratio:.1f}×).')
            print(f'  Spikes are likely stochastic (AMP scale events or '
                  f'hard random crops), not a corrupt image pair.')
    else:
        print('  No spikes to analyse.')

    # ── Task 3.4: Validation oscillation ──────────────────────────────── #
    print('\n' + '=' * 60)
    print('TASK 3.4 — Validation PSNR oscillation')
    print('=' * 60)
    if val_rows:
        val_iters = np.array([r['iter'] for r in val_rows])
        val_psnr  = np.array([r['psnr'] for r in val_rows])

        # Last 50 val points
        last50 = val_psnr[-50:]
        late_val_iters = val_iters[-50:]
        print(f'  Total val points  : {len(val_psnr)}')
        print(f'  Val PSNR all-time : {val_psnr.max():.4f} (best)  '
              f'{val_psnr.mean():.4f} (mean)  {val_psnr.std():.4f} (std)')
        print(f'  Last 50 val pts   : '
              f'mean={last50.mean():.4f}  std={last50.std():.4f}  '
              f'range=[{last50.min():.4f}, {last50.max():.4f}]')
        print(f'  Oscillation band  : {last50.max()-last50.min():.4f} dB '
              f'(target: <0.10 dB for reliable comparison)')

        # LR at those late val points
        late_lr_idx = np.searchsorted(iters, late_val_iters)
        late_lr_idx = np.clip(late_lr_idx, 0, len(lrs)-1)
        late_lrs = lrs[late_lr_idx]
        print(f'  LR at late val    : {late_lrs.min():.2e} – {late_lrs.max():.2e}')
    else:
        print('  No validation rows found in log.')

    # ── Plot ──────────────────────────────────────────────────────────── #
    os.makedirs(args.out_dir, exist_ok=True)
    plot_path = os.path.join(args.out_dir, 'loss_curve_a4.png')

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                     gridspec_kw={'height_ratios': [3, 1, 1.5]})

            # Panel 1: l_pix full training
            ax = axes[0]
            ax.semilogy(iters, l_pix, color='steelblue', lw=0.4,
                        alpha=0.6, label='l_pix')
            ax.semilogy(iters, rm,    color='orange',    lw=1.5,
                        label=f'running median (w={args.window})')
            ax.semilogy(iters[spike_mask], l_pix[spike_mask],
                        'r.', ms=3, alpha=0.7,
                        label=f'spikes (>{args.spike_factor}× median), n={n_spikes}')
            ax.set_ylabel('l_pix (log scale)')
            ax.set_title('A4 Training Loss Curve')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Panel 2: Learning rate
            ax2 = axes[1]
            ax2.semilogy(iters, lrs, color='purple', lw=0.8)
            ax2.set_ylabel('LR')
            ax2.grid(True, alpha=0.3)

            # Panel 3: Validation PSNR
            ax3 = axes[2]
            if val_rows:
                ax3.plot(val_iters, val_psnr, color='green', lw=0.8,
                         marker='.', ms=2)
                ax3.axhline(val_psnr[-50:].mean(), color='red', lw=1,
                            ls='--', label=f'last-50 mean {val_psnr[-50:].mean():.2f}')
                ax3.set_ylabel('Val PSNR (dB)')
                ax3.legend(fontsize=8)
                ax3.grid(True, alpha=0.3)
            ax3.set_xlabel('Iteration')

            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f'\nPlot saved → {plot_path}')
        except ImportError:
            print('\nmatplotlib not available; skipping plot (run with --no_plot to suppress this)')

    # ── Save JSON summary ─────────────────────────────────────────────── #
    summary = {
        'grad_clip_value': 0.01,
        'amp_enabled': True,
        'n_train_rows': len(train_rows),
        'iter_range': [int(iters[0]), int(iters[-1])],
        'n_val_rows': len(val_rows),
        'spike_analysis': {
            'factor': args.spike_factor,
            'n_spikes': n_spikes,
            'spike_fraction_pct': round(spike_frac * 100, 2),
            'late_50k_mean': round(float(late_l_pix.mean()), 6),
            'late_50k_std':  round(float(late_l_pix.std()),  6),
        },
        'val_psnr': {
            'best':     round(float(val_psnr.max()), 4),
            'mean_all': round(float(val_psnr.mean()), 4),
            'std_all':  round(float(val_psnr.std()),  4),
            'last50_mean': round(float(val_psnr[-50:].mean()), 4),
            'last50_std':  round(float(val_psnr[-50:].std()),  4),
            'last50_range': round(float(val_psnr[-50:].max()-val_psnr[-50:].min()), 4),
        } if val_rows else {},
        'epoch_len': args.epoch_len,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    json_path = os.path.join(args.out_dir, 'training_diag.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'JSON saved → {json_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
