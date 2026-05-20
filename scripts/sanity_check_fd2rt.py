"""
500-iteration overfit-single-batch sanity check for FD2RT_V1.

Tests:
  1. Model builds and forward pass runs without error.
  2. L1 loss decreases by at least 30% in 500 gradient steps.
  3. Gradient norms are finite throughout (no NaN / inf).

Usage (GPU machine):
    cd /workspace/fd2rt/Retinexformer
    python scripts/sanity_check_fd2rt.py

Usage (CPU debug):
    python scripts/sanity_check_fd2rt.py --cpu

Design:
  A single fixed synthetic batch [B, 3, 64, 64] is created once and
  reused every iteration — this isolates training dynamics from data
  loading and is the canonical way to check that a model can overfit.
  64×64 patches keep CPU time manageable (~0.4 s/iter → ~4 min total).
"""

import sys
import os
import argparse
import time

# Repo root on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Must import basicsr so the arch registry is populated
from basicsr.models.archs.fd2rt_v1_arch import FD2RT_V1


# ── CLI ──────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cpu', action='store_true',
                   help='Force CPU even if CUDA is available')
    p.add_argument('--iters', type=int, default=500,
                   help='Number of gradient steps (default: 500)')
    p.add_argument('--batch', type=int, default=4,
                   help='Batch size (default: 4)')
    p.add_argument('--patch', type=int, default=64,
                   help='Spatial patch size (default: 64)')
    p.add_argument('--lr', type=float, default=2e-4,
                   help='Learning rate (default: 2e-4, matches training config)')
    p.add_argument('--print-every', type=int, default=50,
                   help='Print loss every N iters (default: 50)')
    p.add_argument('--decrease-threshold', type=float, default=0.30,
                   help='Required fractional loss decrease (default: 0.30 = 30%%)')
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────── #

def main():
    args = parse_args()

    device = torch.device(
        'cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda'
    )
    print(f"\nDevice : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Build model ────────────────────────────────────────────────────  #
    model = FD2RT_V1(
        in_channels=3,
        out_channels=3,
        n_feat=40,
        stage=1,
        num_blocks=[1, 2, 2],
    ).to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params : {n_params:,}")

    # ── Optimizer (identical to training config) ────────────────────── #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999))

    # ── Fixed synthetic batch ──────────────────────────────────────── #
    # lq: low-light  (dark, gamma-compressed)
    # gt: normal-light (the target we want to reproduce)
    torch.manual_seed(42)
    lq = (torch.rand(args.batch, 3, args.patch, args.patch) * 0.3).to(device)
    gt = (torch.rand(args.batch, 3, args.patch, args.patch) * 0.7 + 0.3).to(device)
    # lq ∈ [0, 0.3]  (dark),  gt ∈ [0.3, 1.0]  (bright)

    print(f"\nBatch  : [{args.batch}, 3, {args.patch}, {args.patch}]")
    print(f"lq range: [{lq.min():.3f}, {lq.max():.3f}]")
    print(f"gt range: [{gt.min():.3f}, {gt.max():.3f}]")
    print(f"Iters  : {args.iters}")
    print(f"LR     : {args.lr}")
    print(f"Decrease threshold: {args.decrease_threshold*100:.0f}%")
    print()
    print(f"{'Iter':>6}  {'Loss':>10}  {'Grad norm':>12}  {'Elapsed':>10}")
    print("-" * 48)

    losses = []
    grad_norms = []
    t_start = time.perf_counter()

    for step in range(1, args.iters + 1):
        optimizer.zero_grad()

        pred = model(lq)                        # [B, 3, H, W]
        loss = F.l1_loss(pred, gt)              # MAE — matches training config
        loss.backward()

        # Gradient clipping (matches image_restoration_model.py line 194)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        # Check for NaN / Inf in gradients
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5
        grad_norms.append(grad_norm)

        if step % args.print_every == 0 or step == 1:
            elapsed = time.perf_counter() - t_start
            print(f"{step:>6}  {loss_val:>10.6f}  {grad_norm:>12.6f}  {elapsed:>8.1f}s")

    elapsed_total = time.perf_counter() - t_start

    # ── Results ───────────────────────────────────────────────────────  #
    loss_init  = losses[0]
    loss_final = losses[-1]
    decrease   = (loss_init - loss_final) / loss_init
    passed     = decrease >= args.decrease_threshold

    any_nan    = any(not (g == g) for g in grad_norms)   # NaN check
    any_inf    = any(g == float('inf') for g in grad_norms)

    print()
    print("=" * 60)
    print("SANITY CHECK RESULTS")
    print("=" * 60)
    print(f"  Initial loss  : {loss_init:.6f}")
    print(f"  Final loss    : {loss_final:.6f}")
    print(f"  Decrease      : {decrease*100:.1f}%   "
          f"(threshold: {args.decrease_threshold*100:.0f}%)")
    print(f"  Gradient NaN  : {'YES ⚠' if any_nan else 'no'}")
    print(f"  Gradient Inf  : {'YES ⚠' if any_inf else 'no'}")
    print(f"  Wall time     : {elapsed_total:.1f}s  "
          f"({elapsed_total/args.iters*1000:.1f} ms/iter)")
    print()

    if not passed:
        print("  RESULT: *** FAIL — loss did not decrease by 30% ***")
        print("  Diagnose:")
        print("    1. Check lq/gt are in [0, 1]")
        print("    2. Check for dead ReLUs in nmap_branch (all-zero N_map)")
        print("    3. Print pred.min/max — if stuck at 0, check illu_map scale")
        print("    4. Try a lower LR (1e-4) or remove grad clipping")
    elif any_nan or any_inf:
        print("  RESULT: *** FAIL — NaN/Inf gradients detected ***")
        print("  Diagnose: possible exploding gradients; check clip_grad_norm value")
    else:
        print(f"  RESULT: *** PASS *** — {decrease*100:.1f}% loss decrease, "
              f"gradients finite throughout")

    print("=" * 60)
    return 0 if (passed and not any_nan and not any_inf) else 1


if __name__ == '__main__':
    sys.exit(main())
