"""
Ablation evaluation: A0 (Retinexformer) vs A1 (FD2RT_V1) on LOL-v1 test set.

Runs both models on the 15-image LOL-v1 test set, computes per-image and mean
PSNR / SSIM, writes results/ablation_log.json, and applies the PSNR decision gate:

  A1 > A0 + 0.2 dB  →  GATE PASSED — proceed to Phase 2
  A0 < A1 ≤ A0+0.2  →  MARGINAL — review visual diagnostics
  A1 ≤ A0           →  GATE FAILED — diagnose before Phase 2

Usage (GPU machine, from /workspace/fd2rt/Retinexformer):
    python scripts/evaluate_ablation.py \
        --a0_weights pretrained_weights/LOL_v1.pth \
        --a1_weights experiments/FD2RT_V1_LOL_v1/models/net_g_best.pth \
        --data_root  data/LOLv1/Test \
        --out_dir    results

CPU debug (tiny synthetic set):
    python scripts/evaluate_ablation.py --cpu --data_root data/LOLv1/Test \
        --a0_weights pretrained_weights/LOL_v1.pth \
        --a1_weights experiments/FD2RT_V1_LOL_v1/models/net_g_best.pth
"""

import sys
import os
import json
import argparse
import time
from glob import glob
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import img_as_ubyte
from natsort import natsorted

# basicsr must be importable for arch auto-registration
import basicsr  # noqa: F401
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
from basicsr.models.archs.fd2rt_v1_arch import FD2RT_V1

sys.path.insert(0, os.path.join(_ROOT, 'Enhancement'))
import utils as enh_utils  # PSNR / calculate_ssim / load_img / save_img


# ── CLI ────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--a0_weights', required=True,
                   help='Path to Retinexformer LOL-v1 checkpoint (A0 baseline)')
    p.add_argument('--a1_weights', required=True,
                   help='Path to FD2RT_V1 checkpoint (A1)')
    p.add_argument('--data_root', default='data/LOLv1/Test',
                   help='Root of LOL-v1 test set (contains input/ and target/)')
    p.add_argument('--out_dir', default='results',
                   help='Output directory for ablation_log.json and saved images')
    p.add_argument('--save_images', action='store_true',
                   help='Save enhanced images alongside the log (adds disk I/O)')
    p.add_argument('--cpu', action='store_true',
                   help='Force CPU even if CUDA is available')
    p.add_argument('--factor', type=int, default=4,
                   help='Padding factor (must be divisible); default 4')
    return p.parse_args()


# ── Model helpers ─────────────────────────────────────────────────────── #

LOL_V1_KWARGS = dict(in_channels=3, out_channels=3, n_feat=40,
                     stage=1, num_blocks=[1, 2, 2])


def _load_state(path, model, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get('params', ckpt.get('state_dict', ckpt))
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # some checkpoints are saved with module. prefix from DataParallel
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(new_state)
    return model


def build_a0(weights_path, device):
    model = RetinexFormer(**LOL_V1_KWARGS)
    model = _load_state(weights_path, model, device)
    return model.to(device).eval()


def build_a1(weights_path, device):
    model = FD2RT_V1(**LOL_V1_KWARGS)
    model = _load_state(weights_path, model, device)
    return model.to(device).eval()


# ── Inference on a single image ────────────────────────────────────────── #

def infer(model, img_np_float32, device, factor):
    """
    img_np_float32: HxWx3 float32 in [0,1] (RGB).
    Returns: HxWx3 float32 in [0,1].
    """
    t = torch.from_numpy(img_np_float32).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, h, w = t.shape
    padh = (-h % factor)
    padw = (-w % factor)
    if padh or padw:
        t = F.pad(t, (0, padw, 0, padh), mode='reflect')
    with torch.inference_mode():
        out = model(t)
    out = out[:, :, :h, :w]
    out = torch.clamp(out, 0, 1).cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    return out.astype(np.float32)


# ── Per-image metrics ─────────────────────────────────────────────────── #

def compute_metrics(pred_f32, gt_f32):
    """Both arrays: HxWx3 float32 in [0,1]."""
    psnr_val = enh_utils.PSNR(gt_f32, pred_f32)
    ssim_val = enh_utils.calculate_ssim(img_as_ubyte(gt_f32), img_as_ubyte(pred_f32))
    return float(psnr_val), float(ssim_val)


# ── Main ──────────────────────────────────────────────────────────────── #

def main():
    args = parse_args()

    device = torch.device(
        'cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda'
    )
    print(f"\nDevice : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Data paths ────────────────────────────────────────────────────── #
    lq_dir = os.path.join(args.data_root, 'input')
    gt_dir = os.path.join(args.data_root, 'target')
    lq_paths = natsorted(glob(os.path.join(lq_dir, '*.png'))
                         + glob(os.path.join(lq_dir, '*.jpg')))
    gt_paths  = natsorted(glob(os.path.join(gt_dir, '*.png'))
                          + glob(os.path.join(gt_dir, '*.jpg')))
    assert len(lq_paths) > 0, f"No images found in {lq_dir}"
    assert len(lq_paths) == len(gt_paths), \
        f"Input/target count mismatch: {len(lq_paths)} vs {len(gt_paths)}"
    n = len(lq_paths)
    print(f"Test images: {n}")

    # ── Build models ──────────────────────────────────────────────────── #
    print("\nLoading A0 (Retinexformer) …")
    a0 = build_a0(args.a0_weights, device)
    print(f"  Params: {sum(p.numel() for p in a0.parameters()):,}")

    print("Loading A1 (FD2RT_V1) …")
    a1 = build_a1(args.a1_weights, device)
    print(f"  Params: {sum(p.numel() for p in a1.parameters()):,}")

    # ── Output dirs ───────────────────────────────────────────────────── #
    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.out_dir, 'A0_enhanced'), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, 'A1_enhanced'), exist_ok=True)

    # ── Evaluation loop ───────────────────────────────────────────────── #
    per_image = []
    print(f"\n{'Image':>15}  {'A0 PSNR':>8}  {'A0 SSIM':>8}  "
          f"{'A1 PSNR':>8}  {'A1 SSIM':>8}  {'ΔPSNR':>7}")
    print("-" * 65)

    t0 = time.perf_counter()
    for lq_p, gt_p in zip(lq_paths, gt_paths):
        name = os.path.splitext(os.path.basename(lq_p))[0]
        lq = np.float32(enh_utils.load_img(lq_p)) / 255.
        gt = np.float32(enh_utils.load_img(gt_p)) / 255.

        pred_a0 = infer(a0, lq, device, args.factor)
        pred_a1 = infer(a1, lq, device, args.factor)

        p0, s0 = compute_metrics(pred_a0, gt)
        p1, s1 = compute_metrics(pred_a1, gt)

        per_image.append({
            'image': name,
            'A0': {'psnr': round(p0, 4), 'ssim': round(s0, 4)},
            'A1': {'psnr': round(p1, 4), 'ssim': round(s1, 4)},
            'delta_psnr': round(p1 - p0, 4),
        })
        print(f"{name:>15}  {p0:>8.4f}  {s0:>8.4f}  "
              f"{p1:>8.4f}  {s1:>8.4f}  {p1-p0:>+7.4f}")

        if args.save_images:
            enh_utils.save_img(
                os.path.join(args.out_dir, 'A0_enhanced', name + '.png'),
                img_as_ubyte(pred_a0))
            enh_utils.save_img(
                os.path.join(args.out_dir, 'A1_enhanced', name + '.png'),
                img_as_ubyte(pred_a1))

    elapsed = time.perf_counter() - t0

    # ── Aggregate ─────────────────────────────────────────────────────── #
    mean_a0_psnr = float(np.mean([r['A0']['psnr'] for r in per_image]))
    mean_a0_ssim = float(np.mean([r['A0']['ssim'] for r in per_image]))
    mean_a1_psnr = float(np.mean([r['A1']['psnr'] for r in per_image]))
    mean_a1_ssim = float(np.mean([r['A1']['ssim'] for r in per_image]))
    delta_psnr   = mean_a1_psnr - mean_a0_psnr

    # ── Decision gate ─────────────────────────────────────────────────── #
    THRESHOLD_PASS     = 0.20   # dB above A0
    THRESHOLD_MARGINAL = 0.0    # dB: above A0 but below PASS

    if delta_psnr > THRESHOLD_PASS:
        gate = 'GATE PASSED'
        verdict = f'GATE PASSED — proceed to Phase 2  (Δ={delta_psnr:+.4f} dB)'
    elif delta_psnr > THRESHOLD_MARGINAL:
        gate = 'MARGINAL'
        verdict = f'MARGINAL — review visual diagnostics  (Δ={delta_psnr:+.4f} dB)'
    else:
        gate = 'GATE FAILED'
        verdict = f'GATE FAILED — diagnose before Phase 2  (Δ={delta_psnr:+.4f} dB)'

    # ── Print summary ─────────────────────────────────────────────────── #
    print()
    print("=" * 65)
    print("ABLATION SUMMARY")
    print("=" * 65)
    print(f"  A0 (Retinexformer)  PSNR: {mean_a0_psnr:.4f} dB  SSIM: {mean_a0_ssim:.4f}")
    print(f"  A1 (FD2RT_V1)       PSNR: {mean_a1_psnr:.4f} dB  SSIM: {mean_a1_ssim:.4f}")
    print(f"  Δ PSNR (A1−A0)    : {delta_psnr:+.4f} dB")
    print()
    print(f"  DECISION GATE: {verdict}")
    print(f"\n  Wall time: {elapsed:.1f}s for {n} images "
          f"({elapsed/n:.1f}s/image) on {device}")
    print("=" * 65)

    # ── Write ablation_log.json ────────────────────────────────────────── #
    log = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'device': str(device),
        'n_images': n,
        'a0_weights': args.a0_weights,
        'a1_weights': args.a1_weights,
        'summary': {
            'A0': {'mean_psnr': round(mean_a0_psnr, 4),
                   'mean_ssim': round(mean_a0_ssim, 4)},
            'A1': {'mean_psnr': round(mean_a1_psnr, 4),
                   'mean_ssim': round(mean_a1_ssim, 4)},
            'delta_psnr': round(delta_psnr, 4),
            'gate': gate,
        },
        'per_image': per_image,
    }
    log_path = os.path.join(args.out_dir, 'ablation_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nLog written → {log_path}")

    return 0 if gate != 'GATE FAILED' else 1


if __name__ == '__main__':
    sys.exit(main())
