"""
Visual diagnostics: Retinexformer vs FD2RT_V1 on a single LOL-v1 test image.

Saves three side-by-side comparison images to results/:

  illumination_prior_comparison.png
      Left:  mean_c(I)  — Retinexformer's pixel-domain illumination prior
      Right: LL_mean    — our wavelet LL-subband illumination prior

  lit_up_comparison.png
      Left:  I_lu(A0) = I * illu_map(A0) + I
      Right: I_lu(A1) = I * illu_map(A1) + I

  output_comparison.png
      Left:  A0 output  (Retinexformer)
      Center: A1 output (FD2RT_V1)
      Right:  GT

Usage (GPU machine, from /workspace/fd2rt/Retinexformer):
    python scripts/visual_diagnostics.py \
        --a0_weights pretrained_weights/LOL_v1.pth \
        --a1_weights experiments/FD2RT_V1_LOL_v1/models/net_g_best.pth \
        --lq_path    data/LOLv1/Test/input/00001.png \
        --gt_path    data/LOLv1/Test/target/00001.png \
        --out_dir    results

CPU debug (no GPU needed):
    python scripts/visual_diagnostics.py --cpu \
        --a0_weights pretrained_weights/LOL_v1.pth \
        --a1_weights experiments/FD2RT_V1_LOL_v1/models/net_g_best.pth \
        --lq_path    data/LOLv1/Test/input/00001.png \
        --gt_path    data/LOLv1/Test/target/00001.png
"""

import sys
import os
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import basicsr  # noqa: F401 — triggers arch auto-registration
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
from basicsr.models.archs.fd2rt_v1_arch import FD2RT_V1

from wavelet_utils import HaarDWT2D

sys.path.insert(0, os.path.join(_ROOT, 'Enhancement'))
import utils as enh_utils


# ── CLI ────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--a0_weights', required=True)
    p.add_argument('--a1_weights', required=True)
    p.add_argument('--lq_path', default='data/LOLv1/Test/input/00001.png',
                   help='Low-light input image path')
    p.add_argument('--gt_path', default='data/LOLv1/Test/target/00001.png',
                   help='Ground-truth target image path')
    p.add_argument('--out_dir', default='results')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--factor', type=int, default=4)
    return p.parse_args()


# ── Shared helpers ─────────────────────────────────────────────────────── #

LOL_V1_KWARGS = dict(in_channels=3, out_channels=3, n_feat=40,
                     stage=1, num_blocks=[1, 2, 2])


def _load_state(path, model, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get('params', ckpt.get('state_dict', ckpt))
    try:
        model.load_state_dict(state)
    except RuntimeError:
        new_state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(new_state)
    return model


def load_lq_tensor(path, device, factor):
    """Returns (img_np [H,W,3] float32), (tensor [1,3,H,W]), h, w, padh, padw."""
    img_np = np.float32(enh_utils.load_img(path)) / 255.
    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, h, w = t.shape
    padh = (-h % factor)
    padw = (-w % factor)
    if padh or padw:
        t_pad = F.pad(t, (0, padw, 0, padh), mode='reflect')
    else:
        t_pad = t
    return img_np, t_pad, h, w, padh, padw


def infer_and_unpad(model, t_pad, h, w):
    with torch.inference_mode():
        out = model(t_pad)
    out = out[:, :, :h, :w]
    return torch.clamp(out, 0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)


def _stats(arr, name):
    print(f"  {name:40s}  min={arr.min():.4f}  max={arr.max():.4f}  "
          f"mean={arr.mean():.4f}")


def side_by_side(*panels, labels=None, pad=4):
    """
    Stack panels horizontally with a thin white separator.
    panels: list of HxWx3 float32 arrays (values in [0,1]).
    Returns: HxW'x3 float32.
    """
    h = panels[0].shape[0]
    sep = np.ones((h, pad, 3), dtype=np.float32)
    out = []
    for i, p in enumerate(panels):
        out.append(p)
        if i < len(panels) - 1:
            out.append(sep)
    return np.concatenate(out, axis=1)


def add_labels(img_f32, labels, font_scale=0.6, thickness=1):
    """Burn text labels into the top-left of each equally-wide panel."""
    img = (img_f32 * 255).clip(0, 255).astype(np.uint8)
    h, W = img.shape[:2]
    n = len(labels)
    panel_w = W // n
    for i, label in enumerate(labels):
        x = i * panel_w + 6
        cv2.putText(img, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img.astype(np.float32) / 255.


def save_png(path, img_f32):
    img_bgr = cv2.cvtColor((img_f32 * 255).clip(0, 255).astype(np.uint8),
                           cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)
    print(f"  Saved → {path}")


# ── Illumination prior extraction (no-grad hooks) ─────────────────────── #

def extract_a0_priors(model_a0, t_pad, h, w, device):
    """
    Returns illu_map [H,W,3] and I_lu [H,W,3] for Retinexformer.
    Hooks into the single-stage estimator forward.
    """
    stage = model_a0.body[0]     # RetinexFormer_Single_Stage
    estimator = stage.estimator  # Illumination_Estimator

    captured = {}

    def _hook_fwd(module, inp, out):
        # out = (illu_fea, illu_map)
        captured['illu_map'] = out[1].detach()

    h_fwd = estimator.register_forward_hook(_hook_fwd)
    with torch.inference_mode():
        model_a0(t_pad)
    h_fwd.remove()

    illu_map = captured['illu_map'][:, :, :h, :w]      # [1,3,H,W]
    mean_c = t_pad[:, :, :h, :w].mean(dim=1, keepdim=True)  # [1,1,H,W] — the actual prior

    # Re-run to get I_lu (img * illu_map + img) — not stored as attribute
    lq_crop = t_pad[:, :, :h, :w]
    I_lu = (lq_crop * illu_map + lq_crop).clamp(0, 1)

    # mean_c prior (scalar field → broadcast to 3ch for display)
    mean_c_vis = mean_c.expand(-1, 3, -1, -1)[:, :, :h, :w]

    def _to_np(t):
        return t.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)

    return _to_np(mean_c_vis), _to_np(illu_map.clamp(0, 1)), _to_np(I_lu)


def extract_a1_priors(model_a1, t_pad, h, w, device):
    """
    Returns LL_mean [H,W,3] and I_lu [H,W,3] for FD2RT_V1.
    Hooks into the WaveletIlluminationEstimator forward.
    """
    stage = model_a1.body[0]     # FD2RT_Single_Stage
    estimator = stage.estimator  # WaveletIlluminationEstimator

    captured = {}

    def _hook_fwd(module, inp, out):
        # out = (F_lu, I_lu, N_map)
        captured['I_lu']  = out[1].detach()

    h_fwd = estimator.register_forward_hook(_hook_fwd)

    # Compute LL_mean directly from the image using the same DWT
    dwt = HaarDWT2D().to(device)
    lq_crop = t_pad[:, :, :h, :w]
    with torch.inference_mode():
        model_a1(t_pad)
        LL, _, _, _ = dwt(lq_crop)
        LL_up = F.interpolate(LL, size=(h, w), mode='bilinear', align_corners=False)
        LL_mean = LL_up.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)

    h_fwd.remove()

    I_lu = captured['I_lu'][:, :, :h, :w].clamp(0, 1)

    def _to_np(t):
        return t.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)

    return _to_np(LL_mean.clamp(0, 1)), _to_np(I_lu)


# ── Main ──────────────────────────────────────────────────────────────── #

def main():
    args = parse_args()

    device = torch.device(
        'cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda'
    )
    print(f"\nDevice : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load image ────────────────────────────────────────────────────── #
    lq_np, t_pad, h, w, _, _ = load_lq_tensor(args.lq_path, device, args.factor)
    gt_np = np.float32(enh_utils.load_img(args.gt_path)) / 255.
    print(f"\nImage  : {os.path.basename(args.lq_path)}  ({h}×{w})")

    # ── Load models ───────────────────────────────────────────────────── #
    print("\nLoading A0 (Retinexformer) …")
    a0 = _load_state(args.a0_weights, RetinexFormer(**LOL_V1_KWARGS), device).to(device).eval()
    print("Loading A1 (FD2RT_V1) …")
    a1 = _load_state(args.a1_weights, FD2RT_V1(**LOL_V1_KWARGS), device).to(device).eval()

    # ── Outputs ──────────────────────────────────────────────────────── #
    print("\nRunning inference …")
    pred_a0 = infer_and_unpad(a0, t_pad, h, w)
    pred_a1 = infer_and_unpad(a1, t_pad, h, w)

    # ── Illumination priors ───────────────────────────────────────────── #
    print("Extracting illumination priors …")
    mean_c_vis, illu_map_a0, I_lu_a0 = extract_a0_priors(a0, t_pad, h, w, device)
    LL_mean_vis, I_lu_a1              = extract_a1_priors(a1, t_pad, h, w, device)

    # ── Value range report ────────────────────────────────────────────── #
    print("\nIllumination prior value ranges:")
    _stats(mean_c_vis[:, :, 0],  'A0  mean_c(I)   [single channel]')
    _stats(LL_mean_vis[:, :, 0], 'A1  LL_mean     [single channel]')

    # Flag significant divergence
    a0_mean = float(mean_c_vis.mean())
    a1_mean = float(LL_mean_vis.mean())
    if a0_mean > 0:
        rel_diff = abs(a1_mean - a0_mean) / a0_mean
        flag = '  *** >20% DIFFERENCE — FLAGGED ***' if rel_diff > 0.20 else ''
        print(f"\n  Relative mean difference: {rel_diff*100:.1f}%{flag}")

    # ── Figure 1: Illumination prior comparison ────────────────────────  #
    panel = side_by_side(lq_np, mean_c_vis, LL_mean_vis)
    panel = add_labels(panel, ['Input LQ', 'A0: mean_c(I)', 'A1: LL_mean'])
    out_path = os.path.join(args.out_dir, 'illumination_prior_comparison.png')
    save_png(out_path, panel)

    # ── Figure 2: Lit-up image comparison ──────────────────────────────  #
    print("\nLit-up image (I_lu) value ranges:")
    _stats(I_lu_a0, 'A0  I_lu')
    _stats(I_lu_a1, 'A1  I_lu')

    panel2 = side_by_side(lq_np, I_lu_a0, I_lu_a1)
    panel2 = add_labels(panel2, ['Input LQ', 'A0: I_lu', 'A1: I_lu'])
    out_path2 = os.path.join(args.out_dir, 'lit_up_comparison.png')
    save_png(out_path2, panel2)

    # ── Figure 3: Output comparison ────────────────────────────────────  #
    from skimage import img_as_ubyte
    from Enhancement import utils as _eu  # noqa — already imported as enh_utils

    p0 = enh_utils.PSNR(gt_np, pred_a0)
    s0 = enh_utils.calculate_ssim(img_as_ubyte(gt_np), img_as_ubyte(pred_a0))
    p1 = enh_utils.PSNR(gt_np, pred_a1)
    s1 = enh_utils.calculate_ssim(img_as_ubyte(gt_np), img_as_ubyte(pred_a1))

    print(f"\nMetrics on this image:")
    print(f"  A0  PSNR={p0:.4f} dB  SSIM={s0:.4f}")
    print(f"  A1  PSNR={p1:.4f} dB  SSIM={s1:.4f}")
    print(f"  Δ   PSNR={p1-p0:+.4f} dB  ΔSSIM={s1-s0:+.4f}")

    panel3 = side_by_side(pred_a0, pred_a1, gt_np)
    panel3 = add_labels(panel3,
        [f'A0 {p0:.2f}dB', f'A1 {p1:.2f}dB', 'GT'])
    out_path3 = os.path.join(args.out_dir, 'output_comparison.png')
    save_png(out_path3, panel3)

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
