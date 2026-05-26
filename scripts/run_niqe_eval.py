"""
scripts/run_niqe_eval.py
────────────────────────
Inference + NIQE / BRISQUE / PIQE evaluation on no-reference (no-GT)
low-light datasets.

Loads a trained model, enhances every image in --input_dir, saves the
enhanced PNGs to --output_dir, and computes three no-reference metrics:

  NIQE   — built into basicsr/metrics/niqe.py (always available)
  BRISQUE — tries piq library, then cv2.quality (opencv-contrib)
  PIQE   — tries piq library

All three: lower score = better perceptual quality.

IMPORTANT: Run from the repository root, e.g.:
    cd /workspace/fd2rt/Retinexformer
    python scripts/run_niqe_eval.py ...

This is required because basicsr/metrics/niqe.py loads its pristine-dataset
params from the relative path 'basicsr/metrics/niqe_pris_params.npz'.

If BRISQUE / PIQE show "nan" the piq library is not installed.
Install it with:  pip install piq
(opencv-contrib also works for BRISQUE: pip install opencv-contrib-python)

Usage examples
──────────────
# A4_fixed on LIME
python scripts/run_niqe_eval.py \
    --arch        FD2RT_A4 \
    --weights     experiments/train_FD2RT_A4_LOL_v1_fixed/models/best_psnr_24.06_98000.pth \
    --input_dir   data/LIME \
    --output_dir  qualitative_outputs/non_reference/LIME/IAML_A4 \
    --dataset_name LIME \
    --model_name  IAML_A4

# Retinexformer_A0 on LIME
python scripts/run_niqe_eval.py \
    --arch        RetinexFormer \
    --weights     experiments/train_Retinexformer_LOL_v1_fixed/models/net_g_best.pth \
    --input_dir   data/LIME \
    --output_dir  qualitative_outputs/non_reference/LIME/Retinexformer_A0 \
    --dataset_name LIME \
    --model_name  Retinexformer_A0

Output
──────
  <output_dir>/
      <filename>.png   — enhanced image (lossless)
      niqe_results.csv — columns: dataset, model, filename,
                                  niqe_score, brisque_score, piqe_score
"""

import sys
import os
import csv
import time
import argparse
from glob import glob
from datetime import datetime, timezone

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from natsort import natsorted
from PIL import Image


# ── Architecture registry ──────────────────────────────────────────────── #

LOL_V1_KWARGS = dict(in_channels=3, out_channels=3, n_feat=40,
                     stage=1, num_blocks=[1, 2, 2])


def build_model(arch_name: str):
    import basicsr  # noqa — triggers arch registration
    if arch_name == 'FD2RT_V1':
        from basicsr.models.archs.fd2rt_v1_arch import FD2RT_V1
        return FD2RT_V1(**LOL_V1_KWARGS)
    elif arch_name == 'FD2RT_A2':
        from basicsr.models.archs.fd2rt_a2_arch import FD2RT_A2
        return FD2RT_A2(**LOL_V1_KWARGS)
    elif arch_name == 'FD2RT_A4':
        from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4
        return FD2RT_A4(**LOL_V1_KWARGS)
    elif arch_name in ('RetinexFormer', 'Retinexformer'):
        from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
        return RetinexFormer(**LOL_V1_KWARGS)
    else:
        raise ValueError(
            f'Unknown arch "{arch_name}". '
            f'Choices: FD2RT_V1, FD2RT_A2, FD2RT_A4, RetinexFormer')


def load_weights(model, weights_path: str, device):
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get('params', ckpt.get('state_dict', ckpt))
    try:
        model.load_state_dict(state, strict=True)
        print(f'[Weights] Loaded strict=True: {os.path.basename(weights_path)}')
    except RuntimeError as e:
        print(f'[Weights] strict=True failed ({e}), trying strict=False '
              f'with module. prefix strip …')
        state = {k.replace('module.', ''): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'[Weights] strict=False: missing={len(missing)}, '
              f'unexpected={len(unexpected)}')
        if missing:
            print(f'  Missing keys : {missing[:5]}{"…" if len(missing)>5 else ""}')
        if unexpected:
            print(f'  Unexpected   : {unexpected[:5]}{"…" if len(unexpected)>5 else ""}')
    return model


# ── Inference ─────────────────────────────────────────────────────────── #

def infer_image(model, img_rgb_f32: np.ndarray, device, factor: int = 4) -> np.ndarray:
    """Run model on a float32 [0,1] RGB numpy array. Returns float32 [0,1] RGB."""
    t = torch.from_numpy(img_rgb_f32).permute(2, 0, 1).unsqueeze(0).to(device)
    h, w = t.shape[2], t.shape[3]
    padh = (-h % factor)
    padw = (-w % factor)
    if padh or padw:
        t = F.pad(t, (0, padw, 0, padh), mode='reflect')
    with torch.inference_mode():
        out = model(t)
    if isinstance(out, (list, tuple)):
        out = out[-1]
    out = out[:, :, :h, :w]
    return torch.clamp(out, 0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)


# ── NIQE ──────────────────────────────────────────────────────────────── #

def compute_niqe(img_rgb_f32: np.ndarray) -> float:
    """
    Compute NIQE score on a float32 [0,1] RGB image.

    Protocol (matches Retinexformer paper):
      - Convert RGB [0,1] → BGR [0,255] uint8  (OpenCV convention)
      - Pass to calculate_niqe with input_order='HWC', convert_to='y'
        which internally converts BGR → Y-channel of YCbCr
      - crop_border=0 (no GT crop needed for no-reference images)
    Returns float NIQE score (lower = better), or nan if image is too small.
    """
    from basicsr.metrics import calculate_niqe

    # Convert RGB [0,1] float32 → BGR [0,255] float32
    img_bgr = cv2.cvtColor((img_rgb_f32 * 255.0).clip(0, 255).astype(np.float32),
                            cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]
    if h < 96 or w < 96:
        print(f'  [NIQE] Image too small ({h}×{w}) — need ≥96×96. Returning nan.')
        return float('nan')

    try:
        score = float(calculate_niqe(img_bgr, crop_border=0,
                                     input_order='HWC', convert_to='y'))
    except Exception as exc:
        print(f'  [NIQE] Computation failed: {exc}. Returning nan.')
        score = float('nan')
    return score


def compute_brisque(img_rgb_f32: np.ndarray) -> float:
    """
    Compute BRISQUE score on a float32 [0,1] RGB image (lower = better).

    Backend priority:
      1. piq.brisque()          — pip install piq
      2. cv2.quality.QualityBRISQUE_compute()  — pip install opencv-contrib-python
         (requires brisque_model_live.yml and brisque_range_live.yml on PATH)
      3. Returns nan with a one-time warning if neither is available.

    Input convention for piq: float32 tensor [0,1], RGB, shape (1,3,H,W).
    Input convention for cv2.quality: uint8 BGR, shape (H,W,3).
    """
    # --- backend 1: piq ---
    try:
        import piq
        t = torch.from_numpy(img_rgb_f32).permute(2, 0, 1).unsqueeze(0)
        return float(piq.brisque(t, data_range=1.0, reduction='none')[0])
    except ImportError:
        pass
    except Exception as exc:
        print(f'  [BRISQUE/piq] Failed: {exc}')
        return float('nan')

    # --- backend 2: cv2.quality (opencv-contrib) ---
    try:
        import cv2.quality as cvq  # noqa
        # cv2.quality needs the LIVE model files; look in common locations
        model_candidates = [
            'brisque_model_live.yml',
            '/usr/share/opencv4/quality/brisque_model_live.yml',
        ]
        range_candidates = [
            'brisque_range_live.yml',
            '/usr/share/opencv4/quality/brisque_range_live.yml',
        ]
        model_path = next((p for p in model_candidates if os.path.isfile(p)), None)
        range_path = next((p for p in range_candidates if os.path.isfile(p)), None)
        if model_path and range_path:
            img_bgr_u8 = cv2.cvtColor(
                (img_rgb_f32 * 255.0).clip(0, 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR)
            qs = cvq.QualityBRISQUE_create(model_path, range_path)
            return float(qs.compute(img_bgr_u8)[0])
        else:
            # Model files not found — fall through to nan
            pass
    except (ImportError, AttributeError):
        pass
    except Exception as exc:
        print(f'  [BRISQUE/cv2] Failed: {exc}')
        return float('nan')

    return float('nan')


def compute_piqe(img_rgb_f32: np.ndarray) -> float:
    """
    Compute PIQE score on a float32 [0,1] RGB image (lower = better).

    Uses piq.piqe() — pip install piq.
    Returns nan if piq is not installed.

    Input convention: float32 tensor [0,1], RGB, shape (1,3,H,W).
    """
    try:
        import piq
        t = torch.from_numpy(img_rgb_f32).permute(2, 0, 1).unsqueeze(0)
        return float(piq.piqe(t, data_range=1.0, reduction='none')[0])
    except ImportError:
        pass
    except Exception as exc:
        print(f'  [PIQE/piq] Failed: {exc}')
    return float('nan')


# ── One-time availability check ───────────────────────────────────────── #

def check_metric_backends():
    """Print which optional backends are available at startup."""
    try:
        import piq
        print(f'  BRISQUE/PIQE : piq {piq.__version__} ✓')
        return
    except ImportError:
        pass
    try:
        import cv2.quality  # noqa
        print('  BRISQUE      : cv2.quality (opencv-contrib) ✓')
        print('  PIQE         : not available (install piq)')
        return
    except (ImportError, AttributeError):
        pass
    print('  BRISQUE/PIQE : NOT AVAILABLE — scores will be nan')
    print('                 Install with: pip install piq')


# ── CLI ───────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(
        description='Inference + NIQE evaluation on no-reference low-light datasets.')
    p.add_argument('--arch',         required=True,
                   choices=['FD2RT_V1', 'FD2RT_A2', 'FD2RT_A4', 'RetinexFormer'],
                   help='Model architecture name.')
    p.add_argument('--weights',      required=True,
                   help='Path to .pth checkpoint.')
    p.add_argument('--input_dir',    required=True,
                   help='Directory of low-light input images (no GT required).')
    p.add_argument('--output_dir',   required=True,
                   help='Where to save enhanced PNGs and niqe_results.csv.')
    p.add_argument('--dataset_name', required=True,
                   help='Dataset label used in the CSV (e.g. LIME, NPE, MEF).')
    p.add_argument('--model_name',   required=True,
                   help='Model label used in the CSV (e.g. IAML_A4, Retinexformer_A0).')
    p.add_argument('--factor',       type=int, default=4,
                   help='Padding factor for inference (default 4).')
    p.add_argument('--cpu',          action='store_true',
                   help='Force CPU inference.')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────── #

def main():
    args = parse_args()

    # Verify NIQE params file is reachable (must run from repo root)
    niqe_params = os.path.join(_ROOT, 'basicsr', 'metrics', 'niqe_pris_params.npz')
    if not os.path.isfile(niqe_params):
        print(f'ERROR: NIQE params file not found at:\n  {niqe_params}')
        print('Make sure you run this script from the repository root.')
        return 1

    device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda')
    print(f'\n[Setup]')
    print(f'  Arch        : {args.arch}')
    print(f'  Weights     : {args.weights}')
    print(f'  Input dir   : {args.input_dir}')
    print(f'  Output dir  : {args.output_dir}')
    print(f'  Dataset     : {args.dataset_name}')
    print(f'  Model label : {args.model_name}')
    print(f'  Device      : {device}')

    # Discover input images
    img_paths = natsorted(
        glob(os.path.join(args.input_dir, '*.png')) +
        glob(os.path.join(args.input_dir, '*.jpg')) +
        glob(os.path.join(args.input_dir, '*.JPG')) +
        glob(os.path.join(args.input_dir, '*.bmp')) +
        glob(os.path.join(args.input_dir, '*.PNG'))
    )
    if not img_paths:
        print(f'ERROR: no images found in {args.input_dir}')
        return 1
    print(f'  Images found: {len(img_paths)}')

    os.makedirs(args.output_dir, exist_ok=True)

    # Build and load model
    print('\n[Model]')
    model = build_model(args.arch)
    model = load_weights(model, args.weights, device)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters  : {n_params:,}')

    print('\n[Metric backends]')
    check_metric_backends()

    # Run inference + metrics
    print(f'\n[Inference + NIQE / BRISQUE / PIQE]')
    print(f'{"Filename":40s}  {"NIQE":>8}  {"BRISQUE":>8}  {"PIQE":>8}  {"Time ms":>8}')
    print('─' * 82)

    csv_rows = []
    niqe_scores    = []
    brisque_scores = []
    piqe_scores    = []

    for img_path in img_paths:
        fname = os.path.basename(img_path)
        t0 = time.perf_counter()

        # Load image
        img_pil = Image.open(img_path).convert('RGB')
        img_f32 = np.array(img_pil, dtype=np.float32) / 255.0

        # Infer
        out_f32 = infer_image(model, img_f32, device, args.factor)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Save enhanced PNG (lossless)
        stem = os.path.splitext(fname)[0]
        save_path = os.path.join(args.output_dir, stem + '.png')
        out_uint8 = (out_f32 * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(out_uint8, mode='RGB').save(save_path)

        # Metrics
        niqe_score    = compute_niqe(out_f32)
        brisque_score = compute_brisque(out_f32)
        piqe_score    = compute_piqe(out_f32)

        if not np.isnan(niqe_score):
            niqe_scores.append(niqe_score)
        if not np.isnan(brisque_score):
            brisque_scores.append(brisque_score)
        if not np.isnan(piqe_score):
            piqe_scores.append(piqe_score)

        def fmt(v):
            return f'{v:8.4f}' if not np.isnan(v) else '     nan'

        print(f'{fname:40s}  {fmt(niqe_score)}  {fmt(brisque_score)}  {fmt(piqe_score)}  {elapsed_ms:>8.1f}')
        csv_rows.append({
            'dataset':      args.dataset_name,
            'model':        args.model_name,
            'filename':     fname,
            'niqe_score':   f'{niqe_score:.6f}'    if not np.isnan(niqe_score)    else 'nan',
            'brisque_score': f'{brisque_score:.6f}' if not np.isnan(brisque_score) else 'nan',
            'piqe_score':   f'{piqe_score:.6f}'    if not np.isnan(piqe_score)    else 'nan',
        })

    # Summary stats
    print('─' * 82)

    def summarise(label, scores):
        if scores:
            m, s = float(np.mean(scores)), float(np.std(scores))
            print(f'  {label:<12} mean ± std : {m:.4f} ± {s:.4f}  (N={len(scores)}, lower is better)')
            return m, s
        print(f'  {label:<12} : no valid scores (metric not available or all nan).')
        return float('nan'), float('nan')

    print()
    summarise('NIQE',    niqe_scores)
    summarise('BRISQUE', brisque_scores)
    summarise('PIQE',    piqe_scores)

    # Save CSV
    csv_path = os.path.join(args.output_dir, 'niqe_results.csv')
    fieldnames = ['dataset', 'model', 'filename', 'niqe_score', 'brisque_score', 'piqe_score']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    print(f'\n[Saved] {csv_path}')
    print(f'[Saved] {len(csv_rows)} enhanced images → {args.output_dir}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
