"""
Script 1: run_qualitative_inference.py
Run inference with a single model arch on a directory of LQ images,
compute per-image metrics vs GT, and save results to metrics.csv.

Usage example:
  python scripts/run_qualitative_inference.py \
      --arch FD2RT_A4 \
      --weights pretrained_weights/fd2rt_a4_lolv1.pth \
      --input_dir data/LOL-v1/eval15/low \
      --gt_dir data/LOL-v1/eval15/high \
      --output_dir qualitative_outputs/LOL-v1/A1 \
      --config Options/train_FD2RT_A4_LOL_v1.yml \
      --factor 4
"""

import sys
import os
import argparse
import csv
import time
import math

# ---------------------------------------------------------------------------
# Repo root on sys.path so basicsr and arch modules are importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim

# ---------------------------------------------------------------------------
# Default model kwargs for LOL-v1 (shared by all FD2RT variants and Retinexformer)
# ---------------------------------------------------------------------------
LOL_V1_KWARGS = dict(
    in_channels=3,
    out_channels=3,
    n_feat=40,
    stage=1,
    num_blocks=[1, 2, 2],
)

SUPPORTED_ARCHS = ("FD2RT_V1", "FD2RT_A2", "FD2RT_A4", "RetinexFormer")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(arch_name: str):
    """Instantiate the requested architecture with LOL-v1 default kwargs.

    Importing basicsr first ensures the arch registry is populated from the
    package's __init__.py auto-discovery.  Each arch module is then imported
    explicitly so the class is guaranteed to be available even if auto-discovery
    is incomplete.
    """
    import basicsr  # noqa: F401  -- triggers registry side-effects

    if arch_name == "FD2RT_V1":
        from basicsr.models.archs.fd2rt_v1_arch import FD2RT_V1
        return FD2RT_V1(**LOL_V1_KWARGS)
    elif arch_name == "FD2RT_A2":
        from basicsr.models.archs.fd2rt_a2_arch import FD2RT_A2
        return FD2RT_A2(**LOL_V1_KWARGS)
    elif arch_name == "FD2RT_A4":
        from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4
        return FD2RT_A4(**LOL_V1_KWARGS)
    elif arch_name == "RetinexFormer":
        from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
        return RetinexFormer(**LOL_V1_KWARGS)
    else:
        raise ValueError(
            f"Unknown arch '{arch_name}'. Choose from: {SUPPORTED_ARCHS}"
        )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_weights(model, weights_path: str, device: torch.device):
    """Load checkpoint weights into model.

    Strategy:
    1. Try strict=True.
    2. If RuntimeError: strip 'module.' prefix from all keys, then try
       strict=False.  Always log missing and unexpected keys — never suppress.
    """
    print(f"[Checkpoint] Loading '{weights_path}' …")
    ckpt = torch.load(weights_path, map_location=device)

    # Unwrap common checkpoint wrapper keys
    state_dict = ckpt
    for key in ("state_dict", "params", "params_ema", "model"):
        if isinstance(ckpt, dict) and key in ckpt:
            state_dict = ckpt[key]
            print(f"[Checkpoint] Extracted sub-key '{key}' from checkpoint dict.")
            break

    # Attempt strict load
    try:
        result = model.load_state_dict(state_dict, strict=True)
        print("[Checkpoint] Loaded with strict=True.")
        print(f"  missing keys    : {result.missing_keys}")
        print(f"  unexpected keys : {result.unexpected_keys}")
        return
    except RuntimeError as exc:
        print(f"[Checkpoint] strict=True failed: {exc}")

    # Strip 'module.' prefix, retry with strict=False
    print("[Checkpoint] Stripping 'module.' prefix and retrying with strict=False …")
    stripped = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    result = model.load_state_dict(stripped, strict=False)
    print(f"[Checkpoint] Loaded with strict=False.")
    print(f"  missing keys    ({len(result.missing_keys)}) : {result.missing_keys}")
    print(f"  unexpected keys ({len(result.unexpected_keys)}) : {result.unexpected_keys}")


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def pad_to_multiple(tensor: torch.Tensor, factor: int):
    """Reflect-pad [1,3,H,W] so H and W become exact multiples of factor.

    Padding is applied on the right/bottom edges only (simpler to unpad).
    Returns (padded_tensor, (pad_h, pad_w)) — amounts added.
    """
    _, _, h, w = tensor.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    # F.pad order: (left, right, top, bottom)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (pad_h, pad_w)


def unpad(tensor: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """Remove right/bottom padding added by pad_to_multiple."""
    _, _, H, W = tensor.shape
    h_end = H - pad_h if pad_h > 0 else H
    w_end = W - pad_w if pad_w > 0 else W
    return tensor[:, :, :h_end, :w_end]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def rgb_to_y_bt601(rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB [H,W,3] to BT.601 Y channel in [16, 235] (float64)."""
    r = rgb_uint8[:, :, 0].astype(np.float64)
    g = rgb_uint8[:, :, 1].astype(np.float64)
    b = rgb_uint8[:, :, 2].astype(np.float64)
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return np.clip(y, 16.0, 235.0)


def compute_psnr_rgb(pred_f32: np.ndarray, gt_f32: np.ndarray) -> float:
    """RGB PSNR: 10*log10(1 / MSE) on float32 [0,1] arrays."""
    mse = np.mean((pred_f32.astype(np.float64) - gt_f32.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def compute_psnr_y(pred_uint8: np.ndarray, gt_uint8: np.ndarray) -> float:
    """Y-channel PSNR via BT.601; Y values are in [0,255] float range."""
    y_pred = rgb_to_y_bt601(pred_uint8)
    y_gt = rgb_to_y_bt601(gt_uint8)
    mse = np.mean((y_pred - y_gt) ** 2)
    if mse == 0:
        return 100.0
    return 10.0 * math.log10((255.0 ** 2) / mse)


def compute_ssim_y(pred_uint8: np.ndarray, gt_uint8: np.ndarray) -> float:
    """Y-channel SSIM using skimage with data_range=255, win_size=11,
    gaussian_weights=True.  Inputs are uint8 RGB [H,W,3]."""
    y_pred = rgb_to_y_bt601(pred_uint8).astype(np.float64)
    y_gt = rgb_to_y_bt601(gt_uint8).astype(np.float64)
    return float(
        sk_ssim(y_gt, y_pred, data_range=255.0, win_size=11, gaussian_weights=True)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Qualitative inference + per-image metrics for low-light enhancement."
    )
    p.add_argument("--arch", required=True, choices=list(SUPPORTED_ARCHS),
                   help="Model architecture name.")
    p.add_argument("--weights", required=True,
                   help="Path to .pth checkpoint.")
    p.add_argument("--input_dir", required=True,
                   help="Directory of input LQ images.")
    p.add_argument("--gt_dir", required=True,
                   help="Directory of GT images (same filenames as input).")
    p.add_argument("--output_dir", required=True,
                   help="Where to save enhanced PNG outputs and metrics.csv.")
    p.add_argument("--config", default=None,
                   help="Path to training YAML (for reference; model kwargs fixed to LOL_V1_KWARGS).")
    p.add_argument("--factor", type=int, default=4,
                   help="Padding factor: H and W are padded to multiples of this (default: 4).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # Build and load model
    print(f"[Model] Building arch '{args.arch}' …")
    model = build_model(args.arch)
    load_weights(model, args.weights, device)
    model.to(device)
    model.eval()

    # Collect input image filenames
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    filenames = sorted(
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not filenames:
        print(f"[ERROR] No images found in '{args.input_dir}'.")
        sys.exit(1)
    print(f"[Dataset] Found {len(filenames)} images in '{args.input_dir}'.")

    rows = []

    with torch.inference_mode():
        for fname in filenames:
            inp_path = os.path.join(args.input_dir, fname)
            gt_path = os.path.join(args.gt_dir, fname)

            # Load input image: PIL RGB → float32 [0,1] → tensor [1,3,H,W]
            pil_inp = Image.open(inp_path).convert("RGB")
            inp_f32 = np.array(pil_inp, dtype=np.float32) / 255.0   # H,W,3
            orig_h, orig_w = inp_f32.shape[:2]
            inp_t = (
                torch.from_numpy(inp_f32.transpose(2, 0, 1))
                .unsqueeze(0)
                .to(device)
            )  # 1,3,H,W

            # Pad to multiple of factor
            inp_padded, (pad_h, pad_w) = pad_to_multiple(inp_t, args.factor)

            # Inference
            t0 = time.perf_counter()
            out_padded = model(inp_padded)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            infer_ms = (t1 - t0) * 1000.0

            # Handle multi-output models (some return tuples)
            if isinstance(out_padded, (list, tuple)):
                out_padded = out_padded[0]

            # Unpad to original H,W and clamp [0,1]
            out_t = unpad(out_padded, pad_h, pad_w)
            out_t = torch.clamp(out_t, 0.0, 1.0)

            # Convert to uint8 numpy [H,W,3]
            out_f32 = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()  # H,W,3
            out_uint8 = (out_f32 * 255.0).round().astype(np.uint8)

            # Save as PNG with same filename as input
            out_path = os.path.join(args.output_dir, fname)
            # Ensure .png extension
            base, _ = os.path.splitext(fname)
            out_path = os.path.join(args.output_dir, base + ".png")
            Image.fromarray(out_uint8).save(out_path)

            # Load GT and compute metrics
            if os.path.exists(gt_path):
                gt_pil = Image.open(gt_path).convert("RGB")
                gt_uint8 = np.array(gt_pil, dtype=np.uint8)
                gt_f32 = gt_uint8.astype(np.float32) / 255.0

                m_psnr_rgb = compute_psnr_rgb(out_f32, gt_f32)
                m_psnr_y = compute_psnr_y(out_uint8, gt_uint8)
                m_ssim_y = compute_ssim_y(out_uint8, gt_uint8)
            else:
                print(f"[WARNING] GT not found for '{fname}' — metrics set to NaN.")
                m_psnr_rgb = float("nan")
                m_psnr_y = float("nan")
                m_ssim_y = float("nan")

            rows.append({
                "filename": fname,
                "psnr_rgb": round(m_psnr_rgb, 6),
                "psnr_y": round(m_psnr_y, 6),
                "ssim_y": round(m_ssim_y, 6),
                "inference_time_ms": round(infer_ms, 3),
            })

            print(
                f"  {fname:30s}  PSNR_RGB={m_psnr_rgb:7.4f}  "
                f"PSNR_Y={m_psnr_y:7.4f}  SSIM_Y={m_ssim_y:.4f}  "
                f"time={infer_ms:.1f}ms"
            )

    # Save metrics.csv
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    fieldnames = ["filename", "psnr_rgb", "psnr_y", "ssim_y", "inference_time_ms"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Metrics] Saved to '{csv_path}'.")

    # Print final mean ± std for psnr_y and ssim_y
    valid_psnr_y = [r["psnr_y"] for r in rows if not math.isnan(r["psnr_y"])]
    valid_ssim_y = [r["ssim_y"] for r in rows if not math.isnan(r["ssim_y"])]

    if valid_psnr_y:
        arr_py = np.array(valid_psnr_y)
        arr_sy = np.array(valid_ssim_y)
        print("\n" + "=" * 60)
        print(f"  Arch   : {args.arch}")
        print(f"  Images : {len(filenames)}")
        print(f"  PSNR_Y : {arr_py.mean():.4f} ± {arr_py.std():.4f} dB")
        print(f"  SSIM_Y : {arr_sy.mean():.4f} ± {arr_sy.std():.4f}")
        print("=" * 60)
    else:
        print("[WARNING] No valid metrics computed (GT images missing?).")


if __name__ == "__main__":
    main()
