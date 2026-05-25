"""
run_qualitative_inference.py
────────────────────────────
Run inference for one model (FD2RT_V1, FD2RT_A2, FD2RT_A4, RetinexFormer)
on a directory of low-light images and compute PSNR / SSIM metrics vs GT.

Usage example:
  python scripts/run_qualitative_inference.py \
      --arch FD2RT_A4 \
      --weights pretrained_weights/FD2RT_A4_LOL_v1.pth \
      --input_dir data/LOL/eval15/low \
      --gt_dir data/LOL/eval15/high \
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
import logging

# ── repo root on sys.path ────────────────────────────────────────────────── #
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim

# ── logging setup ────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── default model kwargs for LOL-v1 ─────────────────────────────────────── #
LOL_V1_KWARGS = dict(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])

SUPPORTED_ARCHS = ("FD2RT_V1", "FD2RT_A2", "FD2RT_A4", "RetinexFormer")


# ────────────────────────────────────────────────────────────────────────────
# Model factory
# ────────────────────────────────────────────────────────────────────────────

def build_model(arch_name: str):
    """Instantiate the requested architecture with LOL-v1 default kwargs."""
    import basicsr  # noqa: F401 – triggers basicsr registry side-effects

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
        raise ValueError(f"Unknown arch '{arch_name}'. Choose from: {SUPPORTED_ARCHS}")


# ────────────────────────────────────────────────────────────────────────────
# Weight loading
# ────────────────────────────────────────────────────────────────────────────

def load_weights(model, weights_path: str, device: torch.device):
    """
    Load checkpoint into model.
    1. Try strict=True.
    2. On RuntimeError: strip 'module.' prefix from keys, try strict=False,
       and log missing / unexpected keys — never suppress them.
    """
    log.info(f"Loading weights from {weights_path}")
    ckpt = torch.load(weights_path, map_location=device)

    # unwrap common checkpoint wrappers
    state_dict = ckpt
    for key in ("state_dict", "params", "params_ema", "model"):
        if isinstance(ckpt, dict) and key in ckpt:
            state_dict = ckpt[key]
            log.info(f"  Extracted sub-key '{key}' from checkpoint.")
            break

    # first attempt: strict=True
    try:
        model.load_state_dict(state_dict, strict=True)
        log.info("  Weights loaded with strict=True.")
        return
    except RuntimeError as exc:
        log.warning(f"  strict=True failed: {exc}")

    # second attempt: strip 'module.' prefix, strict=False
    stripped = {k[len("module."):] if k.startswith("module.") else k: v
                for k, v in state_dict.items()}
    result = model.load_state_dict(stripped, strict=False)
    if result.missing_keys:
        log.warning(f"  MISSING keys ({len(result.missing_keys)}): {result.missing_keys[:10]}"
                    + (" ..." if len(result.missing_keys) > 10 else ""))
    if result.unexpected_keys:
        log.warning(f"  UNEXPECTED keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:10]}"
                    + (" ..." if len(result.unexpected_keys) > 10 else ""))
    log.info("  Weights loaded with strict=False (module. prefix stripped).")


# ────────────────────────────────────────────────────────────────────────────
# Padding helpers
# ────────────────────────────────────────────────────────────────────────────

def pad_to_multiple(tensor: torch.Tensor, factor: int):
    """Reflect-pad NCHW tensor so H and W are multiples of factor.
    Returns (padded_tensor, (pad_left, pad_right, pad_top, pad_bottom)).
    """
    _, _, h, w = tensor.shape
    new_h = math.ceil(h / factor) * factor
    new_w = math.ceil(w / factor) * factor
    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left
    padded = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
    return padded, (pad_left, pad_right, pad_top, pad_bottom)


def unpad(tensor: torch.Tensor, pads, orig_h: int, orig_w: int):
    """Remove padding applied by pad_to_multiple."""
    pad_left, pad_right, pad_top, pad_bottom = pads
    _, _, ph, pw = tensor.shape
    return tensor[
        :,
        :,
        pad_top: ph - pad_bottom if pad_bottom > 0 else ph,
        pad_left: pw - pad_right if pad_right > 0 else pw,
    ]


# ────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ────────────────────────────────────────────────────────────────────────────

def rgb_to_y_bt601(rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB [H,W,3] to BT.601 Y channel in [16, 235]."""
    r = rgb_uint8[:, :, 0].astype(np.float32)
    g = rgb_uint8[:, :, 1].astype(np.float32)
    b = rgb_uint8[:, :, 2].astype(np.float32)
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return np.clip(y, 16.0, 235.0)


def psnr_rgb(pred_f32: np.ndarray, gt_f32: np.ndarray) -> float:
    """PSNR on float32 [0,1] arrays (full RGB)."""
    mse = np.mean((pred_f32 - gt_f32) ** 2)
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def psnr_y(pred_uint8: np.ndarray, gt_uint8: np.ndarray) -> float:
    """PSNR on BT.601 Y channel, values in [0,255] float range."""
    y_pred = rgb_to_y_bt601(pred_uint8)
    y_gt = rgb_to_y_bt601(gt_uint8)
    mse = np.mean((y_pred - y_gt) ** 2)
    if mse == 0:
        return 100.0
    return 10.0 * math.log10((255.0 ** 2) / mse)


def ssim_y(pred_uint8: np.ndarray, gt_uint8: np.ndarray) -> float:
    """SSIM on BT.601 Y channel (uint8, data_range=255)."""
    y_pred = rgb_to_y_bt601(pred_uint8).astype(np.float32)
    y_gt = rgb_to_y_bt601(gt_uint8).astype(np.float32)
    return float(sk_ssim(y_gt, y_pred, data_range=255, win_size=11,
                         gaussian_weights=True))


# ────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Qualitative inference + metrics")
    p.add_argument("--arch", required=True, choices=list(SUPPORTED_ARCHS),
                   help="Model architecture name")
    p.add_argument("--weights", required=True, help="Path to .pth checkpoint")
    p.add_argument("--input_dir", required=True, help="Directory of LQ input images")
    p.add_argument("--gt_dir", required=True, help="Directory of GT images")
    p.add_argument("--output_dir", required=True, help="Where to save enhanced PNG outputs")
    p.add_argument("--config", default=None,
                   help="Path to training YAML (currently for reference; kwargs are fixed)")
    p.add_argument("--factor", type=int, default=4,
                   help="Padding factor so H,W are multiples of this (default: 4)")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── build & load model ──────────────────────────────────────────────── #
    model = build_model(args.arch)
    load_weights(model, args.weights, device)
    model.to(device)
    model.eval()

    # ── discover input images ───────────────────────────────────────────── #
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    filenames = sorted(
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not filenames:
        log.error(f"No images found in {args.input_dir}")
        sys.exit(1)
    log.info(f"Found {len(filenames)} images in {args.input_dir}")

    # ── per-image processing ────────────────────────────────────────────── #
    rows = []
    psnr_rgb_list, psnr_y_list, ssim_y_list = [], [], []

    with torch.inference_mode():
        for fname in filenames:
            inp_path = os.path.join(args.input_dir, fname)
            gt_path = os.path.join(args.gt_dir, fname)

            # load input
            pil_inp = Image.open(inp_path).convert("RGB")
            inp_np = np.array(pil_inp, dtype=np.float32) / 255.0  # [H,W,3]
            orig_h, orig_w = inp_np.shape[:2]

            # to tensor
            inp_t = torch.from_numpy(inp_np.transpose(2, 0, 1)).unsqueeze(0).to(device)  # [1,3,H,W]

            # pad
            inp_padded, pads = pad_to_multiple(inp_t, args.factor)

            # inference
            t0 = time.perf_counter()
            out_padded = model(inp_padded)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            infer_ms = (t1 - t0) * 1000.0

            # unpad + clamp
            out_t = unpad(out_padded, pads, orig_h, orig_w)
            out_t = torch.clamp(out_t, 0.0, 1.0)

            # to numpy uint8
            out_np_f32 = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H,W,3] float32
            out_uint8 = (out_np_f32 * 255.0).round().astype(np.uint8)

            # save PNG
            out_path = os.path.join(args.output_dir, os.path.splitext(fname)[0] + ".png")
            Image.fromarray(out_uint8).save(out_path)

            # metrics vs GT
            if os.path.exists(gt_path):
                gt_pil = Image.open(gt_path).convert("RGB")
                gt_uint8 = np.array(gt_pil, dtype=np.uint8)
                gt_f32 = gt_uint8.astype(np.float32) / 255.0

                # resize GT to match output if needed (safety)
                if gt_uint8.shape[:2] != (orig_h, orig_w):
                    gt_pil = gt_pil.resize((orig_w, orig_h), Image.BICUBIC)
                    gt_uint8 = np.array(gt_pil, dtype=np.uint8)
                    gt_f32 = gt_uint8.astype(np.float32) / 255.0

                m_psnr_rgb = psnr_rgb(out_np_f32, gt_f32)
                m_psnr_y = psnr_y(out_uint8, gt_uint8)
                m_ssim_y = ssim_y(out_uint8, gt_uint8)
            else:
                log.warning(f"GT not found for {fname}, metrics set to NaN")
                m_psnr_rgb = float("nan")
                m_psnr_y = float("nan")
                m_ssim_y = float("nan")

            psnr_rgb_list.append(m_psnr_rgb)
            psnr_y_list.append(m_psnr_y)
            ssim_y_list.append(m_ssim_y)

            row = dict(
                filename=fname,
                psnr_rgb=round(m_psnr_rgb, 4),
                psnr_y=round(m_psnr_y, 4),
                ssim_y=round(m_ssim_y, 4),
                inference_time_ms=round(infer_ms, 2),
            )
            rows.append(row)

            log.info(
                f"  {fname:30s}  PSNR_RGB={m_psnr_rgb:.2f}  "
                f"PSNR_Y={m_psnr_y:.2f}  SSIM_Y={m_ssim_y:.4f}  "
                f"t={infer_ms:.1f}ms"
            )

    # ── save metrics.csv ────────────────────────────────────────────────── #
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    fieldnames = ["filename", "psnr_rgb", "psnr_y", "ssim_y", "inference_time_ms"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Metrics saved to {csv_path}")

    # ── summary statistics ──────────────────────────────────────────────── #
    valid_psnr_y = [v for v in psnr_y_list if not math.isnan(v)]
    valid_ssim_y = [v for v in ssim_y_list if not math.isnan(v)]

    if valid_psnr_y:
        mean_psnr_y = float(np.mean(valid_psnr_y))
        std_psnr_y = float(np.std(valid_psnr_y))
        mean_ssim_y = float(np.mean(valid_ssim_y))
        std_ssim_y = float(np.std(valid_ssim_y))

        print("\n" + "=" * 60)
        print(f"  Arch   : {args.arch}")
        print(f"  Images : {len(filenames)}")
        print(f"  PSNR_Y : {mean_psnr_y:.4f} ± {std_psnr_y:.4f} dB")
        print(f"  SSIM_Y : {mean_ssim_y:.4f} ± {std_ssim_y:.4f}")
        print("=" * 60)
    else:
        log.warning("No valid metrics computed (GT images missing?).")


if __name__ == "__main__":
    main()
