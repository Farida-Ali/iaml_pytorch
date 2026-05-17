"""
Verify that training validation metrics match test_IAML_from_dataset.py exactly.
"""
import cv2
import math
import sys
import numpy as np
from skimage import img_as_ubyte

# ── Method A: exactly as test_IAML_from_dataset.py calls utils.py ────────────

sys.path.insert(0, 'Enhancement')
import utils as enhancement_utils

def method_a(target_f32, enhanced_f32):
    """Verbatim copy of the metric block in test_IAML_from_dataset.py."""
    psnr = enhancement_utils.PSNR(target_f32, enhanced_f32)
    ssim = enhancement_utils.calculate_ssim(
        img_as_ubyte(target_f32), img_as_ubyte(enhanced_f32))
    return psnr, ssim

# ── Method B: exactly as the fixed train_iaml.py validate() does ─────────────

def _psnr_rgb(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)

def _ssim_channel(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq  = mu1 ** 2
    mu2_sq  = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12   = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
               (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

def _ssim_rgb(img1, img2):
    return float(np.mean([_ssim_channel(img1[:, :, i], img2[:, :, i]) for i in range(3)]))

def method_b(target_f32, enhanced_f32):
    """Verbatim copy of the metric block in fixed train_iaml.py validate()."""
    psnr = _psnr_rgb(target_f32, enhanced_f32)
    ssim = _ssim_rgb(img_as_ubyte(target_f32), img_as_ubyte(enhanced_f32))
    return psnr, ssim

# ── Test with three different fake image pairs ────────────────────────────────

rng = np.random.default_rng(0)

print("Testing metric consistency across 3 image pairs:\n")
all_pass = True

for trial, (seed, shape) in enumerate([
    (0,  (400, 600, 3)),
    (1,  (256, 256, 3)),
    (42, (720, 1280, 3)),
], start=1):
    rng2 = np.random.default_rng(seed)
    enhanced_u8 = (rng2.random(shape) * 255).astype(np.uint8)
    gt_u8       = (rng2.random(shape) * 255).astype(np.uint8)

    # Both methods receive float32 [0,1] (same starting point as validate())
    enhanced_f32 = enhanced_u8.astype(np.float32) / 255.0
    gt_f32       = gt_u8.astype(np.float32) / 255.0

    psnr_a, ssim_a = method_a(gt_f32, enhanced_f32)
    psnr_b, ssim_b = method_b(gt_f32, enhanced_f32)

    psnr_diff = abs(psnr_a - psnr_b)
    ssim_diff = abs(ssim_a - ssim_b)

    ok = psnr_diff < 1e-4 and ssim_diff < 1e-4
    all_pass = all_pass and ok

    print(f"  Trial {trial} ({shape[0]}x{shape[1]}):")
    print(f"    Method A (test script):   PSNR={psnr_a:.6f}  SSIM={ssim_a:.6f}")
    print(f"    Method B (training val):  PSNR={psnr_b:.6f}  SSIM={ssim_b:.6f}")
    print(f"    Difference:               PSNR={psnr_diff:.2e}  SSIM={ssim_diff:.2e}  {'OK' if ok else 'MISMATCH'}")
    print()

if all_pass:
    print("PASS: Both methods give identical results (difference < 1e-4)")
    sys.exit(0)
else:
    print("FAIL: Methods differ — fix is incomplete")
    sys.exit(1)
