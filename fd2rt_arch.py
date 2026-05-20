"""
FD²RT — Frequency-Dual-Domain Retinex Transformer  (Phase 1 component)
=======================================================================
WaveletIlluminationEstimator: drop-in replacement for Retinexformer's
pixel-domain Illumination_Estimator.

Physics motivation
------------------
Retinex theory decomposes an image as  I = R ⊙ L,  where:
  L  (illumination)  is smooth → low-frequency  → captured by DWT LL subband
  R  (reflectance)   is high-frequency → captured by LH, HL, HH subbands
  n  (noise)         is highest-frequency → dominant in HH subband

Retinexformer uses  mean_c(I) = mean over colour channels  as its illumination
prior.  That scalar field lives in the pixel domain and mixes illumination with
texture and noise.  We replace it with the LL subband, which is, by the
definition of the Haar DWT, the low-frequency approximation of I — directly
aligned with the smooth-illumination assumption of the Retinex model.

Interface
---------
forward(I: [B, 3, H, W]) -> (F_lu, I_lu, N_map)

  F_lu  [B, C, H, W]   Illumination guidance feature  (same role as
                        Retinexformer's illu_fea — passed into every IGAB).
  I_lu  [B, 3, H, W]   Light-up image: I * illu_map + I  (identical formula
                        to Retinexformer; feeds into the Denoiser).
  N_map [B, 1, H, W]   Noise-awareness map estimated from HH subband.
                        Reserved for Phase 2 DDA attention; not consumed here.

Compared with original Illumination_Estimator
---------------------------------------------
Changed:  mean_c(I)          →  LL_mean  (channel-mean of upsampled LL subband)
Changed:  returns (illu_fea, illu_map)  →  returns (F_lu, I_lu, N_map)
Added:    N_map branch (two conv3×3 layers on HH subband, ~297 params)
Unchanged: conv1 → depth_conv → conv2 stack, all shapes, all scale factors
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavelet_utils import HaarDWT2D, HaarIDWT2D


# ---------------------------------------------------------------------------
# WaveletIlluminationEstimator
# ---------------------------------------------------------------------------

class WaveletIlluminationEstimator(nn.Module):
    """
    Wavelet-domain illumination estimator for FD²RT.

    Args:
        n_fea_middle (int): Feature channels C throughout the estimator.
                            Must equal Retinexformer's n_feat (40 for LOL-v1).
        n_fea_in     (int): Input channels to conv1.  Default 4 preserves the
                            original architecture: cat([I(3ch), prior(1ch)]).
        n_fea_out    (int): Output channels of illu_map.  Default 3 = RGB.
        n_nmap_hidden(int): Hidden channels in the N_map branch.  Default 8.
    """

    def __init__(
        self,
        n_fea_middle: int,
        n_fea_in: int = 4,
        n_fea_out: int = 3,
        n_nmap_hidden: int = 8,
    ) -> None:
        super().__init__()

        # ── DWT / IDWT ────────────────────────────────────────────────── #
        # No learnable parameters; move with .to(device) automatically.
        self.dwt = HaarDWT2D()
        # IDWT kept for potential future use; not called in this phase.
        self.idwt = HaarIDWT2D()

        # ── Main illumination path (identical to Illumination_Estimator) ─ #
        # conv1: [B, n_fea_in, H, W] → [B, n_fea_middle, H, W]
        self.conv1 = nn.Conv2d(
            n_fea_in, n_fea_middle, kernel_size=1, bias=True
        )

        # depth_conv: depthwise-grouped 5×5 conv, groups = n_fea_in
        # weight shape: [n_fea_middle, n_fea_middle // n_fea_in, 5, 5]
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle,
            kernel_size=5, padding=2, bias=True, groups=n_fea_in,
        )

        # conv2: [B, n_fea_middle, H, W] → [B, n_fea_out, H, W]  (illu_map)
        self.conv2 = nn.Conv2d(
            n_fea_middle, n_fea_out, kernel_size=1, bias=True
        )

        # ── N_map branch: HH subband → noise-awareness map ─────────────── #
        # Input: HH [B, 3, H/2, W/2]  (3 = RGB channels of the input image)
        # Output after upsample: N_map [B, 1, H, W]
        self.nmap_branch = nn.Sequential(
            nn.Conv2d(3, n_nmap_hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_nmap_hidden, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),   # N_map ∈ [0, 1] — interpretable as noise magnitude
        )

        # Store for external inspection / assertions
        self._n_fea_middle = n_fea_middle
        self._n_fea_out = n_fea_out

    # ------------------------------------------------------------------ #
    def forward(self, img: torch.Tensor):
        """
        Args:
            img: [B, 3, H, W]   input low-light image, values in [0, 1]
                                H and W must be even (DWT requirement).
        Returns:
            F_lu  [B, C, H, W]   illumination guidance features
            I_lu  [B, 3, H, W]   lit-up image (I * illu_map + I)
            N_map [B, 1, H, W]   noise-awareness map (from HH subband)
        """
        B, _, H, W = img.shape

        # ── 1. Haar DWT: decompose I into 4 subbands ──────────────────── #
        # Each subband: [B, 3, H/2, W/2]
        LL, LH, HL, HH = self.dwt(img)
        # LL ≈ smooth illumination field  (Retinex L component)
        # HH ≈ high-freq detail + noise   (Retinex noise proxy)

        # ── 2. Build illumination prior from LL ────────────────────────── #
        # Upsample LL from H/2×W/2 back to H×W (bilinear, no artefacts).
        LL_up = F.interpolate(LL, size=(H, W), mode='bilinear', align_corners=False)
        # [B, 3, H, W]

        # Channel-mean → 1 ch  (same cardinality as original mean_c).
        # This preserves the 4-channel input to conv1 and all downstream
        # weight shapes — no architecture surgery required.
        LL_mean = LL_up.mean(dim=1, keepdim=True)   # [B, 1, H, W]

        # ── 3. Illumination feature extraction (identical to original) ─── #
        # cat([img(3), LL_mean(1)]) mirrors cat([img(3), mean_c(1)]) exactly.
        inp  = torch.cat([img, LL_mean], dim=1)      # [B, 4, H, W]
        x_1  = self.conv1(inp)                        # [B, C, H, W]
        F_lu = self.depth_conv(x_1)                   # [B, C, H, W]
        illu_map = self.conv2(F_lu)                   # [B, 3, H, W]

        # ── 4. Light-up image (identical formula to Retinexformer) ──────── #
        I_lu = img * illu_map + img                   # [B, 3, H, W]

        # ── 5. N_map from HH subband ─────────────────────────────────────  #
        # HH [B, 3, H/2, W/2] → conv block → [B, 1, H/2, W/2]
        N_map_small = self.nmap_branch(HH)
        # Upsample to full resolution for later use in DDA attention
        N_map = F.interpolate(N_map_small, size=(H, W), mode='bilinear',
                              align_corners=False)     # [B, 1, H, W]

        return F_lu, I_lu, N_map


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def run_verification():
    """
    Three checks:
      1. Output shapes on [4, 3, 256, 256] random input.
      2. Wall-clock runtime vs original Illumination_Estimator.
      3. Parameter count delta.
    """
    import time
    import sys

    # ── Inline original estimator for fair comparison ──────────────────── #
    class _OriginalEstimator(nn.Module):
        def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
            super().__init__()
            self.conv1      = nn.Conv2d(n_fea_in, n_fea_middle, 1, bias=True)
            self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, 5,
                                        padding=2, bias=True, groups=n_fea_in)
            self.conv2      = nn.Conv2d(n_fea_middle, n_fea_out, 1, bias=True)

        def forward(self, img):
            mean_c = img.mean(dim=1, keepdim=True)
            x      = torch.cat([img, mean_c], dim=1)
            x      = self.conv1(x)
            illu_fea = self.depth_conv(x)
            illu_map = self.conv2(illu_fea)
            return illu_fea, illu_map

    n_feat = 40
    B, C_in, H, W = 4, 3, 256, 256
    x = torch.randn(B, C_in, H, W)

    wie = WaveletIlluminationEstimator(n_feat).eval()
    orig = _OriginalEstimator(n_feat).eval()

    PASS = "PASS"
    FAIL = "FAIL"
    all_ok = True

    # ── Check 1: Output shapes ─────────────────────────────────────────── #
    print("\n" + "="*60)
    print("CHECK 1 — Output shapes on input [4, 3, 256, 256]")
    print("="*60)
    with torch.no_grad():
        F_lu, I_lu, N_map = wie(x)

    expected = {
        "F_lu  (illumination features)": (B, n_feat, H, W),
        "I_lu  (lit-up image)         ": (B, 3,      H, W),
        "N_map (noise-awareness map)   ": (B, 1,      H, W),
    }
    actual = {
        "F_lu  (illumination features)": tuple(F_lu.shape),
        "I_lu  (lit-up image)         ": tuple(I_lu.shape),
        "N_map (noise-awareness map)   ": tuple(N_map.shape),
    }
    for name, exp in expected.items():
        got = actual[name]
        ok  = got == exp
        all_ok &= ok
        print(f"  {PASS if ok else FAIL}  {name}  expected={exp}  got={got}")

    # N_map values must be in [0, 1] (Sigmoid output)
    nmap_ok = N_map.min().item() >= 0.0 and N_map.max().item() <= 1.0
    all_ok &= nmap_ok
    print(f"  {'PASS' if nmap_ok else 'FAIL'}  N_map range [0,1]"
          f"  min={N_map.min().item():.4f}  max={N_map.max().item():.4f}")

    # ── Check 2: Runtime comparison ────────────────────────────────────── #
    print("\n" + "="*60)
    print("CHECK 2 — Wall-clock runtime (CPU, 10 warm-up + 20 timed iters)")
    print("="*60)
    WARMUP, REPS = 10, 20

    with torch.no_grad():
        for _ in range(WARMUP):
            orig(x)
        t0 = time.perf_counter()
        for _ in range(REPS):
            orig(x)
        t_orig = (time.perf_counter() - t0) / REPS * 1000

        for _ in range(WARMUP):
            wie(x)
        t0 = time.perf_counter()
        for _ in range(REPS):
            wie(x)
        t_wie = (time.perf_counter() - t0) / REPS * 1000

    overhead_pct = (t_wie - t_orig) / t_orig * 100
    print(f"  Original IE  : {t_orig:.2f} ms/iter")
    print(f"  W-IE (ours)  : {t_wie:.2f} ms/iter")
    print(f"  Overhead     : {overhead_pct:+.1f}%")

    # ── Check 3: Parameter count ───────────────────────────────────────── #
    print("\n" + "="*60)
    print("CHECK 3 — Parameter count")
    print("="*60)

    n_wie  = _count_params(wie)
    n_orig = _count_params(orig)
    delta  = n_wie - n_orig
    within = abs(delta) <= 50_000
    all_ok &= within

    print(f"  Original IE  : {n_orig:>8,} params")
    print(f"  W-IE (ours)  : {n_wie:>8,} params")
    print(f"  Delta        : {delta:>+8,} params")
    print(f"  Within ±50K  : {'PASS' if within else 'FAIL'}")

    # ── Detailed W-IE breakdown ─────────────────────────────────────────  #
    print("\n  W-IE parameter breakdown:")
    for name, p in wie.named_parameters():
        print(f"    {name:45s} {str(tuple(p.shape)):20s} {p.numel():>6,}")
    print(f"    {'DWT/IDWT (register_buffer, no grad)':45s} {'':20s} {'0':>6}")

    # ── Summary ──────────────────────────────────────────────────────────  #
    print("\n" + "="*60)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print("="*60 + "\n")
    return all_ok


if __name__ == "__main__":
    import sys
    ok = run_verification()
    sys.exit(0 if ok else 1)
