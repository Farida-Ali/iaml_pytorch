"""
FD²RT v1 — Wavelet Illumination Estimator plugged into Retinexformer.
=====================================================================
This file is auto-discovered by basicsr/models/archs/__init__.py because
it ends with _arch.py.  It registers:

    FD2RT_Single_Stage   — Retinexformer single stage with W-IE
    FD2RT_V1             — Full model usable via config  network_g: type: FD2RT_V1

Design principle (Phase 1):
    Only the illumination estimator is replaced.  The Denoiser (U-Net with
    IGAB / IG-MSA blocks) is kept exactly as in Retinexformer so that any
    PSNR gain is attributable solely to the wavelet-domain illumination prior.

Inheritance chain:
    FD2RT_Single_Stage  ←  RetinexFormer_Single_Stage
    FD2RT_V1            ←  RetinexFormer

Only __init__ (to swap self.estimator) and forward (to unpack the new
3-tuple return of W-IE) are overridden.  All other methods are inherited.
"""

import sys
import os

# Make repo root (three levels up from basicsr/models/archs/) importable
# so wavelet_utils.py and fd2rt_arch.py can be found.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch.nn as nn

from basicsr.models.archs.RetinexFormer_arch import (
    RetinexFormer_Single_Stage,
    RetinexFormer,
)
from fd2rt_arch import WaveletIlluminationEstimator


# ---------------------------------------------------------------------------
# FD2RT_Single_Stage
# ---------------------------------------------------------------------------

class FD2RT_Single_Stage(RetinexFormer_Single_Stage):
    """
    One Retinexformer stage with the pixel-domain Illumination_Estimator
    replaced by WaveletIlluminationEstimator (W-IE).

    Differences from parent:
      __init__: self.estimator swapped to WaveletIlluminationEstimator
      forward:  unpacks (F_lu, I_lu, N_map) instead of (illu_fea, illu_map);
                I_lu is already computed inside W-IE as  img * illu_map + img
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_feat: int = 31,
        level: int = 2,
        num_blocks: list = None,
    ) -> None:
        if num_blocks is None:
            num_blocks = [1, 1, 1]
        # Call parent to build self.denoiser with the correct shape
        super().__init__(in_channels, out_channels, n_feat, level, num_blocks)
        # ── Replace estimator (parent built Illumination_Estimator) ─────── #
        # n_feat must match so F_lu channels align with what Denoiser expects.
        self.estimator = WaveletIlluminationEstimator(n_feat)

    def forward(self, img):
        """
        Args:
            img: [B, 3, H, W]  input low-light image
        Returns:
            output_img: [B, 3, H, W]  enhanced image
        """
        # W-IE returns three tensors; N_map is not used until Phase 2
        F_lu, I_lu, N_map = self.estimator(img)
        # I_lu = img * illu_map + img  (computed inside W-IE, identical formula)
        output_img = self.denoiser(I_lu, F_lu)
        return output_img


# ---------------------------------------------------------------------------
# FD2RT_V1
# ---------------------------------------------------------------------------

class FD2RT_V1(RetinexFormer):
    """
    FD²RT Phase 1 model: Retinexformer with wavelet illumination estimator.

    Config entry point:
        network_g:
          type: FD2RT_V1
          in_channels: 3
          out_channels: 3
          n_feat: 40
          stage: 1
          num_blocks: [1, 2, 2]

    All hyperparameters, optimizer, loss, and training schedule are shared
    with Retinexformer — only the illumination estimator differs.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_feat: int = 31,
        stage: int = 3,
        num_blocks: list = None,
    ) -> None:
        if num_blocks is None:
            num_blocks = [1, 1, 1]
        # Parent builds self.body as a Sequential of RetinexFormer_Single_Stage
        super().__init__(in_channels, out_channels, n_feat, stage, num_blocks)
        # ── Replace every stage with FD2RT_Single_Stage ─────────────────── #
        self.body = nn.Sequential(*[
            FD2RT_Single_Stage(
                in_channels=in_channels,
                out_channels=out_channels,
                n_feat=n_feat,
                level=2,
                num_blocks=num_blocks,
            )
            for _ in range(stage)
        ])

    # forward() is inherited from RetinexFormer:
    #   def forward(self, x):
    #       return self.body(x)
    # No override needed — body now contains FD2RT_Single_Stage modules.
