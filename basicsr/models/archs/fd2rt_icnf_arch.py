"""
FD²RT-ICNF — A4 with the Illumination-Conditioned Noise Floor.

WHAT CHANGES VS A4
──────────────────
A4 has two mechanisms that are structurally sound but empirically ungrounded:

  1. `N_map` — a 297-parameter branch off the HH subband, trained with no
     supervision beyond the final L1. Nothing makes it encode noise; it is just
     free capacity that happens to sit where a noise map would go.

  2. `gates[i]` — one learned SCALAR per block, mixing the frequency branch in
     globally. A single scalar cannot express "trust high frequencies HERE but
     not THERE", which is exactly the decision the block needs to make, because
     in a low-light frame the same absolute high-frequency energy means texture
     in a lit region and pure noise in a shadow.

Both are replaced by one analytic quantity (see archs/icnf.py):

    evidence(x) = softplus( (E_hf(x) − (a·I(x) + b)) / (a·I(x) + b + eps) )

  * `N_map` -> the ICNF evidence map (2 parameters instead of 297, and it
    provably tracks texture: AUC 0.83 vs 0.60 for raw HF energy under a 19:1
    illumination range).
  * scalar `gates[i]` -> a per-pixel `ICNFGate` driven by that evidence.

WHICH SIGNAL CONDITIONS THE FLOOR
─────────────────────────────────
Photon statistics make the noise variance affine in the *observed signal
level*: var = a·I + b. In a Retinex decomposition I = L·R, and because
reflectance is high-frequency its local mean is roughly constant, so the
spatially varying part of the floor is dominated by the illumination L. That is
why the measured advantage grows with illumination range (+0.023 at 1:1,
+0.227 at 19:1) — the effect really is illumination-driven.

`illum_source` selects what estimates I:
  'input'    local mean of the observed image (default; what the feasibility
             study measured, so reported numbers are not oracle-inflated)
  'illu_map' the W-IE's learned illumination map — the tighter Retinex
             coupling, kept as an ablation arm rather than assumed better.

AT INITIALISATION
─────────────────
A4 zero-initialises Freq_MSA.out_proj so its frequency residual B is exactly 0
at step 0. That cannot be kept here: with B == 0, dL/d(gate) = dL/dx * B = 0, so
the gate — and ICNF's sensor parameters upstream of it — would receive exactly
zero gradient and the mechanism could never start. out_proj therefore gets a
small nonzero init (std 1e-3) instead, which is what preserves near-A1
behaviour; forcing the frequency branch off at init still changes the output by
under 1e-5 relative. The gate is then free to start partly open (0.20), which
buys 13x more gradient to the mechanism than a shut gate.

A4 checkpoints load with strict=False; only the `gates.*` keys differ (scalar
Parameter -> slope/bias pair).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
from basicsr.models.archs.fd2rt_a4_arch import (
    DDA_Block_Dual, DDA_Denoiser_Dual, FD2RT_A4_Single_Stage,
)
from basicsr.models.archs.icnf import ICNF, ICNFGate
from fd2rt_arch import WaveletIlluminationEstimator


# ── ICNF_Block_Dual ────────────────────────────────────────────────────── #

class ICNF_Block_Dual(DDA_Block_Dual):
    """DDA_Block_Dual with a spatially-varying, evidence-driven gate.

    Adopts an existing DDA_Block_Dual's submodules so that the `blocks.*` and
    `freq_blocks.*` checkpoint keys — and their initialisation, including the
    zero-init frequency projection — are preserved exactly. Only `gates.*`
    changes shape, from a scalar Parameter to an ICNFGate's (slope, bias).
    """

    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2,
                 gate_slope=1.0, gate_bias=2.0):
        super().__init__(dim=dim, dim_head=dim_head, heads=heads,
                         num_blocks=num_blocks)
        # Swap the scalar ParameterList for per-pixel gates.
        n = len(self.blocks)
        del self.gates
        self.gates = nn.ModuleList(
            [ICNFGate(slope_init=gate_slope, bias_init=gate_bias)
             for _ in range(n)])

    @classmethod
    def from_dda(cls, dda, gate_slope=1.0, gate_bias=2.0, freq_init_std=1e-3,
                 gate_per_channel=False):
        """Build from an existing DDA_Block_Dual, transplanting its submodules.

        IDENTIFIABILITY FIX. A4 zero-initialises Freq_MSA.out_proj so the
        frequency residual B is exactly 0 at step 0 and the model reproduces A1.
        That is fine when the mixing coefficient is a scalar the optimiser can
        leave alone, but here the gate IS the contribution: with B == 0,

            dL/d(gate) = dL/dx * B = 0

        so the gate — and therefore ICNF's sensor parameters (a, b) upstream of
        it — receive exactly zero gradient at initialisation and the whole
        mechanism can never start learning. scripts/test_p2_icnf_arch.py caught
        this as |grad| = 0.000000 on both groups.

        We keep the A1-like starting behaviour a different way: the GATE stays
        closed (~0.02), while out_proj gets a small nonzero init so B != 0 and
        the gate is identifiable from the first step. The contribution at init
        is gate * B ~ 0.02 * small, still negligible.
        """
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.blocks = dda.blocks              # keys blocks.* preserved
        obj.freq_blocks = dda.freq_blocks    # keys freq_blocks.* preserved
        # Infer feature width from the transplanted attention block so the
        # per-channel gate emits exactly one value per feature channel.
        # freq_blocks[i].out_proj is Conv2d(dim, dim, 1), so its weight's first
        # dim IS the feature width. (Taking the attention block's first
        # parameter instead gives the fused qkv width, which is a multiple of
        # dim and produces a broadcast error at the fusion.)
        n_ch = 1
        if gate_per_channel:
            n_ch = int(dda.freq_blocks[0].out_proj.weight.shape[0])
        obj.gates = nn.ModuleList(
            [ICNFGate(slope_init=gate_slope, bias_init=gate_bias,
                      out_channels=n_ch)
             for _ in range(len(dda.blocks))])

        for fb in obj.freq_blocks:
            if hasattr(fb, 'out_proj') and freq_init_std > 0:
                nn.init.normal_(fb.out_proj.weight, mean=0.0, std=freq_init_std)
        return obj

    def forward(self, x, illu_fea, evidence):
        """
        x, illu_fea: [b, c, h, w]
        evidence:    [b, 1, H, W]  ICNF texture evidence at input resolution
        """
        x = x.permute(0, 2, 3, 1)                          # [b, h, w, c]
        illu_fea_t = illu_fea.permute(0, 2, 3, 1)

        h, w = x.shape[1], x.shape[2]
        # Resize evidence to this pyramid level once per block.
        ev = evidence if evidence.shape[-2:] == (h, w) else F.interpolate(
            evidence, size=(h, w), mode='bilinear', align_corners=False)

        for i, (attn, ff) in enumerate(self.blocks):
            # Per-pixel gate in (0,1): open where texture rises above the
            # illumination-predicted noise floor, closed where it does not.
            gate = self.gates[i](ev)                       # [b, 1, h, w]
            gate = gate.permute(0, 2, 3, 1)                # [b, h, w, 1]

            A = attn(x, illu_fea_trans=illu_fea_t)

            x_c = x.permute(0, 3, 1, 2).contiguous()
            B_c = self.freq_blocks[i](x_c, ev)             # evidence replaces N_map
            B = B_c.permute(0, 2, 3, 1)

            x = x + A + gate * B                            # spatially fused
            x = ff(x) + x

        return x.permute(0, 3, 1, 2)


# ── ICNF_Denoiser ──────────────────────────────────────────────────────── #

class ICNF_Denoiser(DDA_Denoiser_Dual):
    """DDA_Denoiser_Dual whose blocks use evidence-driven spatial gates."""

    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=None,
                 gate_slope=1.0, gate_bias=2.0, freq_init_std=1e-3,
                 gate_per_channel=False):
        super().__init__(in_dim=in_dim, out_dim=out_dim, dim=dim,
                         level=level, num_blocks=num_blocks)
        conv = lambda b: ICNF_Block_Dual.from_dda(
            b, gate_slope, gate_bias, freq_init_std, gate_per_channel)
        for lyr in self.encoder_layers:
            lyr[0] = conv(lyr[0])
        self.bottleneck = conv(self.bottleneck)
        for lyr in self.decoder_layers:
            lyr[2] = conv(lyr[2])


# ── Single stage ───────────────────────────────────────────────────────── #

class FD2RT_ICNF_Single_Stage(nn.Module):
    """W-IE for illumination; ICNF for the noise floor; evidence gates the
    frequency branch at every block."""

    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2,
                 num_blocks=None, illum_source='input', icnf_mode='subtract',
                 a_init=0.02, b_init=1e-5, learn_sensor=True,
                 icnf_window=9, gate_slope=1.0, gate_bias=2.0,
                 freq_init_std=1e-3, gate_per_channel=False):
        super().__init__()
        if num_blocks is None:
            num_blocks = [1, 1, 1]
        if illum_source not in ('input', 'illu_map', 'constant'):
            raise ValueError(f"illum_source must be 'input', 'illu_map' or "
                             f"'constant', got {illum_source}")
        self.illum_source = illum_source

        self.estimator = WaveletIlluminationEstimator(n_feat)
        self.icnf = ICNF(a_init=a_init, b_init=b_init,
                         learn_params=learn_sensor, window=icnf_window,
                         mode=icnf_mode)
        self.denoiser = ICNF_Denoiser(
            in_dim=in_channels, out_dim=out_channels, dim=n_feat,
            level=level, num_blocks=num_blocks,
            gate_slope=gate_slope, gate_bias=gate_bias,
            freq_init_std=freq_init_std, gate_per_channel=gate_per_channel)

    def forward(self, img):
        # W-IE still supplies illumination features; its N_map output is no
        # longer routed anywhere — ICNF replaces it.
        F_lu, I_lu, _ = self.estimator(img)

        if self.illum_source == 'illu_map':
            # illu_map = I_lu / img - 1, recovered without a second forward.
            illum = (I_lu / img.clamp_min(1e-4) - 1.0).clamp(0, 10)
        elif self.illum_source == 'constant':
            # ABLATION CONTROL. Replace the spatially-varying illumination with
            # its per-image global mean, so the noise floor a*I + b is UNIFORM.
            # The gate is still per-pixel (driven by the spatial variation of the
            # HF energy E), but it is no longer illumination-CONDITIONED. This
            # isolates the causal question the mismatched-noise pilot raised: if
            # ICNF's advantage survives when the exact noise law is wrong, is it
            # coming from conditioning the floor on illumination at all, or purely
            # from having a per-pixel adaptive gate? ICNF-vs-constant answers it.
            lm = img.mean(dim=1, keepdim=True)                 # [B,1,H,W]
            illum = lm.mean(dim=(2, 3), keepdim=True).expand_as(lm)
        else:
            illum = None            # ICNF falls back to a local mean of img

        evidence = self.icnf(img, illum=illum)
        return self.denoiser(I_lu, F_lu, evidence)


# ── Top-level model ────────────────────────────────────────────────────── #

class FD2RT_ICNF(RetinexFormer):
    """FD²RT with the Illumination-Conditioned Noise Floor.

    Config:
        network_g:
          type: FD2RT_ICNF
          in_channels: 3
          out_channels: 3
          n_feat: 40
          stage: 1
          num_blocks: [1, 2, 2]
          illum_source: input        # or illu_map (ablation arm)
          icnf_mode: subtract        # or ratio   (ablation arm)
    """

    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3,
                 num_blocks=None, illum_source='input', icnf_mode='subtract',
                 a_init=0.02, b_init=1e-5, learn_sensor=True, icnf_window=9,
                 gate_slope=1.0, gate_bias=2.0, freq_init_std=1e-3,
                 gate_per_channel=False):
        if num_blocks is None:
            num_blocks = [1, 1, 1]
        nn.Module.__init__(self)
        self.stage = stage
        self.body = nn.Sequential(*[
            FD2RT_ICNF_Single_Stage(
                in_channels=in_channels, out_channels=out_channels,
                n_feat=n_feat, level=2, num_blocks=num_blocks,
                illum_source=illum_source, icnf_mode=icnf_mode,
                a_init=a_init, b_init=b_init, learn_sensor=learn_sensor,
                icnf_window=icnf_window, gate_slope=gate_slope,
                gate_bias=gate_bias, freq_init_std=freq_init_std,
                gate_per_channel=gate_per_channel)
            for _ in range(stage)
        ])

    def forward(self, x):
        return self.body(x)
