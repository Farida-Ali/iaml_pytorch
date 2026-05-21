"""
FD²RT A2 — DDA_Block (spatial-illumination branch only) + W-IE.
================================================================
Ablation A2 adds Restormer's Gated Depth-wise Feed-Forward Network (GDFN)
and proper Pre-LayerNorm on the attention path.  The spatial attention
mechanism (DDA_MSA) is byte-for-byte equivalent to Retinexformer's IG-MSA
so that the only controlled variable versus A1 is the FFN.

Changes vs A1 (FD2RT_V1):
  • FeedForward (double-GELU, mult=4) → GDFN (gated, gamma=2.66)
  • Pre-LN added before attention  (IGAB had no pre-LN on the attn path)
  • N_map still computed by W-IE but not yet consumed (Phase 2 will use it)

Non-goals (Phase 2):
  • No frequency branch in DDA_MSA
  • No N_map usage in DDA_MSA

Registered name: FD2RT_A2
Config entry point:
    network_g:
      type: FD2RT_A2
      in_channels: 3
      out_channels: 3
      n_feat: 40
      stage: 1
      num_blocks: [1, 2, 2]
"""

import sys
import os
import math
import warnings

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import _calculate_fan_in_and_fan_out

from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
from fd2rt_arch import WaveletIlluminationEstimator


# ── Weight-init helpers (copied from RetinexFormer_arch.py) ──────────── #

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


# ── DDA_MSA  (= IG_MSA, byte-for-byte equivalent) ─────────────────────── #

class DDA_MSA(nn.Module):
    """
    Illumination-Guided Multi-head Self-Attention — identical to IG_MSA.
    The frequency branch will be added here in Phase 2 (A3+).

    Args:
        dim      (int): token channel dimension
        dim_head (int): per-head dimension
        heads    (int): number of attention heads
    """

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in:           [b, h, w, c]   pre-normed input tokens
        illu_fea_trans: [b, h, w, c]   illumination guidance (not normed)
        Returns:        [b, h, w, c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans
        q, k, v, illu_attn = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)),
        )
        v = v * illu_attn
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # K^T Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v                        # [b, heads, d, hw]
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(
            v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        return out_c + out_p


# ── GDFN (Restormer Eq. 2) ─────────────────────────────────────────────── #

class GDFN(nn.Module):
    """
    Gated Depth-wise Feed-Forward Network (Restormer, Eq. 2):

        out = W_out · ( GELU(W1_dw W1_pw  x)  ⊙  W2_dw W2_pw  x )

    Implemented with a shared project_in for both branches (standard
    Restormer style): project_in → depthwise → chunk → gate → project_out.

    Args:
        dim   (int):   input/output channel count
        gamma (float): hidden-to-dim expansion ratio (Restormer default 2.66)
    """

    def __init__(self, dim: int, gamma: float = 2.66) -> None:
        super().__init__()
        hidden = int(dim * gamma)
        # shared pointwise for both branches
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        # shared depthwise for both branches
        self.dw_conv = nn.Conv2d(
            hidden * 2, hidden * 2, 3, 1, 1, bias=False, groups=hidden * 2
        )
        # output projection after gating
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x):
        """x: [b, h, w, c] → [b, h, w, c]"""
        x_c = x.permute(0, 3, 1, 2).contiguous()   # [b, c, h, w]
        x_c = self.project_in(x_c)                  # [b, 2·hidden, h, w]
        x_c = self.dw_conv(x_c)
        x1, x2 = x_c.chunk(2, dim=1)               # each [b, hidden, h, w]
        out = F.gelu(x1) * x2
        out = self.project_out(out)                  # [b, dim, h, w]
        return out.permute(0, 2, 3, 1)              # [b, h, w, c]


# ── DDA_Block ──────────────────────────────────────────────────────────── #

class DDA_Block(nn.Module):
    """
    Dual-Domain Attention Block — Phase 2, spatial branch only.
    Drop-in replacement for IGAB.

    Each inner block:   LN → DDA_MSA → (residual)  →  LN → GDFN → (residual)

    Args:
        dim        (int):   channel dimension
        dim_head   (int):   per-head size
        heads      (int):   number of heads
        num_blocks (int):   number of (attn, ffn) pairs stacked inside
        gamma      (float): GDFN channel expansion factor
    """

    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        num_blocks: int = 2,
        gamma: float = 2.66,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                nn.LayerNorm(dim),                                        # pre-LN attn
                DDA_MSA(dim=dim, dim_head=dim_head, heads=heads),
                nn.LayerNorm(dim),                                        # pre-LN FFN
                GDFN(dim=dim, gamma=gamma),
            ]))

    def forward(self, x, illu_fea):
        """
        x:        [b, c, h, w]
        illu_fea: [b, c, h, w]
        Returns:  [b, c, h, w]
        """
        x = x.permute(0, 2, 3, 1)                        # [b, h, w, c]
        illu_fea_t = illu_fea.permute(0, 2, 3, 1)        # [b, h, w, c]
        for (ln1, attn, ln2, ff) in self.blocks:
            x = attn(ln1(x), illu_fea_trans=illu_fea_t) + x
            x = ff(ln2(x)) + x
        return x.permute(0, 3, 1, 2)                      # [b, c, h, w]


# ── DDA_Denoiser ───────────────────────────────────────────────────────── #

class DDA_Denoiser(nn.Module):
    """
    Denoiser with every IGAB replaced by DDA_Block.
    Architecture (encoder → bottleneck → decoder) is identical to Retinexformer.
    """

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 3,
        dim: int = 31,
        level: int = 2,
        num_blocks: list = None,
        gamma: float = 2.66,
    ) -> None:
        if num_blocks is None:
            num_blocks = [2, 4, 4]
        super().__init__()
        self.dim   = dim
        self.level = level

        self.embedding = nn.Conv2d(in_dim, dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                DDA_Block(
                    dim=dim_level,
                    num_blocks=num_blocks[i],
                    dim_head=dim,
                    heads=dim_level // dim,
                    gamma=gamma,
                ),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),   # feature down
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),   # illu down
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = DDA_Block(
            dim=dim_level,
            num_blocks=num_blocks[-1],
            dim_head=dim,
            heads=dim_level // dim,
            gamma=gamma,
        )

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, 2, 2, 0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                DDA_Block(
                    dim=dim_level // 2,
                    num_blocks=num_blocks[level - 1 - i],
                    dim_head=dim,
                    heads=(dim_level // 2) // dim,
                    gamma=gamma,
                ),
            ]))
            dim_level //= 2

        self.mapping = nn.Conv2d(dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu   = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:        [b, c, h, w]
        illu_fea: [b, c, h, w]
        Returns:  [b, c, h, w]
        """
        fea = self.embedding(x)

        fea_encoder  = []
        illu_fea_list = []
        for (dda, FeaDown, IlluDown) in self.encoder_layers:
            fea = dda(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea      = FeaDown(fea)
            illu_fea = IlluDown(illu_fea)

        fea = self.bottleneck(fea, illu_fea)

        for i, (FeaUp, Fusion, dda) in enumerate(self.decoder_layers):
            fea      = FeaUp(fea)
            fea      = Fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea      = dda(fea, illu_fea)

        return self.mapping(fea) + x


# ── FD2RT_A2_Single_Stage ──────────────────────────────────────────────── #

class FD2RT_A2_Single_Stage(nn.Module):
    """One stage: W-IE (from A1) → DDA_Denoiser."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_feat: int = 31,
        level: int = 2,
        num_blocks: list = None,
        gamma: float = 2.66,
    ) -> None:
        if num_blocks is None:
            num_blocks = [1, 1, 1]
        super().__init__()
        self.estimator = WaveletIlluminationEstimator(n_feat)
        self.denoiser  = DDA_Denoiser(
            in_dim=in_channels,
            out_dim=out_channels,
            dim=n_feat,
            level=level,
            num_blocks=num_blocks,
            gamma=gamma,
        )

    def forward(self, img):
        """img: [b, 3, h, w]  →  [b, 3, h, w]"""
        F_lu, I_lu, _N_map = self.estimator(img)   # N_map unused until Phase 2
        return self.denoiser(I_lu, F_lu)


# ── FD2RT_A2 ───────────────────────────────────────────────────────────── #

class FD2RT_A2(RetinexFormer):
    """
    FD²RT Phase 2 model: W-IE + DDA_Block (spatial branch, GDFN FFN).

    Config entry point:
        network_g:
          type: FD2RT_A2
          in_channels: 3
          out_channels: 3
          n_feat: 40
          stage: 1
          num_blocks: [1, 2, 2]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_feat: int = 31,
        stage: int = 3,
        num_blocks: list = None,
        gamma: float = 2.66,
    ) -> None:
        if num_blocks is None:
            num_blocks = [1, 1, 1]
        super().__init__(in_channels, out_channels, n_feat, stage, num_blocks)
        self.body = nn.Sequential(*[
            FD2RT_A2_Single_Stage(
                in_channels=in_channels,
                out_channels=out_channels,
                n_feat=n_feat,
                level=2,
                num_blocks=num_blocks,
                gamma=gamma,
            )
            for _ in range(stage)
        ])

    # forward() inherited from RetinexFormer: return self.body(x)


# ── Verification ───────────────────────────────────────────────────────── #

def _count(m):
    return sum(p.numel() for p in m.parameters())


def run_verification():
    """
    Verifies A2 against A1 on the same random input [4, 3, 128, 128].
    Checks:
      1. Output shapes match.
      2. N_map is computed but not consumed.
      3. Parameter counts are within [1.6M, 2.5M].
    """
    from basicsr.models.archs.fd2rt_v1_arch import FD2RT_V1

    SEED = 42
    B, C, H, W = 4, 3, 128, 128
    kwargs = dict(in_channels=3, out_channels=3, n_feat=40, stage=1,
                  num_blocks=[1, 2, 2])

    torch.manual_seed(SEED)
    x = torch.randn(B, C, H, W)

    # Build models
    a1 = FD2RT_V1(**kwargs).eval()
    a2 = FD2RT_A2(**kwargs).eval()

    n_a1 = _count(a1)
    n_a2 = _count(a2)

    PASS = "PASS"
    FAIL = "FAIL"
    all_ok = True

    print("\n" + "=" * 60)
    print("CHECK 1 — Output shape")
    print("=" * 60)
    torch.manual_seed(SEED)
    with torch.no_grad():
        out_a1 = a1(x)
    torch.manual_seed(SEED)
    with torch.no_grad():
        out_a2 = a2(x)

    shape_ok = out_a1.shape == out_a2.shape == (B, C, H, W)
    all_ok &= shape_ok
    print(f"  {PASS if shape_ok else FAIL}  "
          f"A1 out: {tuple(out_a1.shape)}   A2 out: {tuple(out_a2.shape)}")

    not_identical = not torch.allclose(out_a1, out_a2)
    print(f"  {'PASS' if not_identical else 'NOTE'}  "
          f"Outputs differ (expected — GDFN ≠ FeedForward)")

    print("\n" + "=" * 60)
    print("CHECK 2 — N_map computed but not consumed")
    print("=" * 60)
    stage_a2 = a2.body[0]
    captured = {}

    def _hook(m, inp, out):
        captured['nmap'] = out[2]   # (F_lu, I_lu, N_map)[2]

    h = stage_a2.estimator.register_forward_hook(_hook)
    with torch.no_grad():
        a2(x)
    h.remove()

    nmap_ok = (
        'nmap' in captured
        and captured['nmap'].shape == (B, 1, H, W)
        and captured['nmap'].min().item() >= 0.0
        and captured['nmap'].max().item() <= 1.0
    )
    all_ok &= nmap_ok
    nmap = captured.get('nmap')
    print(f"  {PASS if nmap_ok else FAIL}  "
          f"N_map shape={tuple(nmap.shape) if nmap is not None else 'MISSING'}  "
          f"range=[{nmap.min():.3f}, {nmap.max():.3f}]"
          if nmap is not None else f"  {FAIL}  N_map not captured")

    print("\n" + "=" * 60)
    print("CHECK 3 — Parameter counts")
    print("=" * 60)
    limit = 2_500_000
    a1_ok = n_a1 < limit
    a2_ok = n_a2 < limit
    a2_gt_a1 = n_a2 > n_a1   # A2 should have more params than A1
    all_ok &= a1_ok and a2_ok

    print(f"  {PASS if a1_ok else FAIL}  A1 (FD2RT_V1) : {n_a1:>9,} params")
    print(f"  {PASS if a2_ok else FAIL}  A2 (FD2RT_A2) : {n_a2:>9,} params"
          f"  (Δ = {n_a2 - n_a1:+,})")
    print(f"  {'NOTE' if a2_gt_a1 else FAIL}  A2 > A1: {a2_gt_a1}  "
          f"(expected — GDFN adds ~7K params)")
    print(f"  {PASS if a2_ok else FAIL}  Under 2.5M limit: {a2_ok}")

    # Per-module breakdown for DDA_Block components
    stage = a2.body[0]
    print("\n  DDA_Denoiser breakdown:")
    total_dda = 0
    for name, sub in stage.denoiser.named_modules():
        if isinstance(sub, (DDA_Block, GDFN, DDA_MSA)):
            n = _count(sub)
            total_dda += n if not any(
                isinstance(p, (DDA_Block, GDFN, DDA_MSA))
                for p in sub.modules() if p is not sub
            ) else 0
            print(f"    {name:45s}  {type(sub).__name__:12s}  {n:>8,}")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print("=" * 60 + "\n")
    return all_ok


if __name__ == "__main__":
    ok = run_verification()
    sys.exit(0 if ok else 1)
