"""
FD²RT A2 — DDA_Block (spatial branch, original FFN) + W-IE.
============================================================
GDFN has been dropped in favour of Retinexformer's original FeedForward.
DDA_MSA is byte-for-byte identical to IG_MSA.  DDA_Block is therefore
structurally identical to IGAB — same nn.ModuleList layout, same key
names, same forward pass — making it a verified drop-in replacement
that loads cleanly from an A1 (FD2RT_V1 / W-IE + IGAB) checkpoint.

Key layout (matches IGAB exactly):
  blocks[i][0]  DDA_MSA       ≡ IG_MSA
  blocks[i][1]  PreNorm(FFN)  ≡ PreNorm(FeedForward)

Registered name: FD2RT_A2
"""

import sys, os, math, warnings
from einops import rearrange
from torch.nn.init import _calculate_fan_in_and_fan_out

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.RetinexFormer_arch import (
    RetinexFormer,
    PreNorm,
    FeedForward,
    GELU,
)
from fd2rt_arch import WaveletIlluminationEstimator


# ── Weight-init (same as Retinexformer) ────────────────────────────────── #

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b].", stacklevel=2)
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


# ── DDA_MSA  (= IG_MSA, byte-for-byte copy) ────────────────────────────── #

class DDA_MSA(nn.Module):
    """Spatial self-attention — identical to IG_MSA in every detail."""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.num_heads = heads
        self.dim_head  = dim_head
        self.to_q  = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k  = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v  = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj  = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """x_in: [b,h,w,c]  illu_fea_trans: [b,h,w,c]  →  [b,h,w,c]"""
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
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(
            v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        return out_c + out_p


# ── DDA_Block ──────────────────────────────────────────────────────────── #

class DDA_Block(nn.Module):
    """
    IGAB-equivalent with DDA_MSA.  Module layout is byte-for-byte identical
    to IGAB so checkpoint keys are interchangeable:
      blocks[i][0]  DDA_MSA           (same keys as IG_MSA)
      blocks[i][1]  PreNorm(FFN)      (same keys as PreNorm(FeedForward))
    """

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8,
                 num_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                DDA_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim)),
            ]))

    def forward(self, x, illu_fea):
        """x: [b,c,h,w]  illu_fea: [b,c,h,w]  →  [b,c,h,w]"""
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


# ── DDA_Denoiser ───────────────────────────────────────────────────────── #

class DDA_Denoiser(nn.Module):
    """Denoiser with every IGAB replaced by DDA_Block (structurally identical)."""

    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2,
                 num_blocks=None) -> None:
        if num_blocks is None:
            num_blocks = [2, 4, 4]
        super().__init__()
        self.dim   = dim
        self.level = level

        self.embedding = nn.Conv2d(in_dim, dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList()
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                DDA_Block(dim=dim_level, num_blocks=num_blocks[i],
                          dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
            ]))
            dim_level *= 2

        self.bottleneck = DDA_Block(
            dim=dim_level, num_blocks=num_blocks[-1],
            dim_head=dim, heads=dim_level // dim,
        )

        self.decoder_layers = nn.ModuleList()
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, 2, 2, 0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                DDA_Block(dim=dim_level // 2,
                          num_blocks=num_blocks[level - 1 - i],
                          dim_head=dim, heads=(dim_level // 2) // dim),
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
        fea = self.embedding(x)
        fea_encoder   = []
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
    def __init__(self, in_channels=3, out_channels=3, n_feat=31,
                 level=2, num_blocks=None) -> None:
        if num_blocks is None:
            num_blocks = [1, 1, 1]
        super().__init__()
        self.estimator = WaveletIlluminationEstimator(n_feat)
        self.denoiser  = DDA_Denoiser(in_dim=in_channels, out_dim=out_channels,
                                      dim=n_feat, level=level,
                                      num_blocks=num_blocks)

    def forward(self, img):
        F_lu, I_lu, _N_map = self.estimator(img)
        return self.denoiser(I_lu, F_lu)


# ── FD2RT_A2 ───────────────────────────────────────────────────────────── #

class FD2RT_A2(RetinexFormer):
    """
    FD²RT A2: W-IE + DDA_Block (spatial branch, original FFN).
    Checkpoint keys are identical to A1 (FD2RT_V1) — weights load without
    any remapping.

    Config:
        network_g:
          type: FD2RT_A2
          in_channels: 3
          out_channels: 3
          n_feat: 40
          stage: 1
          num_blocks: [1, 2, 2]
    """

    def __init__(self, in_channels=3, out_channels=3, n_feat=31,
                 stage=3, num_blocks=None) -> None:
        if num_blocks is None:
            num_blocks = [1, 1, 1]
        super().__init__(in_channels, out_channels, n_feat, stage, num_blocks)
        self.body = nn.Sequential(*[
            FD2RT_A2_Single_Stage(
                in_channels=in_channels, out_channels=out_channels,
                n_feat=n_feat, level=2, num_blocks=num_blocks,
            )
            for _ in range(stage)
        ])


# ── Verification ───────────────────────────────────────────────────────── #

CKPT_PATH = (
    '/root/.claude/uploads/'
    '65d5d802-aad9-4807-87d2-a69bf439e318/'
    'e8684ae8-best_psnr_23.71_126000.pth'
)


def run_verification():
    from basicsr.models.archs.fd2rt_v1_arch import FD2RT_V1

    kwargs = dict(in_channels=3, out_channels=3, n_feat=40,
                  stage=1, num_blocks=[1, 2, 2])

    PASS = 'PASS'; FAIL = 'FAIL'
    all_ok = True

    # ── Check 1: Key-set identity ─────────────────────────────────────── #
    print('\n' + '=' * 60)
    print('CHECK 1 — State-dict keys match A1 checkpoint exactly')
    print('=' * 60)

    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    ckpt_keys = set(ckpt.get('params', ckpt).keys())

    a2 = FD2RT_A2(**kwargs)
    a2_keys = set(a2.state_dict().keys())

    only_in_ckpt = ckpt_keys - a2_keys
    only_in_a2   = a2_keys   - ckpt_keys
    keys_ok = (len(only_in_ckpt) == 0 and len(only_in_a2) == 0)
    all_ok &= keys_ok

    print(f'  {PASS if keys_ok else FAIL}  '
          f'Keys only in checkpoint : {len(only_in_ckpt)}')
    print(f'  {PASS if keys_ok else FAIL}  '
          f'Keys only in A2 model   : {len(only_in_a2)}')
    if not keys_ok:
        for k in sorted(only_in_ckpt)[:5]:
            print(f'    CKPT only: {k}')
        for k in sorted(only_in_a2)[:5]:
            print(f'    A2 only : {k}')

    # ── Check 2: Numerical equivalence ────────────────────────────────── #
    print('\n' + '=' * 60)
    print('CHECK 2 — Forward pass matches FD2RT_V1 (same weights, same input)')
    print('=' * 60)

    state = ckpt.get('params', ckpt)

    a1 = FD2RT_V1(**kwargs).eval()
    a1.load_state_dict(state)

    a2.load_state_dict(state)
    a2.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out_a1 = a1(x)
        out_a2 = a2(x)

    max_diff = (out_a1 - out_a2).abs().max().item()
    equiv_ok = max_diff < 1e-5
    all_ok &= equiv_ok
    print(f'  {PASS if equiv_ok else FAIL}  '
          f'Max absolute diff A1 vs A2: {max_diff:.2e}  '
          f'(threshold 1e-5)')

    # ── Check 3: Parameter count ──────────────────────────────────────── #
    print('\n' + '=' * 60)
    print('CHECK 3 — Parameter count')
    print('=' * 60)
    n_a1 = sum(p.numel() for p in a1.parameters())
    n_a2 = sum(p.numel() for p in a2.parameters())
    same_count = (n_a1 == n_a2)
    all_ok &= same_count
    print(f'  {PASS if same_count else FAIL}  '
          f'A1 params: {n_a1:,}  A2 params: {n_a2:,}  '
          f'delta: {n_a2 - n_a1:+d}')

    print('\n' + '=' * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print('=' * 60 + '\n')
    return all_ok


if __name__ == '__main__':
    ok = run_verification()
    import sys; sys.exit(0 if ok else 1)
