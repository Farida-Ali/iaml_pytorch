"""
FD²RT A4 — W-IE + Dual-Domain DDA_Block (spatial + frequency branches).
========================================================================
Architecture changes vs A2:
  • DDA_Block_Dual adds a Freq_MSA branch alongside the existing DDA_MSA
  • Freq_MSA: Haar DWT on features → attend over LL⊕HH → N_map modulates V
              → IDWT(delta_LL, 0, 0, 0) → spatial residual
  • Zero-initialized out_proj in Freq_MSA: model is identical to A1 at init
  • Adaptive gate (scalar per inner block) fuses spatial + frequency residuals

Key layout (preserves A1 / A2 checkpoint keys):
  blocks[i][0]   DDA_MSA          (same keys as IG_MSA — loads from A1 ckpt)
  blocks[i][1]   PreNorm(FFN)     (same keys as PreNorm(FF) — loads from A1 ckpt)
  freq_blocks[i] Freq_MSA         (new keys — initialized fresh, out_proj=zeros)
  gates[i]       scalar Parameter (new key — initialized to 0, sigmoid(0)=0.5)

At init: Freq_MSA output ≈ 0 → gate * 0 = 0 → model behaviour ≡ A1 exactly.

Registered name: FD2RT_A4
Training config: Options/train_FD2RT_A4_LOL_v1.yml
"""

import sys, os, math, warnings
from einops import rearrange

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.RetinexFormer_arch import (
    RetinexFormer, PreNorm, FeedForward, GELU,
)
from fd2rt_arch import WaveletIlluminationEstimator
from wavelet_utils import HaarDWT2D, HaarIDWT2D


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


# ── DDA_MSA  (= IG_MSA, spatial branch) ───────────────────────────────── #

class DDA_MSA(nn.Module):
    """Spatial self-attention — identical to IG_MSA."""

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
        """x_in, illu_fea_trans: [b,h,w,c]  →  [b,h,w,c]"""
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


# ── Freq_MSA (frequency branch) ────────────────────────────────────────── #

class Freq_MSA(nn.Module):
    """
    Frequency-domain attention branch.

    Forward path:
      1. Haar DWT on input features [b,c,h,w] → LL, LH, HL, HH
      2. Concatenate LL⊕HH (illumination + noise subbands), project to c channels
      3. Transposed attention (same K^T Q mechanism as IG_MSA) over hd×wd tokens
         with N_map modulating value elements (same role as illu_fea in IG_MSA)
      4. Zero-initialized out_proj produces delta_LL ≈ 0 at training start
      5. IDWT(delta_LL, 0, 0, 0) = pure low-freq residual in spatial domain
         LH, HL, HH are passed through unchanged (only LL is modified)

    Output is [b,c,h,w] residual ≈ 0 at init.
    """

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.num_heads = heads
        self.dim_head  = dim_head

        self.dwt  = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        # LL (dim) + HH (dim) → dim
        self.freq_in = nn.Conv2d(2 * dim, dim, 1, bias=False)

        # Attention projections (transposed attention, same as IG_MSA)
        self.to_q    = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k    = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v    = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj    = nn.Linear(dim_head * heads, dim, bias=True)

        # N_map [b,1,H,W] → [b, dim_head*heads, hd, wd] for V modulation
        self.nmap_proj = nn.Conv2d(1, dim_head * heads, 1, bias=False)

        # Zero-initialized: freq branch output is 0 at training start
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, N_map: torch.Tensor) -> torch.Tensor:
        """
        x:     [b, c, h, w]  feature map (must have even spatial dims)
        N_map: [b, 1, H, W]  noise-awareness map (full resolution from W-IE)
        Returns: [b, c, h, w] residual (≈ 0 at init due to zero out_proj)
        """
        b, c, h, w = x.shape

        # 1. DWT
        LL, LH, HL, HH = self.dwt(x)              # each [b, c, hd, wd]
        hd, wd = LL.shape[2], LL.shape[3]

        # 2. Downsample N_map to freq-domain resolution
        N_small = F.interpolate(N_map, size=(hd, wd), mode='bilinear',
                                align_corners=False)  # [b, 1, hd, wd]

        # 3. Combine LL + HH → project to c channels
        freq_feat = self.freq_in(torch.cat([LL, HH], dim=1))  # [b, c, hd, wd]

        # 4. Flatten to tokens [b, hd*wd, c]
        tokens = freq_feat.flatten(2).transpose(1, 2)
        q_inp = self.to_q(tokens)
        k_inp = self.to_k(tokens)
        v_inp = self.to_v(tokens)

        # 5. N_map as value modulation — same role as illu_fea in IG_MSA
        nmap_v = self.nmap_proj(N_small).flatten(2).transpose(1, 2)
        # [b, hd*wd, dim_head*heads]

        q, k, v, nmap_v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
            (q_inp, k_inp, v_inp, nmap_v),
        )
        v = v * nmap_v                             # modulate before transposing

        # 6. Transposed attention (O(d²·n) not O(n²·d)) — identical to IG_MSA
        q = q.transpose(-2, -1)                    # [b, heads, dim_head, hd*wd]
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1)) * self.rescale   # [b, heads, d, d]
        attn = attn.softmax(dim=-1)
        out  = attn @ v                            # [b, heads, d, hd*wd]

        out = out.permute(0, 3, 1, 2).reshape(b, hd * wd,
                                               self.num_heads * self.dim_head)
        out = self.proj(out)                       # [b, hd*wd, c]
        out = out.transpose(1, 2).reshape(b, c, hd, wd)   # [b, c, hd, wd]

        # 7. Zero-init delta on LL; IDWT propagates it to spatial domain
        delta_LL = self.out_proj(out)              # ≈ 0 at init
        zeros    = torch.zeros_like(LH)
        residual = self.idwt(delta_LL, zeros, zeros, zeros)  # [b, c, h, w]
        return residual                            # ≈ 0 at init ✓


# ── DDA_Block_Dual ─────────────────────────────────────────────────────── #

class DDA_Block_Dual(nn.Module):
    """
    Dual-domain DDA block: spatial (DDA_MSA) + frequency (Freq_MSA) branches.

    Module layout is designed to preserve A1/A2 checkpoint key names exactly:
      self.blocks[i][0]   = DDA_MSA      → keys: blocks.i.0.*  (matches A1)
      self.blocks[i][1]   = PreNorm(FFN) → keys: blocks.i.1.*  (matches A1)
      self.freq_blocks[i] = Freq_MSA     → keys: freq_blocks.i.*  (new)
      self.gates[i]       = scalar param → keys: gates.i  (new)

    Fusion: x ← x + A_spatial + sigmoid(gate) · A_freq
    At init: A_freq ≈ 0 → model reduces to A1 regardless of gate value.
    """

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8,
                 num_blocks: int = 2) -> None:
        super().__init__()
        self.blocks      = nn.ModuleList()
        self.freq_blocks = nn.ModuleList()
        self.gates       = nn.ParameterList()

        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                DDA_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim)),
            ]))
            self.freq_blocks.append(
                Freq_MSA(dim=dim, dim_head=dim_head, heads=heads)
            )
            self.gates.append(nn.Parameter(torch.zeros(1)))

    def forward(self, x, illu_fea, N_map):
        """
        x:        [b, c, h, w]
        illu_fea: [b, c, h, w]
        N_map:    [b, 1, H, W]   full-resolution noise map from W-IE
        Returns:  [b, c, h, w]
        """
        x = x.permute(0, 2, 3, 1)                         # [b, h, w, c]
        illu_fea_t = illu_fea.permute(0, 2, 3, 1)         # [b, h, w, c]

        for i, (attn, ff) in enumerate(self.blocks):
            gate = torch.sigmoid(self.gates[i])

            # Branch A: spatial illumination-guided attention
            A = attn(x, illu_fea_trans=illu_fea_t)        # [b, h, w, c]

            # Branch B: frequency-domain attention (≈ 0 at init)
            x_c = x.permute(0, 3, 1, 2).contiguous()      # [b, c, h, w]
            B_c = self.freq_blocks[i](x_c, N_map)         # [b, c, h, w]
            B   = B_c.permute(0, 2, 3, 1)                 # [b, h, w, c]

            x = x + A + gate * B                           # fused residual
            x = ff(x) + x                                  # FFN

        return x.permute(0, 3, 1, 2)                       # [b, c, h, w]


# ── DDA_Denoiser_Dual ──────────────────────────────────────────────────── #

class DDA_Denoiser_Dual(nn.Module):
    """
    Denoiser with DDA_Block_Dual at every position.
    N_map is passed at full resolution; Freq_MSA downsamples it internally.
    """

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
                DDA_Block_Dual(dim=dim_level, num_blocks=num_blocks[i],
                               dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
            ]))
            dim_level *= 2

        self.bottleneck = DDA_Block_Dual(
            dim=dim_level, num_blocks=num_blocks[-1],
            dim_head=dim, heads=dim_level // dim,
        )

        self.decoder_layers = nn.ModuleList()
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, 2, 2, 0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                DDA_Block_Dual(dim=dim_level // 2,
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

    def forward(self, x, illu_fea, N_map):
        """
        x, illu_fea: [b, c, h, w]
        N_map:       [b, 1, H, W]   full-resolution noise map
        """
        fea = self.embedding(x)
        fea_encoder   = []
        illu_fea_list = []

        for (dda, FeaDown, IlluDown) in self.encoder_layers:
            fea = dda(fea, illu_fea, N_map)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea      = FeaDown(fea)
            illu_fea = IlluDown(illu_fea)

        fea = self.bottleneck(fea, illu_fea, N_map)

        for i, (FeaUp, Fusion, dda) in enumerate(self.decoder_layers):
            fea      = FeaUp(fea)
            fea      = Fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea      = dda(fea, illu_fea, N_map)

        return self.mapping(fea) + x


# ── FD2RT_A4_Single_Stage ──────────────────────────────────────────────── #

class FD2RT_A4_Single_Stage(nn.Module):
    """W-IE extracts N_map; DDA_Denoiser_Dual routes it to every Freq_MSA."""

    def __init__(self, in_channels=3, out_channels=3, n_feat=31,
                 level=2, num_blocks=None) -> None:
        if num_blocks is None:
            num_blocks = [1, 1, 1]
        super().__init__()
        self.estimator = WaveletIlluminationEstimator(n_feat)
        self.denoiser  = DDA_Denoiser_Dual(
            in_dim=in_channels, out_dim=out_channels,
            dim=n_feat, level=level, num_blocks=num_blocks,
        )

    def forward(self, img):
        """img: [b, 3, h, w]  →  [b, 3, h, w]"""
        F_lu, I_lu, N_map = self.estimator(img)   # N_map now actively used
        return self.denoiser(I_lu, F_lu, N_map)


# ── FD2RT_A4 ───────────────────────────────────────────────────────────── #

class FD2RT_A4(RetinexFormer):
    """
    FD²RT A4: W-IE + Dual-Domain DDA Block (spatial + frequency) + original FFN.

    Config:
        network_g:
          type: FD2RT_A4
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
            FD2RT_A4_Single_Stage(
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


def _count(m):
    return sum(p.numel() for p in m.parameters())


def run_verification():
    from basicsr.models.archs.fd2rt_v1_arch import FD2RT_V1

    kwargs = dict(in_channels=3, out_channels=3, n_feat=40,
                  stage=1, num_blocks=[1, 2, 2])
    PASS = 'PASS'; FAIL = 'FAIL'
    all_ok = True

    ckpt  = torch.load(CKPT_PATH, map_location='cpu')
    state = ckpt.get('params', ckpt)

    # ── CHECK 1: Weight loading ─────────────────────────────────────────── #
    print('\n' + '=' * 60)
    print('CHECK 1 — Load A1 checkpoint into A4 (strict=False)')
    print('=' * 60)
    a4 = FD2RT_A4(**kwargs)
    missing, unexpected = a4.load_state_dict(state, strict=False)

    # Keys that exist in A1 but NOT in A4 are truly unexpected
    new_keys_in_a4   = [k for k in missing    if 'freq_blocks' in k or 'gates' in k]
    ffn_blocks_match = [k for k in missing    if 'freq_blocks' not in k and 'gates' not in k]
    unexp_in_a4      = unexpected

    load_ok = len(ffn_blocks_match) == 0 and len(unexp_in_a4) == 0
    all_ok &= load_ok

    print(f'  {PASS if load_ok else FAIL}  '
          f'All A1 keys loaded (no unexpected, no missing A1 keys)')
    print(f'  INFO  New A4-only keys (fresh init): {len(new_keys_in_a4)}')
    print(f'        freq_blocks keys : '
          f'{sum(1 for k in new_keys_in_a4 if "freq_blocks" in k)}')
    print(f'        gate keys        : '
          f'{sum(1 for k in new_keys_in_a4 if "gates" in k)}')
    if ffn_blocks_match:
        print(f'  {FAIL}  Missing A1 keys in A4:')
        for k in ffn_blocks_match[:5]:
            print(f'    {k}')
    if unexp_in_a4:
        print(f'  {FAIL}  Unexpected keys:')
        for k in unexp_in_a4[:5]:
            print(f'    {k}')

    # ── CHECK 2: Freq branch is zero at init ────────────────────────────── #
    print('\n' + '=' * 60)
    print('CHECK 2 — Freq branch output ≈ 0 at init (zero out_proj)')
    print('=' * 60)
    freq_outputs = {}

    def _make_hook(name):
        def hook(m, inp, out):
            freq_outputs[name] = out.detach().norm().item()
        return hook

    hooks = []
    for name, m in a4.named_modules():
        if isinstance(m, Freq_MSA):
            hooks.append(m.register_forward_hook(_make_hook(name)))

    torch.manual_seed(0)
    x = torch.randn(1, 3, 128, 128)
    a4.eval()
    with torch.no_grad():
        _ = a4(x)
    for h in hooks:
        h.remove()

    max_freq_norm = max(freq_outputs.values()) if freq_outputs else float('inf')
    freq_zero_ok  = max_freq_norm < 1e-6
    all_ok &= freq_zero_ok
    print(f'  {PASS if freq_zero_ok else FAIL}  '
          f'Max freq-branch output norm: {max_freq_norm:.2e}  (threshold 1e-6)')

    # ── CHECK 3: Forward equivalence with A1 ────────────────────────────── #
    print('\n' + '=' * 60)
    print('CHECK 3 — A4 output matches A1 with same weights (freq branch = 0)')
    print('=' * 60)
    a1 = FD2RT_V1(**kwargs).eval()
    a1.load_state_dict(state)

    with torch.no_grad():
        out_a1 = a1(x)
        out_a4 = a4(x)

    max_diff = (out_a1 - out_a4).abs().max().item()
    equiv_ok = max_diff < 1e-5
    all_ok &= equiv_ok
    print(f'  {PASS if equiv_ok else FAIL}  '
          f'Max absolute diff A1 vs A4: {max_diff:.2e}  (threshold 1e-5)')

    # ── CHECK 4: Parameter count ─────────────────────────────────────────── #
    print('\n' + '=' * 60)
    print('CHECK 4 — Parameter count')
    print('=' * 60)
    n_a1 = _count(a1)
    n_a4 = _count(a4)
    in_range = 1_800_000 <= n_a4 <= 2_500_000
    all_ok &= in_range
    print(f'  {"INFO"}  A1 (FD2RT_V1) : {n_a1:>9,}')
    print(f'  {PASS if in_range else FAIL}  '
          f'A4 (FD2RT_A4) : {n_a4:>9,}  (Δ = {n_a4 - n_a1:+,})')
    print(f'  {PASS if in_range else FAIL}  '
          f'In expected range [1.8M, 2.5M]: {in_range}')

    # Per-component breakdown
    stage = a4.body[0]
    print('\n  New-component breakdown:')
    for name, sub in stage.denoiser.named_modules():
        if isinstance(sub, (Freq_MSA,)):
            print(f'    {name:50s}  {_count(sub):>8,}')

    # ── SUMMARY ─────────────────────────────────────────────────────────── #
    print('\n' + '=' * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print('=' * 60 + '\n')
    return all_ok


if __name__ == '__main__':
    ok = run_verification()
    import sys; sys.exit(0 if ok else 1)
