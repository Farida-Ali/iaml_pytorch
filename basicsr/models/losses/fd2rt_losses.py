"""
basicsr/models/losses/fd2rt_losses.py
─────────────────────────────────────
Two new loss terms for the FD²RT A7 (full) model. The A4 architecture is
NOT modified — these losses are applied on top of it by the A7 training
model wrapper (basicsr/models/fd2rt_a7_model.py).

NOTE ON FILE LOCATION
─────────────────────
The Phase-3 brief asked for `basicsr/losses/fd2rt_losses.py`, but in this
repo the loss registry lives at `basicsr/models/losses/` — losses are looked
up via `getattr(importlib.import_module('basicsr.models.losses'), name)`
(see image_restoration_model.py:13,125). The file is therefore placed here
and exported from basicsr/models/losses/__init__.py so the classes can be
referenced from a config by name (e.g. `type: FrequencyAwareLoss`).

LOSS 1 — FrequencyAwareLoss
  Penalises low- and high-frequency magnitude-spectrum errors separately,
  with a heavier weight on high frequencies (texture / edges / noise).

LOSS 2 — IlluminationTVLoss
  Anisotropic total-variation prior encouraging a smooth illumination
  (LL-subband) estimate, consistent with the Retinex smoothness assumption.

Run the self-tests:
    python basicsr/models/losses/fd2rt_losses.py
"""

import torch
from torch import nn as nn
from torch.nn import functional as F


# ───────────────────────────────────────────────────────────────────────── #
# LOSS 1 — Frequency-Aware Restoration Loss
# ───────────────────────────────────────────────────────────────────────── #

class FrequencyAwareLoss(nn.Module):
    """Frequency-domain magnitude L1 loss split into low / high bands.

    For each of `pred` and `target`:
        F = rfft2(x)                         (complex, shape [B,C,H,W//2+1])
        |F| = magnitude spectrum
    A circular low-pass mask M_low (radius = H / cutoff_div, default H/8) is
    built on the rfft2 frequency grid; M_high = 1 - M_low. The loss is:

        L_freq = w_low  * L1(M_low  * |F_pred|, M_low  * |F_target|)
               + w_high * L1(M_high * |F_pred|, M_high * |F_target|)

    The whole thing is scaled by `loss_weight`.

    Args:
        loss_weight (float): overall weight (config applies 0.1). Default 1.0.
        w_low  (float): weight on the low-frequency band.  Default 1.0.
        w_high (float): weight on the high-frequency band. Default 2.0.
        cutoff_div (float): cutoff radius = H / cutoff_div. Default 8.0.
        reduction (str): 'mean' | 'sum'. Default 'mean'.
    """

    def __init__(self, loss_weight=1.0, w_low=1.0, w_high=2.0,
                 cutoff_div=8.0, reduction='mean'):
        super().__init__()
        if reduction not in ('mean', 'sum'):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
        self.loss_weight = loss_weight
        self.w_low = w_low
        self.w_high = w_high
        self.cutoff_div = cutoff_div
        self.reduction = reduction
        # Cache masks per (H, W, device, dtype) so they are not rebuilt every call.
        self._mask_cache = {}

    def _get_masks(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        # rfft2 output has shape [..., H, W//2 + 1].
        # Row frequency index (signed): 0,1,...,H/2,...,-1  → distance = min(i, H-i)
        row_idx = torch.arange(H, device=device)
        row_dist = torch.minimum(row_idx, H - row_idx).to(dtype)      # [H]
        # Column (rfft) index: 0,1,...,W//2  → distance = j
        col_dist = torch.arange(W // 2 + 1, device=device).to(dtype)  # [W//2+1]

        rr = row_dist.view(H, 1)
        cc = col_dist.view(1, W // 2 + 1)
        radius = torch.sqrt(rr * rr + cc * cc)                        # [H, W//2+1]

        cutoff = H / self.cutoff_div
        m_low = (radius <= cutoff).to(dtype)                          # [H, W//2+1]
        m_high = 1.0 - m_low
        m_low = m_low.view(1, 1, H, W // 2 + 1)
        m_high = m_high.view(1, 1, H, W // 2 + 1)
        self._mask_cache[key] = (m_low, m_high)
        return m_low, m_high

    def forward(self, pred, target, **kwargs):
        """pred, target: [B, C, H, W] real tensors."""
        assert pred.shape == target.shape, \
            f'shape mismatch: {pred.shape} vs {target.shape}'
        B, C, H, W = pred.shape

        # FFT magnitude spectra (use float32 for numerical stability under AMP).
        fp = torch.fft.rfft2(pred.float(), norm='ortho')
        ft = torch.fft.rfft2(target.float(), norm='ortho')
        mag_p = torch.abs(fp)            # [B, C, H, W//2+1]
        mag_t = torch.abs(ft)

        m_low, m_high = self._get_masks(H, W, pred.device, mag_p.dtype)

        diff = torch.abs(mag_p - mag_t)  # |·| spectrum L1, elementwise
        low_err = diff * m_low
        high_err = diff * m_high

        if self.reduction == 'mean':
            # Reduce EACH BAND BY ITS OWN BIN COUNT, not by the full grid.
            #
            # Previously both bands used .mean(), which divides by the full
            # rfft numel. The low-pass disc is only ~5% of the grid, so the low
            # band was diluted ~20x while the high band (95% of the grid) was
            # essentially undiluted. A configured w_low:w_high of 1:2 therefore
            # acted as roughly 1:38, making the loss an almost pure
            # high-frequency penalty -- the likely cause of the 0.49 dB PSNR
            # regression in A7. Dividing by each band's own bin count makes both
            # terms a genuine per-coefficient mean, so the configured weights
            # mean what they say.
            n_low = m_low.sum() * B * C
            n_high = m_high.sum() * B * C
            l_low = low_err.sum() / n_low.clamp_min(1.0)
            l_high = high_err.sum() / n_high.clamp_min(1.0)
        else:  # sum
            l_low = low_err.sum()
            l_high = high_err.sum()

        loss = self.w_low * l_low + self.w_high * l_high
        return self.loss_weight * loss


# ───────────────────────────────────────────────────────────────────────── #
# LOSS 2 — Illumination Smoothness (anisotropic TV) Prior
# ───────────────────────────────────────────────────────────────────────── #

class IlluminationTVLoss(nn.Module):
    """Anisotropic total-variation prior on the illumination / LL map.

        L_tv = mean(|I[:,:,1:,:] - I[:,:,:-1,:]|)        (vertical diffs)
             + mean(|I[:,:,:,1:] - I[:,:,:,:-1]|)        (horizontal diffs)

    Scaled by `loss_weight`.

    Args:
        loss_weight (float): overall weight (config applies 0.01). Default 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, illum_map, **kwargs):
        """illum_map: [B, C, h, w]."""
        assert illum_map.dim() == 4, \
            f'expected 4-D [B,C,h,w], got {tuple(illum_map.shape)}'
        dh = torch.abs(illum_map[:, :, 1:, :] - illum_map[:, :, :-1, :]).mean()
        dw = torch.abs(illum_map[:, :, :, 1:] - illum_map[:, :, :, :-1]).mean()
        return self.loss_weight * (dh + dw)


# ───────────────────────────────────────────────────────────────────────── #
# Self-tests:  python basicsr/models/losses/fd2rt_losses.py
# ───────────────────────────────────────────────────────────────────────── #

def _run_tests():
    torch.manual_seed(0)
    PASS, FAIL = 'PASS', 'FAIL'
    all_ok = True

    print('=' * 64)
    print('FrequencyAwareLoss')
    print('=' * 64)
    freq = FrequencyAwareLoss(loss_weight=1.0, w_low=1.0, w_high=2.0)

    # T1: identical inputs → 0
    x = torch.rand(2, 3, 64, 64)
    l_same = freq(x, x.clone()).item()
    ok = abs(l_same) < 1e-6
    all_ok &= ok
    print(f'  {PASS if ok else FAIL}  identical inputs → loss == 0   (got {l_same:.2e})')

    # T2: gaussian noise on target → loss > 0 and high-band > low-band
    target = torch.rand(2, 3, 64, 64)
    pred = target + 0.1 * torch.randn_like(target)
    # Inspect the two bands separately using two single-band instances.
    low_only = FrequencyAwareLoss(loss_weight=1.0, w_low=1.0, w_high=0.0)
    high_only = FrequencyAwareLoss(loss_weight=1.0, w_low=0.0, w_high=1.0)
    l_total = freq(pred, target).item()
    l_low = low_only(pred, target).item()
    l_high = high_only(pred, target).item()
    ok_pos = l_total > 0
    ok_hf = l_high > l_low
    all_ok &= ok_pos and ok_hf
    print(f'  {PASS if ok_pos else FAIL}  noisy target → loss > 0        (got {l_total:.4f})')
    print(f'  {PASS if ok_hf else FAIL}  high-band > low-band           '
          f'(low={l_low:.4f}, high={l_high:.4f})')

    # T3: gradient flows
    leaf = torch.rand(2, 3, 64, 64, requires_grad=True)
    tgt = torch.rand(2, 3, 64, 64)
    freq(leaf, tgt).backward()
    ok_grad = leaf.grad is not None and torch.isfinite(leaf.grad).all() \
        and leaf.grad.abs().sum() > 0
    all_ok &= ok_grad
    print(f'  {PASS if ok_grad else FAIL}  gradient flows                 '
          f'(grad sum {leaf.grad.abs().sum().item():.4f})')

    # T4: mask cache reused
    _ = freq(x, x.clone())
    cache_ok = len(freq._mask_cache) >= 1
    all_ok &= cache_ok
    print(f'  {PASS if cache_ok else FAIL}  mask cached per resolution     '
          f'({len(freq._mask_cache)} entry)')

    # T5: odd H/W handled by rfft2 (no crash, finite)
    xo = torch.rand(1, 3, 63, 65)
    yo = torch.rand(1, 3, 63, 65)
    lo = freq(xo, yo).item()
    ok_odd = torch.isfinite(torch.tensor(lo))
    all_ok &= bool(ok_odd)
    print(f'  {PASS if ok_odd else FAIL}  odd H×W (63×65) handled        (got {lo:.4f})')

    print()
    print('=' * 64)
    print('IlluminationTVLoss')
    print('=' * 64)
    tv = IlluminationTVLoss(loss_weight=1.0)

    # T1: constant map → 0
    const = torch.full((2, 3, 32, 32), 0.5)
    l_const = tv(const).item()
    ok = abs(l_const) < 1e-8
    all_ok &= ok
    print(f'  {PASS if ok else FAIL}  constant map → loss == 0       (got {l_const:.2e})')

    # T2: checkerboard → large
    cb = torch.zeros(1, 1, 32, 32)
    cb[:, :, ::2, ::2] = 1.0
    cb[:, :, 1::2, 1::2] = 1.0
    l_cb = tv(cb).item()
    # T3: smooth linear gradient → small but nonzero
    ramp = torch.linspace(0, 1, 32).view(1, 1, 1, 32).expand(1, 1, 32, 32).contiguous()
    l_ramp = tv(ramp).item()
    ok_cb = l_cb > l_ramp and l_cb > 0.1
    ok_ramp = 0 < l_ramp < l_cb
    all_ok &= ok_cb and ok_ramp
    print(f'  {PASS if ok_cb else FAIL}  checkerboard large             (got {l_cb:.4f})')
    print(f'  {PASS if ok_ramp else FAIL}  linear ramp small but nonzero  '
          f'(got {l_ramp:.4f}, < checkerboard)')

    # T4: gradient flows
    leaf2 = torch.rand(2, 3, 16, 16, requires_grad=True)
    tv(leaf2).backward()
    ok_grad2 = leaf2.grad is not None and torch.isfinite(leaf2.grad).all() \
        and leaf2.grad.abs().sum() > 0
    all_ok &= ok_grad2
    print(f'  {PASS if ok_grad2 else FAIL}  gradient flows                 '
          f'(grad sum {leaf2.grad.abs().sum().item():.4f})')

    print()
    print('=' * 64)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print('=' * 64)
    return all_ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if _run_tests() else 1)
