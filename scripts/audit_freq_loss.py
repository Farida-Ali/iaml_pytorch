"""Quantify the low/high band imbalance in FrequencyAwareLoss + param counts."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basicsr.models.losses import FrequencyAwareLoss

print('='*70)
print('TEST 3 — FrequencyAwareLoss band normalisation')
print('='*70)
for H in (128, 256):
    f = FrequencyAwareLoss(cutoff_div=8.0)
    m_low, m_high = f._get_masks(H, H, torch.device('cpu'), torch.float32)
    n_tot = m_low.numel()
    n_low, n_high = int(m_low.sum()), int(m_high.sum())
    print(f'\n  {H}x{H}: rfft grid = {m_low.shape[-2]}x{m_low.shape[-1]} = {n_tot} bins')
    print(f'    low-band bins : {n_low:6d}  ({100*n_low/n_tot:5.2f}% of grid)')
    print(f'    high-band bins: {n_high:6d}  ({100*n_high/n_tot:5.2f}% of grid)')
    print(f'    .mean() divides BOTH by {n_tot} (full numel), not by band size')
    dil_low, dil_high = n_tot/n_low, n_tot/n_high
    print(f'    -> low  diluted {dil_low:5.1f}x : effective w_low  = 1.0/{dil_low:.1f} = {1.0/dil_low:.4f}')
    print(f'    -> high diluted {dil_high:5.2f}x : effective w_high = 2.0/{dil_high:.2f} = {2.0/dil_high:.4f}')
    print(f'    INTENDED low:high = 1:2      ACTUAL = 1:{(2.0/dil_high)/(1.0/dil_low):.1f}')

# Empirical: per-band mean error contribution
torch.manual_seed(0)
t = torch.rand(4, 3, 128, 128)
p = t + 0.05*torch.randn_like(t)
lo = FrequencyAwareLoss(loss_weight=1., w_low=1., w_high=0.)(p, t).item()
hi = FrequencyAwareLoss(loss_weight=1., w_low=0., w_high=1.)(p, t).item()
print(f'\n  empirical (128x128, w=1 each): L_low={lo:.6f}  L_high={hi:.6f}  ratio 1:{hi/lo:.1f}')

print('\n' + '='*70)
print('TEST 4 — phase blindness')
print('='*70)
x = torch.rand(1, 1, 64, 64)
F1 = torch.fft.rfft2(x, norm='ortho')
scrambled = torch.fft.irfft2(torch.abs(F1)*torch.exp(1j*torch.rand_like(torch.abs(F1))*6.283),
                             s=(64,64), norm='ortho')
f = FrequencyAwareLoss(loss_weight=1.)
print(f'  L_freq(x, phase-scrambled x) = {f(scrambled, x).item():.6f}')
print(f'  L1(x, phase-scrambled x)     = {(scrambled-x).abs().mean().item():.6f}')
print('  -> loss uses |FFT| only; identical magnitude + random phase is ~free.')

print('\n' + '='*70)
print('TEST 5 — parameter counts')
print('='*70)
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
from basicsr.models.archs.fd2rt_v1_arch import FD2RT_V1
from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4
kw = dict(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2])
for name, cls in [('A0 RetinexFormer', RetinexFormer), ('A1 FD2RT_V1', FD2RT_V1),
                  ('A4 FD2RT_A4', FD2RT_A4)]:
    n = sum(p.numel() for p in cls(**kw).parameters())
    print(f'  {name:<20}: {n:>10,} params  ({n/1e6:.3f} M)')
