"""
Empirical verification of two suspected defects in the FD2RT A7 pipeline.

TEST 1 (critical): does IlluminationTVLoss on the hooked LL subband produce
                   ANY gradient w.r.t. network parameters?
TEST 2: how different is the W-IE illumination prior from RetinexFormer's
        pixel-domain channel mean?
"""
import sys, os, torch
import torch.nn.functional as F
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4
from basicsr.models.losses import IlluminationTVLoss, FrequencyAwareLoss

torch.manual_seed(0)
net = FD2RT_A4(in_channels=3, out_channels=3, n_feat=40,
               stage=1, num_blocks=[1, 2, 2])

captured = {}
for name, m in net.named_modules():
    if name.endswith('estimator.dwt'):
        m.register_forward_hook(lambda mod, i, o: captured.__setitem__('LL', o[0]))
        hooked = name
        break
print(f'hook attached to: {hooked}')

lq = torch.rand(2, 3, 64, 64)
gt = torch.rand(2, 3, 64, 64)

print('\n' + '='*66)
print('TEST 1 — does L_tv generate gradient w.r.t. network parameters?')
print('='*66)

net.zero_grad()
captured.clear()
out = net(lq)
LL = captured['LL']
print(f'  captured LL shape      : {tuple(LL.shape)}')
print(f'  LL.requires_grad       : {LL.requires_grad}')
print(f'  LL.grad_fn             : {LL.grad_fn}')

l_tv = IlluminationTVLoss(loss_weight=0.01)(LL)
print(f'  l_tv value             : {l_tv.item():.8f}')
print(f'  l_tv.requires_grad     : {l_tv.requires_grad}')

try:
    l_tv.backward()
    total = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
    n_nonzero = sum(1 for p in net.parameters()
                    if p.grad is not None and p.grad.abs().sum() > 0)
    print(f'  total |grad| from L_tv : {total:.10f}')
    print(f'  params with nonzero grad: {n_nonzero} / {sum(1 for _ in net.parameters())}')
    verdict = 'L_tv IS A CONSTANT — contributes ZERO gradient' if total == 0 \
              else 'L_tv does produce gradient'
except RuntimeError as e:
    print(f'  backward() RAISED      : {e}')
    verdict = 'L_tv IS A CONSTANT — not in the autograd graph at all'
print(f'\n  VERDICT: {verdict}')

# Contrast: does L_freq produce gradient?
net.zero_grad(); captured.clear()
out = net(lq)
l_freq = FrequencyAwareLoss(loss_weight=0.1, w_low=1.0, w_high=2.0)(out, gt)
l_freq.backward()
gf = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
print(f'  (control) total |grad| from L_freq: {gf:.6f}  -> {"OK" if gf>0 else "ALSO DEAD"}')

print('\n' + '='*66)
print('TEST 2 — W-IE prior vs RetinexFormer pixel-domain mean')
print('='*66)

img = torch.rand(4, 3, 128, 128)

# RetinexFormer original prior
mean_c = img.mean(dim=1, keepdim=True)

# W-IE prior, reproducing fd2rt_arch.py lines 129-142 exactly
from wavelet_utils import HaarDWT2D
dwt = HaarDWT2D()
LL, LH, HL, HH = dwt(img)
LL_up = F.interpolate(LL, size=(128, 128), mode='bilinear', align_corners=False)
LL_mean = LL_up.mean(dim=1, keepdim=True)

print(f'  mean_c   range: [{mean_c.min():.4f}, {mean_c.max():.4f}]  mean {mean_c.mean():.4f}')
print(f'  LL_mean  range: [{LL_mean.min():.4f}, {LL_mean.max():.4f}]  mean {LL_mean.mean():.4f}')
print(f'  ratio LL_mean/mean_c (global): {(LL_mean.mean()/mean_c.mean()).item():.4f}')

# Hypothesis: LL_mean == 2 * (avgpool2x2 -> bilinear upsample)(mean_c)
approx = F.interpolate(F.avg_pool2d(mean_c, 2), size=(128,128),
                       mode='bilinear', align_corners=False) * 2.0
err = (LL_mean - approx).abs()
print(f'\n  hypothesis: LL_mean == 2 * bilinear_up(avgpool2x2(mean_c))')
print(f'  max abs err : {err.max().item():.3e}')
print(f'  mean abs err: {err.mean().item():.3e}')
print(f'  -> {"EXACT MATCH" if err.max() < 1e-5 else "differs"}')

# Correlation with the plain mean
a = (LL_mean - LL_mean.mean()).flatten()
b = (mean_c - mean_c.mean()).flatten()
corr = (a @ b / (a.norm() * b.norm())).item()
print(f'\n  Pearson corr(LL_mean, mean_c) on random noise : {corr:.4f}')

# On a real-ish smooth image the correlation will be even higher
xx = torch.linspace(-3, 3, 128)
smooth = torch.exp(-(xx.view(-1,1)**2 + xx.view(1,-1)**2)/4).expand(1,3,128,128).contiguous()
smooth = smooth + 0.05*torch.randn_like(smooth)
m2 = smooth.mean(1, keepdim=True)
L2,_,_,_ = dwt(smooth)
L2 = F.interpolate(L2, size=(128,128), mode='bilinear', align_corners=False).mean(1, keepdim=True)
a2 = (L2 - L2.mean()).flatten(); b2 = (m2 - m2.mean()).flatten()
print(f'  Pearson corr on smooth natural-like image      : {(a2@b2/(a2.norm()*b2.norm())).item():.4f}')
