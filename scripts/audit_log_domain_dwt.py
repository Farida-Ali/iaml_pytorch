"""
Does a log transform before the Haar DWT actually separate illumination from
reflectance better than DWT on linear intensity?

Retinex:  I = L * R      (L smooth illumination, R textured reflectance)
Log:      log I = log L + log R
Since DWT is linear, in log space it distributes over the sum; in linear space
DWT(L*R) does not factor. That is the theoretical claim. This measures whether
it survives contact with realistic signals.

Protocol: synthesise (L, R) with known ground truth, form I = L*R, then recover
an illumination estimate from the LL subband via each path, fit the best affine
map (so neither path is penalised for scale/offset), and compare the recovered
L to the true L in the LINEAR domain — a common yardstick for both.
"""
import sys, os, math, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavelet_utils import HaarDWT2D

torch.manual_seed(0)
dwt = HaarDWT2D()
N = 256


def one_over_f(shape, beta=1.4):
    """1/f^beta noise — a reasonable stand-in for natural-image texture."""
    H, W = shape
    fy = torch.fft.fftfreq(H).view(-1, 1)
    fx = torch.fft.fftfreq(W).view(1, -1)
    f = torch.sqrt(fy**2 + fx**2)
    f[0, 0] = 1e-6
    amp = 1.0 / (f ** beta)
    amp[0, 0] = 0
    ph = torch.rand(H, W) * 2 * math.pi
    img = torch.fft.ifft2(amp * torch.exp(1j * ph)).real
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def smooth_illum(shape, mean_level):
    """Smooth low-frequency illumination field with the requested mean."""
    H, W = shape
    yy = torch.linspace(-1, 1, H).view(-1, 1)
    xx = torch.linspace(-1, 1, W).view(1, -1)
    L = torch.zeros(H, W)
    for _ in range(3):
        cy, cx = torch.rand(2) * 1.6 - 0.8
        s = 0.3 + torch.rand(1).item() * 0.5
        L = L + torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    L = (L - L.min()) / (L.max() - L.min() + 1e-8)
    L = 0.35 + 0.65 * L                      # contrast range, then rescale mean
    return L * (mean_level / L.mean())


def ll_of(x):
    """LL subband, upsampled back to full resolution."""
    LL, _, _, _ = dwt(x.view(1, 1, N, N))
    return F.interpolate(LL, size=(N, N), mode='bilinear', align_corners=False).view(N, N)


def affine_fit(pred, target):
    """Least-squares a*pred+b -> target. Returns fitted prediction."""
    p = pred.flatten(); t = target.flatten()
    A = torch.stack([p, torch.ones_like(p)], 1)
    sol = torch.linalg.lstsq(A, t.unsqueeze(1)).solution
    return (A @ sol).view_as(target)


def nrmse(est, true):
    return ((est - true) ** 2).mean().sqrt().item() / true.std().item()


def trial(mean_level, c, noise_std=0.0):
    L = smooth_illum((N, N), mean_level)
    R = 0.15 + 0.85 * one_over_f((N, N))          # reflectance in (0,1]
    I = (L * R).clamp(1e-4, 1.0)
    if noise_std > 0:
        I = (I + noise_std * torch.randn_like(I)).clamp(1e-4, 1.0)

    # ---- Path A: DWT on linear intensity (what FD2RT does today) ----
    ll_lin = ll_of(I)
    L_hat_lin = affine_fit(ll_lin, L)

    # ---- Path B: DWT on log-transformed intensity ----
    ll_log = ll_of(torch.log1p(c * I))
    tgt_log = torch.log1p(c * L)
    fitted_log = affine_fit(ll_log, tgt_log)
    L_hat_log = ((fitted_log.exp() - 1) / c).clamp(0, 5)

    return nrmse(L_hat_lin, L), nrmse(L_hat_log, L)


print('=' * 74)
print('TEST A — illumination-recovery error (lower = cleaner separation)')
print('        NRMSE of recovered L vs true L, after best affine fit')
print('=' * 74)
print(f'{"scene mean":>11} {"noise":>7} {"linear DWT":>12} {"log DWT c=5":>13} {"change":>10}')
print('-' * 74)
for mean_level, noise in [(0.08, 0.0), (0.08, 0.01), (0.15, 0.0),
                          (0.30, 0.0), (0.50, 0.0), (0.50, 0.01)]:
    lin, lg = zip(*[trial(mean_level, 5.0, noise) for _ in range(12)])
    lin, lg = sum(lin) / len(lin), sum(lg) / len(lg)
    delta = 100 * (lg - lin) / lin
    tag = 'log BETTER' if delta < -2 else ('log worse' if delta > 2 else 'no real diff')
    print(f'{mean_level:>11.2f} {noise:>7.3f} {lin:>12.4f} {lg:>13.4f} '
          f'{delta:>+9.1f}%  {tag}')

print()
print('=' * 74)
print('TEST B — sensitivity to c (scene mean 0.08, the low-light regime)')
print('=' * 74)
base = sum(trial(0.08, 5.0)[0] for _ in range(12)) / 12
print(f'{"c":>6} {"log-DWT NRMSE":>15} {"vs linear":>12}')
print('-' * 74)
for c in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
    lg = sum(trial(0.08, c)[1] for _ in range(12)) / 12
    print(f'{c:>6.1f} {lg:>15.4f} {100*(lg-base)/base:>+11.1f}%')
print(f'\n  (linear-DWT reference NRMSE = {base:.4f})')

print()
print('=' * 74)
print('TEST C — does log break the "W-IE == box blur of mean_c" identity?')
print('=' * 74)
img = torch.rand(4, 3, 128, 128) * 0.25          # low-light-ish RGB
mean_c = img.mean(1, keepdim=True)

LL, _, _, _ = dwt(img)
ll_mean_lin = F.interpolate(LL, size=(128, 128), mode='bilinear',
                            align_corners=False).mean(1, keepdim=True)
LLg, _, _, _ = dwt(torch.log1p(5.0 * img))
ll_mean_log = F.interpolate(LLg, size=(128, 128), mode='bilinear',
                            align_corners=False).mean(1, keepdim=True)

box = F.interpolate(F.avg_pool2d(mean_c, 2), size=(128, 128),
                    mode='bilinear', align_corners=False) * 2.0
print(f'  |LL_mean_linear - 2*boxblur(mean_c)| max = '
      f'{(ll_mean_lin - box).abs().max().item():.3e}   <- current code')
print(f'  |LL_mean_log    - 2*boxblur(mean_c)| max = '
      f'{(ll_mean_log - box).abs().max().item():.3e}   <- with log')


def corr(a, b):
    a = (a - a.mean()).flatten(); b = (b - b.mean()).flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


print(f'\n  corr(LL_mean_linear, mean_c) = {corr(ll_mean_lin, mean_c):.6f}')
print(f'  corr(LL_mean_log,    mean_c) = {corr(ll_mean_log, mean_c):.6f}')
print('\n  A monotone pointwise map cannot decorrelate much on its own; what it')
print('  changes is the *spacing* of values, i.e. where precision is spent.')

print()
print('=' * 74)
print('TEST D — where does log spend precision? (gradient of the transform)')
print('=' * 74)
print(f'{"intensity":>10} {"d/dx ln(1+5x)":>16} {"relative to x=1":>18}')
print('-' * 74)
g1 = 5.0 / (1 + 5.0 * 1.0)
for x in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00):
    g = 5.0 / (1 + 5.0 * x)
    print(f'{x:>10.2f} {g:>16.4f} {g/g1:>17.2f}x')
