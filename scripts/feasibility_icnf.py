"""
Feasibility test for the Illumination-Conditioned Noise Floor (ICNF) idea.

CLAIM UNDER TEST
  In a low-light image, high-frequency wavelet energy is a mixture of
  reflectance texture and sensor noise. Because photon noise is
  signal-dependent (Poisson-Gaussian: var = a*I + b), the *expected* noise
  energy at each pixel is predictable from the illumination estimate the
  Retinex model already produces. Subtracting that predicted noise floor from
  the observed high-frequency energy should isolate genuine texture far better
  than raw high-frequency energy or a heuristic SNR ratio.

WHY WAVELETS MATTER HERE
  An orthonormal transform preserves variance (Parseval), so white noise of
  variance s^2 lands with variance s^2 in EVERY subband. That makes the noise
  floor analytically known per-subband, and Haar is simultaneously localised,
  so the estimate is spatially varying. Fourier is orthonormal but not
  localised; a plain conv is localised but not orthonormal.

Tests:
  1. Is their HaarDWT2D actually orthonormal? (Parseval)
  2. Does white noise of variance s^2 appear with variance s^2 in each subband?
  3. Does the classical MAD estimator recover s from HH?
  4. THE DECISIVE ONE: does the ICNF statistic separate textured from flat
     regions better than (a) raw HF energy and (b) an SNR-Net-style ratio,
     under realistic signal-dependent noise and a strong illumination gradient?
"""
import sys, os, math, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavelet_utils import HaarDWT2D

torch.manual_seed(0)
dwt = HaarDWT2D()
N = 256


def one_over_f(H, W, beta=1.3):
    fy = torch.fft.fftfreq(H).view(-1, 1); fx = torch.fft.fftfreq(W).view(1, -1)
    f = torch.sqrt(fy**2 + fx**2); f[0, 0] = 1e-6
    amp = 1.0 / (f ** beta); amp[0, 0] = 0
    img = torch.fft.ifft2(amp * torch.exp(1j * torch.rand(H, W) * 2 * math.pi)).real
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


print('=' * 76)
print('TEST 1 — is HaarDWT2D orthonormal? (Parseval identity)')
print('=' * 76)
x = torch.randn(2, 3, 64, 64)
LL, LH, HL, HH = dwt(x)
lhs = (x ** 2).sum().item()
rhs = sum((s ** 2).sum().item() for s in (LL, LH, HL, HH))
print(f'  ||x||^2                       = {lhs:.4f}')
print(f'  ||LL||^2+||LH||^2+||HL||^2+||HH||^2 = {rhs:.4f}')
print(f'  relative error = {abs(lhs-rhs)/lhs:.3e}  -> '
      f'{"ORTHONORMAL" if abs(lhs-rhs)/lhs < 1e-5 else "NOT orthonormal"}')

print()
print('=' * 76)
print('TEST 2 — does white noise keep its variance in every subband?')
print('=' * 76)
print(f'{"true sigma":>12} {"var(LL)":>10} {"var(LH)":>10} {"var(HL)":>10} {"var(HH)":>10}')
print('-' * 76)
for s in (0.01, 0.03, 0.05, 0.10):
    n = s * torch.randn(8, 1, 256, 256)
    a, b, c, d = dwt(n)
    print(f'{s**2:>12.6f} {a.var().item():>10.6f} {b.var().item():>10.6f} '
          f'{c.var().item():>10.6f} {d.var().item():>10.6f}')
print('  (first column is the TRUE variance sigma^2 — subbands should match it)')

print()
print('=' * 76)
print('TEST 3 — MAD estimator on HH:  sigma_hat = median(|HH|) / 0.6745')
print('=' * 76)
print(f'{"true sigma":>12} {"sigma_hat":>12} {"rel err":>10}')
print('-' * 76)
for s in (0.005, 0.01, 0.02, 0.05, 0.10):
    n = s * torch.randn(4, 1, 256, 256)
    _, _, _, hh = dwt(n)
    s_hat = hh.abs().median().item() / 0.6745
    print(f'{s:>12.5f} {s_hat:>12.5f} {abs(s_hat-s)/s*100:>9.2f}%')

print()
print('=' * 76)
print('TEST 4 — DECISIVE: texture-vs-noise separability under low light')
print('=' * 76)

BOX = 9
box_k = torch.ones(1, 1, BOX, BOX) / (BOX * BOX)


def local_mean(v):
    return F.conv2d(v, box_k, padding=BOX // 2)


def auc(scores, labels):
    """P(score[pos] > score[neg]) via Mann-Whitney U. Threshold-free."""
    s = scores.flatten(); y = labels.flatten().bool()
    order = torch.argsort(s)
    ranks = torch.empty_like(s); ranks[order] = torch.arange(len(s), dtype=s.dtype)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return ((ranks[y].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)).item()


def make_scene(mean_level, a=0.02, b=1e-5):
    """Half flat / half textured reflectance under a smooth illumination ramp."""
    R = torch.full((N, N), 0.5)
    R[:, N // 2:] = 0.25 + 0.5 * one_over_f(N, N // 2)     # textured half
    label = torch.zeros(N, N); label[:, N // 2:] = 1.0      # 1 = has texture

    yy = torch.linspace(0, 1, N).view(-1, 1).expand(N, N)   # vertical ramp
    L = 0.15 + 1.7 * yy                                      # strong illum range
    L = L * (mean_level / L.mean())

    clean = (L * R).clamp(1e-4, 1)
    var = a * clean + b                                      # Poisson-Gaussian
    obs = (clean + var.sqrt() * torch.randn(N, N)).clamp(1e-4, 1)
    return obs.view(1, 1, N, N), clean.view(1, 1, N, N), L.view(1, 1, N, N), label, a, b


def statistics(obs, L_est, a, b):
    """Three competing texture-evidence maps."""
    _, lh, hl, hh = dwt(obs)
    E = local_mean(lh**2 + hl**2 + hh**2) / 3.0              # observed HF energy
    E_up = F.interpolate(E, size=(N, N), mode='bilinear', align_corners=False)

    # (a) naive: raw high-frequency energy
    naive = E_up

    # (b) SNR-Net style heuristic: |I| / |I - blur(I)|
    blur = local_mean(obs)
    snr = obs.abs() / (obs - blur).abs().clamp_min(1e-4)

    # (c) PROPOSED: subtract the illumination-predicted noise floor.
    #     Under Parseval the per-coefficient noise variance equals the pixel
    #     noise variance, so predicted subband energy = a*I + b.
    noise_var = a * L_est + b
    icnf = (E_up - noise_var).clamp_min(0) / (noise_var + 1e-8)
    return naive, snr, icnf


print(f'{"scene mean":>11} {"raw HF energy":>15} {"SNR-Net style":>15} {"ICNF (ours)":>14}')
print('-' * 76)
rows = []
for mean_level in (0.05, 0.08, 0.15, 0.30, 0.50):
    aucs = []
    for _ in range(6):
        obs, clean, L, label, a, b = make_scene(mean_level)
        # Use the OBSERVED image as the illumination proxy (what the net has
        # access to at inference) — not the oracle L. Keeps the test honest.
        L_proxy = local_mean(obs)
        naive, snr, icnf = statistics(obs, L_proxy, a, b)
        aucs.append([auc(naive, label), auc(snr, label), auc(icnf, label)])
    m = [sum(c) / len(c) for c in zip(*aucs)]
    rows.append((mean_level, m))
    print(f'{mean_level:>11.2f} {m[0]:>15.4f} {m[1]:>15.4f} {m[2]:>14.4f}')

print('\n  AUC = P(statistic ranks a textured pixel above a flat one).')
print('  0.5 = useless, 1.0 = perfect separation.')

dark = [r for r in rows if r[0] <= 0.08]
if dark:
    avg_naive = sum(r[1][0] for r in dark) / len(dark)
    avg_snr = sum(r[1][1] for r in dark) / len(dark)
    avg_icnf = sum(r[1][2] for r in dark) / len(dark)
    print(f'\n  In the dark regime (mean <= 0.08), averaged:')
    print(f'    raw HF energy : {avg_naive:.4f}')
    print(f'    SNR-Net style : {avg_snr:.4f}')
    print(f'    ICNF (ours)   : {avg_icnf:.4f}')
    best = max(avg_naive, avg_snr, avg_icnf)
    name = ('ICNF' if best == avg_icnf else
            'SNR-Net style' if best == avg_snr else 'raw HF energy')
    print(f'    -> best: {name}')

print()
print('=' * 76)
print('TEST 5 — what fraction of tokens could skip attention?')
print('=' * 76)
obs, clean, L, label, a, b = make_scene(0.08)
_, _, icnf = statistics(obs, local_mean(obs), a, b)
for thr in (0.25, 0.5, 1.0, 2.0):
    frac = (icnf < thr).float().mean().item()
    kept_tex = (icnf[label.view(1,1,N,N) > 0] >= thr).float().mean().item()
    print(f'  threshold {thr:>4.2f}: {100*frac:5.1f}% of pixels below floor '
          f'(skippable) | {100*kept_tex:5.1f}% of true texture retained')
