"""
ICNF feasibility, round 2 — characterise WHERE the statistic works.

Round 1 put all three statistics at chance. Diagnosis: the synthetic texture
(1/f^1.3) carried almost no energy in the top octave, so texture sat far below
the noise floor and no estimator could recover it. That is an information limit,
not an algorithmic one.

Here we sweep the texture-to-noise ratio (TNR) explicitly and report AUC as a
function of it, so we can state the operating regime honestly.
"""
import sys, os, math, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavelet_utils import HaarDWT2D

torch.manual_seed(0)
dwt = HaarDWT2D()
N = 256
BOX = 9
box_k = torch.ones(1, 1, BOX, BOX) / (BOX * BOX)
local_mean = lambda v: F.conv2d(v, box_k, padding=BOX // 2)


def texture(H, W, beta):
    fy = torch.fft.fftfreq(H).view(-1, 1); fx = torch.fft.fftfreq(W).view(1, -1)
    f = torch.sqrt(fy**2 + fx**2); f[0, 0] = 1e-6
    amp = 1.0 / (f ** beta); amp[0, 0] = 0
    img = torch.fft.ifft2(amp * torch.exp(1j * torch.rand(H, W) * 2 * math.pi)).real
    return (img - img.mean()) / (img.std() + 1e-8)


def auc(scores, labels):
    s = scores.flatten(); y = labels.flatten().bool()
    order = torch.argsort(s)
    ranks = torch.empty_like(s); ranks[order] = torch.arange(len(s), dtype=s.dtype)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return ((ranks[y].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)).item()


def hf_energy(v):
    _, lh, hl, hh = dwt(v)
    E = local_mean((lh**2 + hl**2 + hh**2) / 3.0)
    return F.interpolate(E, size=(N, N), mode='bilinear', align_corners=False)


def run(mean_level, tex_amp, beta, a=0.02, b=1e-5, trials=6):
    out = []
    for _ in range(trials):
        R = torch.full((N, N), 0.5)
        R[:, N//2:] = (0.5 + tex_amp * texture(N, N//2, beta)).clamp(0.05, 0.95)
        label = torch.zeros(N, N); label[:, N//2:] = 1.0

        yy = torch.linspace(0, 1, N).view(-1, 1).expand(N, N)
        L = 0.15 + 1.7 * yy
        L = L * (mean_level / L.mean())

        clean = (L * R).clamp(1e-4, 1)
        var = a * clean + b
        obs = (clean + var.sqrt() * torch.randn(N, N)).clamp(1e-4, 1)
        obs4 = obs.view(1,1,N,N); clean4 = clean.view(1,1,N,N)

        # true texture-to-noise ratio in the textured half
        E_clean = hf_energy(clean4)
        tex_mask = label.view(1,1,N,N) > 0
        tnr = (E_clean[tex_mask].mean() / var.view(1,1,N,N)[tex_mask].mean()).item()

        E_obs = hf_energy(obs4)
        L_proxy = local_mean(obs4)

        naive = E_obs
        blur = local_mean(obs4)
        snr = obs4.abs() / (obs4 - blur).abs().clamp_min(1e-4)
        nv = a * L_proxy + b
        icnf = (E_obs - nv).clamp_min(0) / (nv + 1e-8)
        # variant: no clamp, plain ratio
        icnf_r = E_obs / (nv + 1e-8)

        out.append([tnr, auc(naive, label), auc(snr, label),
                    auc(icnf, label), auc(icnf_r, label)])
    return [sum(c)/len(c) for c in zip(*out)]


print('=' * 84)
print('TEST 6 — AUC vs texture-to-noise ratio  (scene mean 0.08, rough texture)')
print('=' * 84)
print(f'{"tex amp":>8} {"TNR":>8} | {"raw HF":>8} {"SNR-Net":>9} {"ICNF-sub":>9} {"ICNF-ratio":>11}')
print('-' * 84)
for amp in (0.01, 0.02, 0.05, 0.10, 0.20, 0.40):
    tnr, n_, s_, i_, ir_ = run(0.08, amp, beta=0.4)
    print(f'{amp:>8.2f} {tnr:>8.3f} | {n_:>8.4f} {s_:>9.4f} {i_:>9.4f} {ir_:>11.4f}')

print()
print('=' * 84)
print('TEST 7 — same, but scene mean 0.30 (moderately lit)')
print('=' * 84)
print(f'{"tex amp":>8} {"TNR":>8} | {"raw HF":>8} {"SNR-Net":>9} {"ICNF-sub":>9} {"ICNF-ratio":>11}')
print('-' * 84)
for amp in (0.01, 0.02, 0.05, 0.10, 0.20, 0.40):
    tnr, n_, s_, i_, ir_ = run(0.30, amp, beta=0.4)
    print(f'{amp:>8.2f} {tnr:>8.3f} | {n_:>8.4f} {s_:>9.4f} {i_:>9.4f} {ir_:>11.4f}')

print()
print('=' * 84)
print('TEST 8 — effect of the ILLUMINATION GRADIENT (the thing ICNF exploits)')
print('   Fixed texture, but illumination ramp made steeper. Raw HF energy')
print('   should degrade (bright texture looks stronger than dark texture);')
print('   ICNF normalises by the illumination-predicted floor and should not.')
print('=' * 84)


def run_ramp(ramp, mean_level=0.15, tex_amp=0.15, beta=0.4, a=0.02, b=1e-5, trials=6):
    out = []
    for _ in range(trials):
        R = torch.full((N, N), 0.5)
        R[:, N//2:] = (0.5 + tex_amp * texture(N, N//2, beta)).clamp(0.05, 0.95)
        label = torch.zeros(N, N); label[:, N//2:] = 1.0
        yy = torch.linspace(0, 1, N).view(-1, 1).expand(N, N)
        L = 1.0 + ramp * (yy - 0.5)
        L = (L * (mean_level / L.mean())).clamp_min(1e-3)
        clean = (L * R).clamp(1e-4, 1)
        var = a * clean + b
        obs = (clean + var.sqrt() * torch.randn(N, N)).clamp(1e-4, 1).view(1,1,N,N)
        E_obs = hf_energy(obs); L_proxy = local_mean(obs)
        blur = local_mean(obs)
        snr = obs.abs() / (obs - blur).abs().clamp_min(1e-4)
        nv = a * L_proxy + b
        out.append([auc(E_obs, label), auc(snr, label),
                    auc(E_obs/(nv+1e-8), label)])
    return [sum(c)/len(c) for c in zip(*out)]


print(f'{"illum ratio":>12} | {"raw HF":>8} {"SNR-Net":>9} {"ICNF-ratio":>11}')
print('-' * 84)
for ramp, desc in [(0.0, '1:1 (flat)'), (0.5, '~1.7:1'), (1.0, '3:1'),
                   (1.5, '7:1'), (1.8, '19:1')]:
    n_, s_, ir_ = run_ramp(ramp)
    print(f'{desc:>12} | {n_:>8.4f} {s_:>9.4f} {ir_:>11.4f}')
