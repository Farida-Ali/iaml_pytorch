"""
Phase-1 tests for the ICNF module.

The bar is NOT "it runs". The bar is that the real module reproduces the
separability measured in the feasibility study, because that is the empirical
claim the paper will make.

  python scripts/test_p1_icnf.py
"""
import sys, os, math, traceback
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

from basicsr.models.archs.icnf import ICNF, ICNFGate

RESULTS = []
N = 256


def check(label, cond, detail=''):
    RESULTS.append((label, bool(cond), detail))
    print(f'  {"PASS" if cond else "FAIL"}  {label}' + (f'  {detail}' if detail else ''))
    return bool(cond)


def texture(H, W, beta=0.4):
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


def scene(mean_level, tex_amp, ramp=1.0, a=0.02, b=1e-5):
    R = torch.full((N, N), 0.5)
    R[:, N//2:] = (0.5 + tex_amp * texture(N, N//2)).clamp(0.05, 0.95)
    label = torch.zeros(N, N); label[:, N//2:] = 1.0
    yy = torch.linspace(0, 1, N).view(-1, 1).expand(N, N)
    L = (1.0 + ramp * (yy - 0.5))
    L = (L * (mean_level / L.mean())).clamp_min(1e-3)
    clean = (L * R).clamp(1e-4, 1)
    var = a * clean + b
    obs = (clean + var.sqrt() * torch.randn(N, N)).clamp(1e-4, 1)
    return obs.view(1, 1, N, N), label


# ───────────────────────────────────────────────────────────────────────── #
def test_shapes_and_params():
    print('\n' + '=' * 74)
    print('P1.1 — shapes, parameter count, device safety')
    print('=' * 74)
    m = ICNF()
    x = torch.rand(2, 3, 128, 128)
    ev = m(x)
    check('output shape [B,1,H,W]', tuple(ev.shape) == (2, 1, 128, 128),
          f'{tuple(ev.shape)}')
    check('evidence is non-negative', ev.min().item() >= 0,
          f'min={ev.min().item():.4f}')
    check('custom out_size honoured',
          tuple(m(x, out_size=(64, 64)).shape) == (2, 1, 64, 64))

    n_par = sum(p.numel() for p in m.parameters())
    check('ICNF adds only the 2 sensor params', n_par == 2, f'({n_par} params)')

    g = ICNFGate()
    gate = g(ev)
    check('gate in (0,1)', gate.min() > 0 and gate.max() < 1,
          f'[{gate.min().item():.4f}, {gate.max().item():.4f}]')
    check('gate starts near-closed (reduces to spatial-only at init)',
          gate.mean().item() < 0.25, f'(mean {gate.mean().item():.4f})')

    check('odd-window enforced', _raises(lambda: ICNF(window=8), ValueError))
    check('bad mode rejected', _raises(lambda: ICNF(mode='nope'), ValueError))


def _raises(fn, exc):
    try:
        fn(); return False
    except exc:
        return True


def test_gradients():
    print('\n' + '=' * 74)
    print('P1.2 — gradient flow (the check that would have caught dead L_tv)')
    print('=' * 74)
    from scripts.gradient_flow import assert_loss_is_live

    m = ICNF(learn_params=True)
    x = torch.rand(2, 3, 64, 64)
    live, g = assert_loss_is_live(m, lambda: m(x).mean())
    check('ICNF sensor params (a,b) receive gradient', live, f'|grad|={g:.6e}')

    gate = ICNFGate()
    ev = torch.rand(2, 1, 32, 32)
    live2, g2 = assert_loss_is_live(gate, lambda: gate(ev).mean())
    check('ICNFGate slope/bias receive gradient', live2, f'|grad|={g2:.6f}')

    # Evidence must carry gradient back to the image path too, so a downstream
    # block gated by it is trainable.
    leaf = torch.rand(1, 3, 64, 64, requires_grad=True)
    ICNF()(leaf).mean().backward()
    check('evidence is differentiable w.r.t. the input image',
          leaf.grad is not None and leaf.grad.abs().sum() > 0)

    # a,b stay positive under the log parameterisation.
    m2 = ICNF(a_init=0.02, b_init=1e-5)
    with torch.no_grad():
        m2.log_a -= 50.0
    check('a stays strictly positive under log-param', m2.a.item() > 0,
          f'(a={m2.a.item():.3e})')


def test_mad_estimator():
    print('\n' + '=' * 74)
    print('P1.3 — MAD sigma estimator (calibration path)')
    print('=' * 74)
    m = ICNF()
    ok = True
    for s in (0.005, 0.01, 0.05, 0.10):
        n = s * torch.randn(4, 1, 256, 256)
        est = m.estimate_sigma_mad(n).mean().item()
        err = abs(est - s) / s
        ok &= err < 0.05
        print(f'      sigma={s:.4f} -> est={est:.5f}  ({100*err:.2f}% err)')
    check('MAD recovers sigma within 5% across the range', ok)


def test_separability():
    print('\n' + '=' * 74)
    print('P1.4 — DECISIVE: reproduces feasibility separability')
    print('=' * 74)
    torch.manual_seed(0)
    m = ICNF(a_init=0.02, b_init=1e-5, learn_params=False)

    def raw_hf(obs):
        E = m.hf_energy(obs).mean(1, keepdim=True)
        return F.interpolate(E, size=(N, N), mode='bilinear', align_corners=False)

    print(f'  {"illum range":>12} {"raw HF":>9} {"ICNF":>9} {"gain":>9}')
    print('  ' + '-' * 44)
    gains = []
    for ramp, desc in [(0.0, '1:1'), (1.0, '3:1'), (1.8, '19:1')]:
        r_, i_ = [], []
        for _ in range(6):
            obs, label = scene(0.15, 0.15, ramp=ramp)
            r_.append(auc(raw_hf(obs), label))
            i_.append(auc(m(obs), label))
        r, i = sum(r_)/len(r_), sum(i_)/len(i_)
        gains.append((desc, r, i, i - r))
        print(f'  {desc:>12} {r:>9.4f} {i:>9.4f} {i-r:>+9.4f}')

    check('ICNF beats raw HF energy at every illumination range',
          all(g[3] > 0 for g in gains))
    flat_gain, steep_gain = gains[0][3], gains[-1][3]
    check('advantage GROWS with illumination range (the mechanism)',
          steep_gain > flat_gain,
          f'(+{flat_gain:.4f} at 1:1 -> +{steep_gain:.4f} at 19:1)')
    check('ICNF stays well above chance under steep illumination',
          gains[-1][2] > 0.70, f'(AUC {gains[-1][2]:.4f})')

    # Honest limitation: below TNR ~0.02 nothing separates.
    obs, label = scene(0.08, 0.01)
    a_lo = auc(m(obs), label)
    check('near-chance when texture is below the noise floor (expected limit)',
          a_lo < 0.60, f'(AUC {a_lo:.4f} — information is genuinely gone)')


def test_modes():
    print('\n' + '=' * 74)
    print('P1.5 — subtract vs ratio mode')
    print('=' * 74)
    torch.manual_seed(0)
    res = {}
    for mode in ('subtract', 'ratio'):
        m = ICNF(learn_params=False, mode=mode)
        vals = []
        for _ in range(6):
            obs, label = scene(0.30, 0.40)
            vals.append(auc(m(obs), label))
        res[mode] = sum(vals) / len(vals)
        print(f'      {mode:<9}: AUC {res[mode]:.4f}')
    check('both modes are strongly discriminative',
          min(res.values()) > 0.85, f'({res})')


if __name__ == '__main__':
    for fn in (test_shapes_and_params, test_gradients, test_mad_estimator,
               test_separability, test_modes):
        try:
            fn()
        except Exception:
            traceback.print_exc()
            RESULTS.append((fn.__name__, False, 'raised'))

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print('\n' + '=' * 74)
    print(f'PHASE 1 TESTS: {n_pass}/{len(RESULTS)} passed')
    print('=' * 74)
    for lbl, ok, det in RESULTS:
        if not ok:
            print(f'  FAILED: {lbl} {det}')
    sys.exit(0 if n_pass == len(RESULTS) else 1)
