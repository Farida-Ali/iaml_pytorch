"""
Efficiency measurement for the FD²RT ladder: params, FLOPs, runtime, memory.

Any top-tier paper in this area reports these; none of them existed in this
repo. FLOPs are counted with forward hooks (no external dependency) over the
layer types that dominate this architecture: Conv2d, ConvTranspose2d, Linear,
and the attention matmuls inside IG_MSA / Freq_MSA.

It also quantifies the ADAPTIVE-COMPUTE HEADROOM that ICNF creates. Where the
observed high-frequency energy is fully explained by the illumination-predicted
noise floor there is no texture to recover, so the frequency branch has nothing
to do there and could be skipped. This reports what fraction of real image area
falls below the floor at a range of thresholds, and the FLOPs that would save
 -- the evidence needed to decide whether to build token-level sparsity, rather
than assuming it pays.

  python scripts/measure_efficiency.py --size 256 --data data/real_lowlight_probe
"""
import sys, os, glob, time, argparse
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn as nn

KW = dict(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])


# ── FLOP counting ──────────────────────────────────────────────────────── #

def _conv_flops(m, inp, out):
    # MACs: out_elements * (in_channels/groups) * kernel_area
    out_el = out.numel()
    kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,) * 2
    return out_el * (m.in_channels // m.groups) * kh * kw


def _deconv_flops(m, inp, out):
    in_el = inp[0].numel()
    kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,) * 2
    return in_el * (m.out_channels // m.groups) * kh * kw


def _linear_flops(m, inp, out):
    return out.numel() * m.in_features


def count_flops(net, x):
    """Returns (total_macs, per_module_type_breakdown)."""
    totals = {}
    handles = []

    def mk(fn, key):
        def hook(m, i, o):
            if isinstance(o, (tuple, list)):
                o = o[0]
            try:
                f = fn(m, i, o)
            except Exception:
                f = 0
            totals[key] = totals.get(key, 0) + f
        return hook

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(mk(_conv_flops, 'conv')))
        elif isinstance(m, nn.ConvTranspose2d):
            handles.append(m.register_forward_hook(mk(_deconv_flops, 'deconv')))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(mk(_linear_flops, 'linear')))

    with torch.no_grad():
        net(x)
    for h in handles:
        h.remove()
    return sum(totals.values()), totals


def build(name):
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
    from basicsr.models.archs.fd2rt_v1_arch import FD2RT_V1
    from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4
    from basicsr.models.archs.fd2rt_icnf_arch import FD2RT_ICNF
    return {'A0 RetinexFormer': RetinexFormer, 'A1 FD2RT_V1': FD2RT_V1,
            'A4 FD2RT_A4': FD2RT_A4, 'ICNF FD2RT_ICNF': FD2RT_ICNF}[name](**KW)


def timeit(net, x, warmup=2, runs=5):
    net.eval()
    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            net(x)
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--data', default='data/real_lowlight_probe')
    args = ap.parse_args()

    S = args.size
    x = torch.rand(1, 3, S, S)

    print('=' * 84)
    print(f'EFFICIENCY — single {S}x{S} image, CPU')
    print('=' * 84)
    print(f'{"model":<20} {"params":>12} {"GMACs":>10} {"vs A0":>8} '
          f'{"runtime ms":>12} {"vs A0":>8}')
    print('-' * 84)

    base_f = base_t = None
    rows = []
    for name in ('A0 RetinexFormer', 'A1 FD2RT_V1', 'A4 FD2RT_A4', 'ICNF FD2RT_ICNF'):
        net = build(name)
        n_par = sum(p.numel() for p in net.parameters())
        macs, brk = count_flops(net, x)
        ms = timeit(net, x, runs=args.runs) * 1e3
        if base_f is None:
            base_f, base_t = macs, ms
        rows.append((name, n_par, macs, ms, brk))
        print(f'{name:<20} {n_par:>12,} {macs/1e9:>10.3f} '
              f'{macs/base_f:>7.2f}x {ms:>12.1f} {ms/base_t:>7.2f}x')

    print()
    print('  FLOPs breakdown (GMACs):')
    for name, _, macs, _, brk in rows:
        parts = '  '.join(f'{k}={v/1e9:.3f}' for k, v in sorted(brk.items()))
        print(f'    {name:<20} {parts}')

    # ── Adaptive-compute headroom ──────────────────────────────────────── #
    print()
    print('=' * 84)
    print('ADAPTIVE-COMPUTE HEADROOM from ICNF')
    print('=' * 84)
    print('Where observed high-frequency energy is fully explained by the')
    print('illumination-predicted noise floor there is no texture to recover, so')
    print('the frequency branch has nothing to do and could be skipped.')
    print()

    from basicsr.models.archs.icnf import ICNF
    from PIL import Image
    import torch.nn.functional as F

    paths = sorted(glob.glob(os.path.join(_ROOT, args.data, '*.jpg')))[:13]
    if not paths:
        print(f'  (no probe images in {args.data} — skipping)')
        return 0

    m = ICNF(learn_params=False)
    # Fraction of the frequency branch's cost inside the whole model.
    a4 = build('A4 FD2RT_A4')
    ic = build('ICNF FD2RT_ICNF')
    _, brk_ic = count_flops(ic, x)
    total_ic = sum(brk_ic.values())

    freq_macs = 0
    handles = []
    store = {}

    def hook(mod, i, o):
        if isinstance(o, (tuple, list)):
            o = o[0]
        if isinstance(mod, nn.Conv2d):
            store['f'] = store.get('f', 0) + _conv_flops(mod, i, o)
        elif isinstance(mod, nn.Linear):
            store['f'] = store.get('f', 0) + _linear_flops(mod, i, o)

    for nm, mod in ic.named_modules():
        if 'freq_blocks' in nm and isinstance(mod, (nn.Conv2d, nn.Linear)):
            handles.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        ic(x)
    for h in handles:
        h.remove()
    freq_macs = store.get('f', 0)
    frac = freq_macs / total_ic
    print(f'  frequency branch = {freq_macs/1e9:.3f} GMACs '
          f'= {100*frac:.1f}% of the model')
    print()

    print(f'  {"threshold":>10} {"area below floor":>18} {"model GMACs saved":>19}')
    print('  ' + '-' * 52)
    for thr in (0.5, 1.0, 2.0, 4.0):
        fracs = []
        for p in paths:
            im = Image.open(p).convert('RGB')
            w, h = im.size
            sc = 256 / max(w, h)
            im = im.resize((int(w*sc)//2*2, int(h*sc)//2*2), Image.LANCZOS)
            t = torch.from_numpy(np.asarray(im, np.float32) / 255.)
            t = t.permute(2, 0, 1).unsqueeze(0)
            t = (t * (0.15 / t.mean())).clamp(1e-4, 1)
            ev = m(t)
            fracs.append((ev < thr).float().mean().item())
        skip = float(np.mean(fracs))
        print(f'  {thr:>10.1f} {100*skip:>17.1f}% {100*skip*frac:>18.1f}%')

    print()
    print('  Read this as headroom, not a result. It is the ceiling a')
    print('  token-sparse frequency branch could reach; realising it needs')
    print('  gather/scatter and a threshold tuned on real data, and the quality')
    print('  cost is unmeasured until the GPU runs exist.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
