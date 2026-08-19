"""
PILOT: controlled A4-vs-ICNF quality comparison on real image content.

WHY THIS EXISTS
Every ICNF result so far measures the MECHANISM (does the evidence map track
texture?) not the OUTCOME (does the model restore better?). Only a GPU run on
LOL-v1 can answer the outcome question at benchmark scale, but a scaled-down
controlled experiment can say which way the effect points -- and can catch the
disaster case where ICNF actively hurts, before GPU-weeks are spent.

SETUP
  content     : 74 real low-light photographs, downsampled 3x so the averaging
                suppresses their own sensor noise and gives a clean reference
  degradation : re-expose to a low mean, then inject Poisson-Gaussian noise
                (var = a*I + b) -- the standard sensor model
  target      : the clean reference
  models      : FD2RT_A4 vs FD2RT_ICNF, identical everything else
  seeds       : several, because a single small run is noise

HONEST CAVEAT, stated up front
The injected degradation is exactly the noise model ICNF assumes. That makes
this a FAVOURABLE test: it asks "does the mechanism work when its assumption
holds", not "does the assumption hold on LOL-v1". A win here is necessary, not
sufficient. A loss here would be decisive. The --mismatch flag re-runs with a
noise model ICNF is NOT tuned for, which is the harder question.

  python scripts/pilot_a4_vs_icnf.py --iters 1200 --seeds 3
"""
import sys, os, glob, math, time, argparse, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4
from basicsr.models.archs.fd2rt_icnf_arch import FD2RT_ICNF


def load_corpus(root, max_side=256, ds=3, limit=None):
    """Real photos -> clean references [N,3,H,W] (variable size, cropped later)."""
    paths = sorted(glob.glob(os.path.join(root, '*.jpg')) +
                   glob.glob(os.path.join(root, '*.bmp')) +
                   glob.glob(os.path.join(root, '*.png')))
    if limit:
        paths = paths[:limit]
    out = []
    for p in paths:
        try:
            im = Image.open(p).convert('RGB')
        except Exception:
            continue
        w, h = im.size
        s = max_side / max(w, h)
        if s < 1:
            im = im.resize((max(2, int(w * s)), max(2, int(h * s))), Image.LANCZOS)
        t = torch.from_numpy(np.asarray(im, np.float32) / 255.)
        t = t.permute(2, 0, 1).unsqueeze(0)
        t = F.avg_pool2d(t, ds)                 # suppress the photo's own noise
        if min(t.shape[-2:]) < 64:
            continue
        out.append(t.squeeze(0).clamp(1e-4, 1))
    return out


def degrade(clean, a, b, target_mean, gen, mismatch=False):
    """Re-expose to low light and inject sensor noise."""
    scale = target_mean / clean.mean().clamp_min(1e-6)
    dark = (clean * scale).clamp(1e-4, 1)
    if mismatch:
        # Noise ICNF is NOT tuned for: signal-INDEPENDENT heavy-tailed noise
        # plus a mild blur, breaking the var = a*I + b assumption.
        n = torch.randn(dark.shape, generator=gen) * math.sqrt(a * target_mean + b)
        n = n * (1.0 + 2.0 * torch.rand(dark.shape, generator=gen) ** 3)
        obs = dark + n
    else:
        var = a * dark + b
        obs = dark + var.sqrt() * torch.randn(dark.shape, generator=gen)
    return obs.clamp(1e-4, 1)


def crops(imgs, n, size, gen):
    """Random paired crops."""
    lq, gt = [], []
    idx = torch.randint(0, len(imgs), (n,), generator=gen)
    for i in idx:
        c = imgs[int(i)]
        H, W = c.shape[-2:]
        if H < size or W < size:
            c = F.interpolate(c.unsqueeze(0), size=(max(size, H), max(size, W)),
                              mode='bilinear', align_corners=False).squeeze(0)
            H, W = c.shape[-2:]
        y = int(torch.randint(0, H - size + 1, (1,), generator=gen))
        x = int(torch.randint(0, W - size + 1, (1,), generator=gen))
        gt.append(c[:, y:y + size, x:x + size])
    return torch.stack(gt)


def psnr(pred, target):
    mse = ((pred - target) ** 2).flatten(1).mean(1).clamp_min(1e-12)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def run_one(model_name, seed, train_imgs, val_lq, val_gt, args):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    kw = dict(in_channels=3, out_channels=3, n_feat=args.n_feat,
              stage=1, num_blocks=[1, 1, 1])
    if model_name == 'A4':
        net = FD2RT_A4(**kw)
    elif model_name == 'A4nz':
        # CONTROL ARM. ICNF differs from A4 in two ways at once: the spatial
        # evidence gate (the mechanism under test) AND a nonzero out_proj init,
        # which ICNF needs because a zero-init frequency branch makes its gate
        # unidentifiable. Without this arm, a win could be attributed to either.
        # A4nz is plain A4 with only the init changed, so ICNF-vs-A4nz isolates
        # the gate mechanism and A4nz-vs-A4 isolates the init.
        net = FD2RT_A4(**kw)
        for nm, mod in net.named_modules():
            if 'freq_blocks' in nm and nm.endswith('out_proj'):
                torch.nn.init.normal_(mod.weight, mean=0.0, std=1e-3)
    elif model_name == 'ICNFc':
        # CONTROL ARM. Full ICNF except the noise floor is spatially UNIFORM
        # (illum_source='constant'): same per-pixel gate, but conditioned only on
        # the HF energy, not on illumination. ICNF-vs-ICNFc isolates whether
        # conditioning the floor on illumination adds anything over a bare
        # adaptive gate — the question the mismatched-noise result forced open.
        net = FD2RT_ICNF(**kw, illum_source='constant')
    else:
        net = FD2RT_ICNF(**kw)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters,
                                                       eta_min=args.lr * 0.01)
    net.train()
    t0 = time.time()
    for it in range(1, args.iters + 1):
        gt = crops(train_imgs, args.batch, args.patch, gen)
        lq = degrade(gt, args.a, args.b, args.mean, gen, args.mismatch)
        opt.zero_grad()
        loss = F.l1_loss(net(lq), gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if args.verbose and it % max(1, args.iters // 6) == 0:
            print(f'      [{model_name} s{seed}] it {it:5d}  loss {loss.item():.5f}',
                  flush=True)

    net.eval()
    with torch.no_grad():
        vals = [psnr(net(l.unsqueeze(0)), g.unsqueeze(0))
                for l, g in zip(val_lq, val_gt)]
    return float(np.mean(vals)), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/tmp/corpus')
    ap.add_argument('--iters', type=int, default=1200)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--patch', type=int, default=64)
    ap.add_argument('--n_feat', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--a', type=float, default=0.02)
    ap.add_argument('--b', type=float, default=1e-5)
    ap.add_argument('--mean', type=float, default=0.12)
    ap.add_argument('--mismatch', action='store_true')
    ap.add_argument('--arms', default='A4,ICNF',
                    help='comma list from A4,A4nz,ICNF. A4nz is the init control.')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--out', default='results/pilot_a4_vs_icnf.json')
    args = ap.parse_args()

    imgs = load_corpus(args.data)
    if len(imgs) < 20:
        print(f'Only {len(imgs)} usable images in {args.data}; need >=20.')
        return 1
    n_val = max(8, len(imgs) // 5)
    val_imgs, train_imgs = imgs[:n_val], imgs[n_val:]

    # Fixed validation pairs — identical for every model and seed.
    vgen = torch.Generator().manual_seed(12345)
    # Crop to a multiple of 16: the U-Net downsamples twice and Freq_MSA applies
    # a DWT to the features, so the input must stay even at every level.
    M = 16
    val_gt = [c[:, :c.shape[-2] // M * M, :c.shape[-1] // M * M] for c in val_imgs]
    val_gt = [g for g in val_gt if min(g.shape[-2:]) >= M]
    val_lq = [degrade(g, args.a, args.b, args.mean, vgen, args.mismatch)
              for g in val_gt]

    print('=' * 80)
    print('PILOT — A4 vs ICNF on real image content')
    print('=' * 80)
    print(f'  train images : {len(train_imgs)}    val images: {len(val_gt)}')
    print(f'  iters {args.iters}  batch {args.batch}  patch {args.patch}  '
          f'n_feat {args.n_feat}  seeds {args.seeds}')
    print(f'  degradation  : mean {args.mean}, '
          f'{"MISMATCHED (heavy-tailed, signal-independent)" if args.mismatch else f"Poisson-Gaussian a={args.a} b={args.b}"}')
    if not args.mismatch:
        print('  NOTE: this degradation matches ICNF\'s assumption exactly, so it')
        print('        is a favourable test. A win is necessary, not sufficient.')
    print()

    arms = args.arms.split(',')
    results = {a: [] for a in arms}
    for seed in range(args.seeds):
        for name in arms:
            p, secs = run_one(name, seed, train_imgs, val_lq, val_gt, args)
            results[name].append(p)
            print(f'  seed {seed}  {name:<5} val PSNR {p:7.4f} dB   ({secs:.0f}s)',
                  flush=True)

    print()
    print('=' * 80)
    print(f'{"model":<8} {"PSNR mean":>11} {"std":>8} {"runs":>6}')
    print('-' * 80)
    for a in arms:
        v = np.array(results[a])
        print(f'{a:<8} {v.mean():>11.4f} {v.std():>8.4f} {len(v):>6}')
    print('-' * 80)
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            u, v = np.array(results[arms[i]]), np.array(results[arms[j]])
            print(f'  {arms[j]} - {arms[i]:<6}: {v.mean()-u.mean():+.4f} dB   '
                  f'(paired wins {int((v > u).sum())}/{len(u)})')
    a4 = np.array(results[arms[0]]); ic = np.array(results[arms[-1]])
    d = ic.mean() - a4.mean()
    wins = int((ic > a4).sum())
    print('-' * 80)
    print(f'  delta (ICNF - A4) : {d:+.4f} dB')
    print(f'  paired wins       : {wins}/{len(a4)} seeds')
    pooled = math.sqrt((a4.std(ddof=1)**2 + ic.std(ddof=1)**2) / 2) if len(a4) > 1 else float('nan')
    if len(a4) > 1 and pooled > 0:
        print(f'  effect size       : {d/pooled:+.2f} pooled sd')
    print()
    if d > 0 and wins > len(a4) / 2:
        print('  DIRECTION: ICNF ahead. Necessary but not sufficient — confirm on LOL-v1.')
    elif d < 0:
        print('  DIRECTION: ICNF BEHIND. Investigate before spending GPU time.')
    else:
        print('  DIRECTION: inconclusive at this scale.')

    os.makedirs(os.path.dirname(os.path.join(_ROOT, args.out)), exist_ok=True)
    with open(os.path.join(_ROOT, args.out), 'w') as f:
        json.dump({'args': vars(args), 'results': results,
                   'delta_db': d, 'wins': wins}, f, indent=2)
    print(f'\n  wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
