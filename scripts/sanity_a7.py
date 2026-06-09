"""
scripts/sanity_a7.py
────────────────────
Step-4 sanity check for FD²RT A7 and all sweep variants. Runs the real A7
training step (A4 net + LL hook + L_pix + L_freq + L_tv) on real LOL-v1
patches and logs every component plus its magnitude RATIO to the main L1 loss.

This mirrors what FD2RT_A7_Model.optimize_parameters does, in a lean loop so
it can run anywhere (CPU smoke test or GPU full run).

GPU full run (the real 2000-iter gate):
  python scripts/sanity_a7.py --device cuda --batch 8 --iters 2000 --log_every 200

A7-lite1 (freq_weight=0.05, w_high=1.5):
  python scripts/sanity_a7.py --freq_weight 0.05 --w_high 1.5 --label A7-lite1

A7-lite2 (freq_weight=0.02, w_high=1.0, tv_weight=0.005):
  python scripts/sanity_a7.py --freq_weight 0.02 --w_high 1.0 --tv_weight 0.005 --label A7-lite2

A7-tv-only (no freq loss):
  python scripts/sanity_a7.py --no_freq --label A7-tv-only

Quick local smoke (CPU, default):
  python scripts/sanity_a7.py

Success criteria (checked + printed at the end):
  (a) total loss decreases, no spike > 3× running median (after warmup)
  (b) weighted L_freq in ~2-30% of L1 ; weighted L_tv < 10% of L1 ; active terms > 0
  (c) no NaN in any component
"""
import sys, os, argparse, glob, random
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from PIL import Image
import torch

from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4
from basicsr.models.losses import FrequencyAwareLoss, IlluminationTVLoss

CKPT = ('/root/.claude/uploads/65d5d802-aad9-4807-87d2-a69bf439e318/'
        'e8684ae8-best_psnr_23.71_126000.pth')


def load_pairs(n, crop, seed=0):
    """Load up to n random LOL-v1 train pairs as a [n,3,crop,crop] tensor pair."""
    in_dir = os.path.join(_ROOT, 'data/LOLv1/Train/input')
    gt_dir = os.path.join(_ROOT, 'data/LOLv1/Train/target')
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(in_dir, '*.png')))
    rng = random.Random(seed)
    files = rng.sample(files, min(n, len(files)))
    lqs, gts = [], []
    for f in files:
        lq = np.asarray(Image.open(os.path.join(in_dir, f)).convert('RGB'), np.float32) / 255.
        gt = np.asarray(Image.open(os.path.join(gt_dir, f)).convert('RGB'), np.float32) / 255.
        H, W = lq.shape[:2]
        y = rng.randint(0, H - crop); x = rng.randint(0, W - crop)
        lqs.append(lq[y:y+crop, x:x+crop])
        gts.append(gt[y:y+crop, x:x+crop])
    to_t = lambda arr: torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).contiguous()
    return to_t(lqs), to_t(gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--iters', type=int, default=24)
    ap.add_argument('--log_every', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--no_ckpt', action='store_true',
                    help='start from random init instead of the A4 checkpoint')
    # Loss weight overrides for sweep variants
    ap.add_argument('--freq_weight', type=float, default=0.1,
                    help='loss_weight for FrequencyAwareLoss (default: 0.1)')
    ap.add_argument('--w_low', type=float, default=1.0,
                    help='w_low for FrequencyAwareLoss (default: 1.0)')
    ap.add_argument('--w_high', type=float, default=2.0,
                    help='w_high for FrequencyAwareLoss (default: 2.0)')
    ap.add_argument('--tv_weight', type=float, default=0.01,
                    help='loss_weight for IlluminationTVLoss (default: 0.01)')
    ap.add_argument('--no_freq', action='store_true',
                    help='disable L_freq entirely (tv-only ablation)')
    ap.add_argument('--label', default='',
                    help='optional variant label for printout (e.g. A7-lite1)')
    args = ap.parse_args()

    torch.manual_seed(100)
    device = args.device

    label = f' [{args.label}]' if args.label else ''
    print(f'[config{label}] freq_weight={args.freq_weight if not args.no_freq else "DISABLED"}'
          f'  w_low={args.w_low}  w_high={args.w_high}  tv_weight={args.tv_weight}')

    net = FD2RT_A4(in_channels=3, out_channels=3, n_feat=40,
                   stage=1, num_blocks=[1, 2, 2]).to(device)
    if not args.no_ckpt and os.path.isfile(CKPT):
        state = torch.load(CKPT, map_location=device).get('params')
        net.load_state_dict(state, strict=False)
        print(f'[init] loaded A4 checkpoint (strict=False)')
    else:
        print('[init] random init')

    # LL-capture hook (same matching rule as FD2RT_A7_Model)
    captured = {}
    for name, m in net.named_modules():
        if name.endswith('estimator.dwt'):
            m.register_forward_hook(lambda mod, i, o: captured.__setitem__('LL', o[0]))
            break

    cri_pix = torch.nn.L1Loss()
    cri_freq = (None if args.no_freq else
                FrequencyAwareLoss(loss_weight=args.freq_weight,
                                   w_low=args.w_low, w_high=args.w_high).to(device))
    cri_tv = IlluminationTVLoss(loss_weight=args.tv_weight).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    lq_all, gt_all = load_pairs(max(args.batch * 4, 16), args.crop)
    lq_all, gt_all = lq_all.to(device), gt_all.to(device)
    n_pool = lq_all.shape[0]

    net.train()
    freq_hdr = 'l_freq' if cri_freq is not None else '(no freq)'
    print(f'\n{"iter":>5} {"l_total":>10} {"l_pix":>10} {freq_hdr:>10} '
          f'{"l_tv":>10} {"freq/pix%":>10} {"tv/pix%":>9}')
    print('─' * 70)

    rows = []
    for it in range(1, args.iters + 1):
        idx = torch.randint(0, n_pool, (args.batch,))
        lq, gt = lq_all[idx], gt_all[idx]

        opt.zero_grad()
        captured.clear()
        out = net(lq)
        l_pix = cri_pix(out, gt)
        l_freq = cri_freq(out, gt) if cri_freq is not None else torch.zeros(1, device=device)
        l_tv = cri_tv(captured['LL'])
        l_total = l_pix + l_freq + l_tv
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        r = dict(it=it,
                 total=l_total.item(), pix=l_pix.item(),
                 freq=l_freq.item(), tv=l_tv.item(),
                 freq_ratio=100*l_freq.item()/l_pix.item(),
                 tv_ratio=100*l_tv.item()/l_pix.item())
        rows.append(r)
        if it % args.log_every == 0 or it == 1:
            print(f'{it:5d} {r["total"]:10.5f} {r["pix"]:10.5f} {r["freq"]:10.5f} '
                  f'{r["tv"]:10.5f} {r["freq_ratio"]:9.2f}% {r["tv_ratio"]:8.2f}%')

    # ── Evaluate success criteria ─────────────────────────────────────── #
    totals = [r['total'] for r in rows]
    finite = all(np.isfinite(v) for r in rows for v in
                 (r['total'], r['pix'], r['freq'], r['tv']))
    # spike test: no value > 3× running median (after a short warmup)
    warm = max(2, len(totals) // 5)
    med = np.median(totals[warm:]) if len(totals) > warm else np.median(totals)
    max_spike = max(totals[warm:]) if len(totals) > warm else max(totals)
    no_spike = max_spike <= 3 * med
    # trend: last-quarter mean < first-quarter mean
    q = max(1, len(totals)//4)
    decreasing = np.mean(totals[-q:]) <= np.mean(totals[:q])
    # ratio bands on the average over the run
    avg_freq_ratio = np.mean([r['freq_ratio'] for r in rows])
    avg_tv_ratio = np.mean([r['tv_ratio'] for r in rows])
    # freq band: 2-30% when active; treat as pass if freq is disabled
    freq_active = cri_freq is not None
    freq_band = (not freq_active) or (2.0 <= avg_freq_ratio <= 30.0)
    tv_band = avg_tv_ratio < 10.0
    # active check: only for enabled terms
    active_ok = ((not freq_active or all(r['freq'] > 0 for r in rows)) and
                 all(r['tv'] > 0 for r in rows))

    print('\n' + '═' * 70)
    print('SANITY CRITERIA')
    print('═' * 70)
    def line(lbl, ok, detail=''):
        print(f'  {"PASS" if ok else "FAIL"}  {lbl}  {detail}')
    line('(c) no NaN/Inf in any component', finite)
    line('(a) no spike > 3× median', no_spike,
         f'(max {max_spike:.4f} vs 3×median {3*med:.4f})')
    line('(a) total loss trending down', decreasing,
         f'(first-q {np.mean(totals[:q]):.4f} → last-q {np.mean(totals[-q:]):.4f})')
    if freq_active:
        line('(b) weighted L_freq in 2-30% of L1', freq_band,
             f'(avg {avg_freq_ratio:.2f}%)')
    else:
        line('(b) L_freq disabled (tv-only ablation)', True, '(skipped)')
    line('(b) weighted L_tv < 10% of L1', tv_band,
         f'(avg {avg_tv_ratio:.2f}%)')
    line('(b) active loss terms > 0', active_ok)

    passed = finite and no_spike and freq_band and tv_band and active_ok
    print('\n' + ('SANITY PASSED' if passed else 'SANITY NEEDS REVIEW'))
    if not (freq_band and tv_band):
        print('  → A weighted component is outside its target band; '
              'adjust freq_opt/tv_opt loss_weight before the full run.')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
