"""
Phase-3b: validate ICNF on REAL image statistics, not synthetic 1/f textures.

Why this matters. Every ICNF number so far came from synthetic scenes whose
texture was 1/f noise. Natural images have heavier-tailed gradients, edges,
and structured illumination, and a statistic that works on 1/f can fail on
photographs. This is the weakest link in the evidence chain, so it gets its
own test.

PROTOCOL (real content, known degradation, honest ground truth)
  1. Take a real low-light photograph.
  2. Downsample 3x. Averaging 9 pixels cuts the sensor noise ~3x, giving a
     usably CLEAN reference with genuine natural-image structure.
  3. Derive the texture mask from that CLEAN reference: pixels whose clean
     high-frequency energy is in the top tercile are "textured", bottom tercile
     "flat". The mask never sees the noisy image, so it is not circular.
  4. Re-impose a low-light exposure and inject Poisson-Gaussian noise at a
     KNOWN (a, b).
  5. Ask each statistic to recover the texture mask from the degraded image.

Also checks the MAD estimator on real content -- real texture inflates
median(|HH|), so this is where the classical estimator is most likely to break.

  python scripts/test_p3b_real_images.py --data /tmp/realimg
"""
import sys, os, glob, argparse
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from basicsr.models.archs.icnf import ICNF

RESULTS = []


def check(label, ok, detail=''):
    RESULTS.append((label, bool(ok), detail))
    print(f'  {"PASS" if ok else "FAIL"}  {label}' + (f'  {detail}' if detail else ''))
    return bool(ok)


def load_clean(path, max_side=384, ds=3):
    """Load, downsample ds x (noise reduction), return [1,1,H,W] luminance."""
    im = Image.open(path).convert('RGB')
    w, h = im.size
    s = max_side / max(w, h)
    if s < 1:
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255.
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
    t = F.avg_pool2d(t, ds)                      # the denoising step
    y = (0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]).unsqueeze(1)
    H, W = y.shape[-2:]
    return y[:, :, :H // 2 * 2, :W // 2 * 2].clamp(1e-4, 1)


def auc(scores, mask_pos, mask_neg):
    s_p = scores[mask_pos].flatten()
    s_n = scores[mask_neg].flatten()
    if len(s_p) == 0 or len(s_n) == 0:
        return float('nan')
    allv = torch.cat([s_p, s_n])
    order = torch.argsort(allv)
    ranks = torch.empty_like(allv); ranks[order] = torch.arange(len(allv), dtype=allv.dtype)
    n_p = len(s_p)
    return ((ranks[:n_p].sum() - n_p * (n_p - 1) / 2) / (n_p * len(s_n))).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/tmp/realimg')
    ap.add_argument('--a', type=float, default=0.02)
    ap.add_argument('--b', type=float, default=1e-5)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, '*.jpg')) +
                   glob.glob(os.path.join(args.data, '*.png')))
    if not paths:
        print(f'No images in {args.data}'); return 1

    m = ICNF(a_init=args.a, b_init=args.b, learn_params=False)
    torch.manual_seed(0)

    print('=' * 84)
    print(f'P3b — ICNF on REAL image content  ({len(paths)} photographs)')
    print('=' * 84)
    print(f'{"image":<16} {"illum range":>12} {"raw HF":>9} {"SNR-style":>10} '
          f'{"ICNF":>9} {"gain":>8}')
    print('-' * 84)

    rows = []
    for p in paths:
        clean = load_clean(p)
        H, W = clean.shape[-2:]

        # Texture mask from the CLEAN reference.
        #
        # It must measure REFLECTANCE variation, not brightness. Taking
        # terciles of raw clean HF energy separates bright from dark on any
        # photo with deep shadows, which makes the mask a proxy for
        # illumination and hands the win to any brightness detector. Under
        # Retinex the physical quantity of interest is R = I / L, which is
        # illumination-independent by construction. Estimating L by a heavy
        # blur of the CLEAN image and taking terciles of HF energy of R gives
        # ground truth that is about texture and nothing else. This is not
        # circular: it uses the Retinex definition on the clean reference, and
        # never touches the noisy image or the noise floor.
        L_clean = F.interpolate(F.avg_pool2d(clean, 8), size=(H, W),
                                mode='bilinear', align_corners=False)
        R_clean = (clean / L_clean.clamp_min(1e-3)).clamp(0, 4)
        e_clean = m.hf_energy(R_clean)
        e_up = F.interpolate(e_clean, size=(H, W), mode='bilinear', align_corners=False)
        lo, hi = torch.quantile(e_up, torch.tensor([0.33, 0.67]))
        pos, neg = e_up > hi, e_up < lo

        # Re-impose low light + inject KNOWN Poisson-Gaussian noise.
        scale = 0.15 / clean.mean().clamp_min(1e-6)
        dark = (clean * scale).clamp(1e-4, 1)
        var = args.a * dark + args.b
        obs = (dark + var.sqrt() * torch.randn_like(dark)).clamp(1e-4, 1)

        # Illumination range actually present in this photograph.
        illum = m._local_mean(obs)
        q = torch.quantile(illum, torch.tensor([0.05, 0.95]))
        irange = (q[1] / q[0].clamp_min(1e-4)).item()

        e_obs = m.hf_energy(obs)
        raw = F.interpolate(e_obs, size=(H, W), mode='bilinear', align_corners=False)
        blur = m._local_mean(obs)
        snr = obs.abs() / (obs - blur).abs().clamp_min(1e-4)
        icnf = m(obs)

        a_raw, a_snr, a_ic = (auc(raw, pos, neg), auc(snr, pos, neg),
                              auc(icnf, pos, neg))
        rows.append((os.path.basename(p), irange, a_raw, a_snr, a_ic))
        print(f'{os.path.basename(p):<16} {irange:>11.1f}x {a_raw:>9.4f} '
              f'{a_snr:>10.4f} {a_ic:>9.4f} {a_ic - a_raw:>+8.4f}')

    n = len(rows)
    mr = sum(r[2] for r in rows) / n
    ms = sum(r[3] for r in rows) / n
    mi = sum(r[4] for r in rows) / n
    print('-' * 84)
    print(f'{"MEAN":<16} {"":>12} {mr:>9.4f} {ms:>10.4f} {mi:>9.4f} {mi-mr:>+8.4f}')
    print()

    check('ICNF beats raw HF energy on real photographs (mean AUC)', mi > mr,
          f'({mi:.4f} vs {mr:.4f}, +{mi-mr:.4f})')
    check('ICNF beats the SNR-Net-style heuristic', mi > ms,
          f'({mi:.4f} vs {ms:.4f})')
    win = sum(1 for r in rows if r[4] > r[2])
    check('ICNF wins on the majority of images', win > n / 2, f'({win}/{n})')

    # Images where the baseline is already at ceiling carry no headroom, so a
    # "gain" there measures nothing. Report the non-saturated subset separately.
    ns = [r for r in rows if r[2] < 0.95]
    if ns:
        nr = sum(r[2] for r in ns) / len(ns)
        ni = sum(r[4] for r in ns) / len(ns)
        nw = sum(1 for r in ns if r[4] > r[2])
        print(f'\n  Non-saturated subset (raw HF < 0.95): {len(ns)}/{n} images')
        print(f'    raw HF {nr:.4f}  ->  ICNF {ni:.4f}   ({ni-nr:+.4f}), '
              f'ICNF wins {nw}/{len(ns)}')
        check('ICNF wins where the baseline has headroom', ni > nr and nw > len(ns)/2,
              f'({ni:.4f} vs {nr:.4f}, {nw}/{len(ns)})')
    check('ICNF is well above chance on real content', mi > 0.65,
          f'(mean AUC {mi:.4f})')

    # ── Mechanism test, CONTROLLED ──────────────────────────────────────
    # A cross-image correlation of gain against illumination range cannot test
    # the mechanism: between photographs the illumination range, the texture
    # content, the scene type and the effective noise level all vary at once,
    # and images where the baseline saturates contribute meaningless gains. The
    # synthetic study held everything fixed but illumination range; the real
    # equivalent is to do the same WITHIN each image -- same content, same
    # noise model, only the illumination ramp changes.
    print('\n' + '=' * 84)
    print('MECHANISM (controlled): same real image, increasing illumination range')
    print('=' * 84)
    print(f'{"ramp":>10} {"illum range":>13} {"raw HF":>9} {"ICNF":>9} {"gain":>9}')
    print('-' * 84)

    ramp_rows = []
    for ramp, desc in [(0.0, '1:1'), (0.6, '4:1'), (0.85, '12:1'), (0.95, '39:1')]:
        rr, ii, ranges_seen = [], [], []
        for p in paths:
            clean = load_clean(p)
            H, W = clean.shape[-2:]
            L_clean = F.interpolate(F.avg_pool2d(clean, 8), size=(H, W),
                                    mode='bilinear', align_corners=False)
            R_clean = (clean / L_clean.clamp_min(1e-3)).clamp(0, 4)
            e_up = F.interpolate(m.hf_energy(R_clean), size=(H, W),
                                 mode='bilinear', align_corners=False)
            lo, hi = torch.quantile(e_up, torch.tensor([0.33, 0.67]))
            pos, neg = e_up > hi, e_up < lo

            # Impose a synthetic illumination ramp of controlled steepness on
            # top of the real content, then the same noise model as before.
            yy = torch.linspace(0, 1, H).view(1, 1, -1, 1).expand(1, 1, H, W)
            ramp_field = (1.0 - ramp) + 2.0 * ramp * yy
            lit = clean * ramp_field
            lit = (lit * (0.15 / lit.mean().clamp_min(1e-6))).clamp(1e-4, 1)
            var = args.a * lit + args.b
            obs = (lit + var.sqrt() * torch.randn_like(lit)).clamp(1e-4, 1)

            il = m._local_mean(obs)
            q = torch.quantile(il, torch.tensor([0.05, 0.95]))
            ranges_seen.append((q[1] / q[0].clamp_min(1e-4)).item())

            raw = F.interpolate(m.hf_energy(obs), size=(H, W),
                                mode='bilinear', align_corners=False)
            rr.append(auc(raw, pos, neg)); ii.append(auc(m(obs), pos, neg))
        mr_, mi_ = sum(rr)/len(rr), sum(ii)/len(ii)
        med_range = float(np.median(ranges_seen))
        ramp_rows.append((desc, med_range, mr_, mi_, mi_ - mr_))
        print(f'{desc:>10} {med_range:>12.1f}x {mr_:>9.4f} {mi_:>9.4f} {mi_-mr_:>+9.4f}')

    # WHAT THIS CAN AND CANNOT SHOW.
    #
    # The synthetic study swept a genuinely flat 1:1 illumination up to 19:1 and
    # found the ICNF advantage GROWING with range (+0.023 -> +0.227), which is
    # what identified illumination conditioning as the mechanism.
    #
    # That sweep cannot be reproduced on photographs, because photographs are
    # never flat: at the mildest ramp here the real images already carry ~28x
    # illumination range. This test therefore explores 28x -> 55x, entirely
    # inside the regime where the synthetic curve had already saturated, and it
    # shows exactly that -- a large, roughly flat advantage (+0.23 to +0.32)
    # rather than a growing one.
    #
    # So the honest claim for the paper is NOT "the gain grows with illumination
    # range on real data". It is: real low-light photographs already sit in the
    # regime where ICNF wins decisively, and the advantage is stable across
    # illumination conditions. The monotone-growth evidence identifying the
    # mechanism rests on the synthetic sweep, where flat illumination is
    # reachable. Assert that, not more.
    gains_r = [r[4] for r in ramp_rows]
    check('ICNF beats raw HF at every illumination range',
          all(g > 0 for g in gains_r),
          f'(min gain {min(gains_r):+.4f})')
    check('advantage is large and stable across illumination range',
          min(gains_r) > 0.15 and (max(gains_r) - min(gains_r)) < 0.20,
          f'(range {min(gains_r):+.4f} to {max(gains_r):+.4f})')
    check('real photographs already sit in the high-illumination-range regime',
          ramp_rows[0][1] > 10.0,
          f'({ramp_rows[0][1]:.1f}x even at the mildest ramp — '
          f'flat illumination is unreachable with real content)')

    # MAD on real content: natural texture inflates median|HH|.
    print('\n' + '=' * 84)
    print('MAD sigma estimator on REAL content (texture biases it upward)')
    print('=' * 84)
    print(f'{"true sigma":>11} {"est (real img)":>16} {"ratio":>8}')
    print('-' * 84)
    ratios = []
    clean = load_clean(paths[0])
    for s in (0.01, 0.02, 0.05):
        noisy = (clean + s * torch.randn_like(clean)).clamp(1e-4, 1)
        est = m.estimate_sigma_mad(noisy).mean().item()
        ratios.append(est / s)
        print(f'{s:>11.4f} {est:>16.5f} {est/s:>7.2f}x')
    check('MAD stays within 2x of truth on real content',
          max(ratios) < 2.0,
          f'(worst {max(ratios):.2f}x — texture inflates it, as expected)')

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print('\n' + '=' * 84)
    print(f'PHASE 3b: {n_pass}/{len(RESULTS)} passed')
    print('=' * 84)
    for lbl, ok, det in RESULTS:
        if not ok:
            print(f'  FAILED: {lbl} {det}')
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
