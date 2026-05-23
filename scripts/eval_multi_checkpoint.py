"""
TASK 1 — Multi-checkpoint evaluation for reliable PSNR/SSIM measurement.

Evaluates N checkpoints (default: last 10 by modification time) for a given
model config and test directory, then reports per-checkpoint values plus
mean ± std.  Produces a CSV and a summary table.

Usage (GPU machine, from /workspace/fd2rt/Retinexformer):
    # Evaluate last 10 A1 checkpoints
    python scripts/eval_multi_checkpoint.py \
        --model  A1 \
        --arch   FD2RT_V1 \
        --ckpt_dir  experiments/FD2RT_V1_LOL_v1/models \
        --data_root data/LOLv1/Test \
        --out_dir   results/diag

    # Evaluate last 10 A4 checkpoints
    python scripts/eval_multi_checkpoint.py \
        --model  A4 \
        --arch   FD2RT_A4 \
        --ckpt_dir  experiments/FD2RT_A4_LOL_v1/models \
        --data_root data/LOLv1/Test \
        --out_dir   results/diag

Run all three in sequence to fill the comparison table.
"""

import sys, os, json, csv, argparse, time
from glob import glob
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from natsort import natsorted
from skimage import img_as_ubyte

sys.path.insert(0, os.path.join(_ROOT, 'Enhancement'))
import utils as enh_utils


# ── Arch registry ──────────────────────────────────────────────────────── #

LOL_V1_KWARGS = dict(in_channels=3, out_channels=3, n_feat=40,
                     stage=1, num_blocks=[1, 2, 2])

def _build_model(arch_name):
    import basicsr  # noqa — triggers arch auto-registration
    if arch_name == 'FD2RT_V1':
        from basicsr.models.archs.fd2rt_v1_arch import FD2RT_V1
        return FD2RT_V1(**LOL_V1_KWARGS)
    elif arch_name == 'FD2RT_A2':
        from basicsr.models.archs.fd2rt_a2_arch import FD2RT_A2
        return FD2RT_A2(**LOL_V1_KWARGS)
    elif arch_name == 'FD2RT_A4':
        from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4
        return FD2RT_A4(**LOL_V1_KWARGS)
    else:
        raise ValueError(f'Unknown arch: {arch_name}. '
                         f'Choices: FD2RT_V1, FD2RT_A2, FD2RT_A4')


def _load_state(path, model, device):
    ckpt  = torch.load(path, map_location=device)
    state = ckpt.get('params', ckpt.get('state_dict', ckpt))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    return model


def _infer(model, img_np_f32, device, factor=4):
    t = torch.from_numpy(img_np_f32).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, h, w = t.shape
    padh, padw = (-h % factor), (-w % factor)
    if padh or padw:
        t = F.pad(t, (0, padw, 0, padh), mode='reflect')
    with torch.inference_mode():
        out = model(t)
    out = out[:, :, :h, :w]
    return torch.clamp(out, 0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)


def eval_checkpoint(ckpt_path, model, device, lq_paths, gt_paths):
    """Returns (mean_psnr, mean_ssim, per_image_list)."""
    _load_state(ckpt_path, model, device)
    model.eval()
    psnr_list, ssim_list = [], []
    for lq_p, gt_p in zip(lq_paths, gt_paths):
        lq = np.float32(enh_utils.load_img(lq_p)) / 255.
        gt = np.float32(enh_utils.load_img(gt_p)) / 255.
        pred = _infer(model, lq, device)
        psnr_list.append(float(enh_utils.PSNR(gt, pred)))
        ssim_list.append(float(enh_utils.calculate_ssim(
            img_as_ubyte(gt), img_as_ubyte(pred))))
    return float(np.mean(psnr_list)), float(np.mean(ssim_list)), psnr_list


# ── CLI ────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',    required=True, help='Short name, e.g. A1, A2, A4')
    p.add_argument('--arch',     required=True,
                   choices=['FD2RT_V1', 'FD2RT_A2', 'FD2RT_A4'])
    p.add_argument('--ckpt_dir', required=True,
                   help='Directory containing net_g_*.pth files')
    p.add_argument('--data_root', default='data/LOLv1/Test')
    p.add_argument('--out_dir',   default='results/diag')
    p.add_argument('--n_ckpts',   type=int, default=10,
                   help='Use last N checkpoints (by iter number). Default 10.')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        'cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda')
    print(f'\nDevice : {device}')

    # Discover checkpoints — sort by iteration number embedded in filename
    all_ckpts = glob(os.path.join(args.ckpt_dir, 'net_g_*.pth'))
    def _iter_num(p):
        base = os.path.basename(p)
        for part in base.replace('.pth','').split('_'):
            try:
                return int(part)
            except ValueError:
                pass
        return 0
    all_ckpts = sorted(all_ckpts, key=_iter_num)
    ckpts = all_ckpts[-args.n_ckpts:]
    if not ckpts:
        print(f'ERROR: no net_g_*.pth found in {args.ckpt_dir}')
        return 1

    print(f'\nModel  : {args.model} ({args.arch})')
    print(f'Checkpoints found : {len(all_ckpts)}  |  using last {len(ckpts)}')
    for c in ckpts:
        print(f'  iter {_iter_num(c):>7d}  {os.path.basename(c)}')

    # Test images
    lq_paths = natsorted(
        glob(os.path.join(args.data_root, 'input', '*.png')) +
        glob(os.path.join(args.data_root, 'input', '*.jpg')))
    gt_paths = natsorted(
        glob(os.path.join(args.data_root, 'target', '*.png')) +
        glob(os.path.join(args.data_root, 'target', '*.jpg')))
    assert len(lq_paths) == len(gt_paths) > 0, \
        f'Test image mismatch: lq={len(lq_paths)} gt={len(gt_paths)}'
    print(f'Test images: {len(lq_paths)}')

    # Build model once, swap weights per checkpoint
    model = _build_model(args.arch).to(device)

    # Evaluate
    rows = []
    print(f'\n{"Iter":>8}  {"PSNR":>8}  {"SSIM":>8}  {"Time":>8}')
    print('-' * 40)
    t0 = time.perf_counter()
    for c in ckpts:
        it = _iter_num(c)
        tc = time.perf_counter()
        mp, ms, _ = eval_checkpoint(c, model, device, lq_paths, gt_paths)
        rows.append({'iter': it, 'psnr': round(mp, 4), 'ssim': round(ms, 4),
                     'ckpt': os.path.basename(c)})
        print(f'{it:>8d}  {mp:>8.4f}  {ms:>8.4f}  '
              f'{time.perf_counter()-tc:>6.1f}s')

    psnr_vals = [r['psnr'] for r in rows]
    ssim_vals = [r['ssim'] for r in rows]
    mean_p, std_p = float(np.mean(psnr_vals)), float(np.std(psnr_vals))
    mean_s, std_s = float(np.mean(ssim_vals)), float(np.std(ssim_vals))
    best_p = float(np.max(psnr_vals))
    best_it = rows[int(np.argmax(psnr_vals))]['iter']

    print(f'\n{"─"*40}')
    print(f'  Model   : {args.model} ({args.arch})')
    print(f'  N ckpts : {len(rows)}')
    print(f'  PSNR    : {mean_p:.4f} ± {std_p:.4f} dB  '
          f'[best {best_p:.4f} @ iter {best_it}]')
    print(f'  SSIM    : {mean_s:.4f} ± {std_s:.4f}')
    print(f'  Wall    : {time.perf_counter()-t0:.1f}s')

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_stem = os.path.join(args.out_dir, f'multickpt_{args.model}')

    with open(out_stem + '.json', 'w') as f:
        json.dump({
            'model': args.model,
            'arch': args.arch,
            'n_images': len(lq_paths),
            'n_checkpoints': len(rows),
            'psnr_mean': round(mean_p, 4),
            'psnr_std':  round(std_p,  4),
            'psnr_best': round(best_p, 4),
            'psnr_best_iter': best_it,
            'ssim_mean': round(mean_s, 4),
            'ssim_std':  round(std_s,  4),
            'per_checkpoint': rows,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    print(f'\nSaved → {out_stem}.json')

    # Append one-liner to shared comparison CSV
    csv_path = os.path.join(args.out_dir, 'comparison_table.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['Model', 'PSNR_mean', 'PSNR_std', 'PSNR_best',
                        'PSNR_best_iter', 'SSIM_mean', 'SSIM_std', 'N_ckpts'])
        w.writerow([args.model, round(mean_p,4), round(std_p,4),
                    round(best_p,4), best_it, round(mean_s,4), round(std_s,4),
                    len(rows)])
    print(f'Appended → {csv_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
