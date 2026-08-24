"""
TASK 3 — Robust multi-checkpoint PSNR/SSIM evaluation.

Evaluates the last N checkpoints (default N=10) for a given model and
reports mean ± std PSNR and SSIM.  A single checkpoint on 15 LOL-v1 images
has ~0.3 dB noise; averaging across N checkpoints reduces this.

Usage (GPU machine, from /workspace/fd2rt/Retinexformer):
    # A1 / FD2RT_V1
    python scripts/robust_eval.py \
        --arch   FD2RT_V1 \
        --ckpt_dir  experiments/FD2RT_V1_LOL_v1/models \
        --data_root data/LOLv1/Test \
        --label  A1 \
        --out_dir   results/robust_eval

    # A4
    python scripts/robust_eval.py \
        --arch   FD2RT_A4 \
        --ckpt_dir  experiments/FD2RT_A4_LOL_v1/models \
        --data_root data/LOLv1/Test \
        --label  A4 \
        --out_dir   results/robust_eval

    # Retinexformer baseline (uses the original arch)
    python scripts/robust_eval.py \
        --arch   RetinexFormer \
        --ckpt_dir  experiments/RetinexFormer_LOL_v1/models \
        --data_root data/LOLv1/Test \
        --label  RetinexFormer \
        --out_dir   results/robust_eval

Output:
    results/robust_eval/<label>_robust.json  — full per-checkpoint data
    results/robust_eval/comparison_table.csv — one row appended per run
    stdout: summary table with mean ± std
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

sys.path.insert(0, os.path.join(_ROOT, 'Enhancement'))
import utils as enh_utils


# ── Architecture registry ──────────────────────────────────────────────── #

LOL_V1_KWARGS = dict(in_channels=3, out_channels=3, n_feat=40,
                     stage=1, num_blocks=[1, 2, 2])


def _build_model(arch_name, extra_kwargs=None):
    """Build an architecture by name.

    extra_kwargs lets Mode B pass the config's network_g settings through -- this
    matters for FD2RT_ICNF, whose forward path depends on illum_source/icnf_mode.
    Evaluating an ICNFc checkpoint with the wrong illum_source silently uses a
    different forward path and reports a meaningless number.
    """
    import basicsr  # noqa
    kw = dict(LOL_V1_KWARGS)
    if extra_kwargs:
        kw.update(extra_kwargs)
    if arch_name == 'FD2RT_V1':
        from basicsr.models.archs.fd2rt_v1_arch import FD2RT_V1
        return FD2RT_V1(**LOL_V1_KWARGS)
    elif arch_name == 'FD2RT_A2':
        from basicsr.models.archs.fd2rt_a2_arch import FD2RT_A2
        return FD2RT_A2(**LOL_V1_KWARGS)
    elif arch_name == 'FD2RT_A4':
        from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4
        return FD2RT_A4(**LOL_V1_KWARGS)
    elif arch_name == 'FD2RT_ICNF':
        from basicsr.models.archs.fd2rt_icnf_arch import FD2RT_ICNF
        # Default to the trained configuration; Mode B overrides from the config.
        kw.setdefault('illum_source', 'constant')
        return FD2RT_ICNF(**kw)
    elif arch_name == 'RetinexFormer':
        from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
        return RetinexFormer(**LOL_V1_KWARGS)
    else:
        raise ValueError(f'Unknown arch: {arch_name}. Choices: FD2RT_V1, '
                         f'FD2RT_A2, FD2RT_A4, FD2RT_ICNF, RetinexFormer')


def _net_kwargs_from_config(config_path):
    """Return the non-type keys of network_g (e.g. illum_source for ICNF)."""
    import yaml
    with open(config_path) as f:
        opt = yaml.safe_load(f)
    ng = dict(opt.get('network_g', {}))
    ng.pop('type', None)
    # Keep only kwargs the arch constructors accept generically; the ICNF ones
    # (illum_source, icnf_mode, a_init, ...) pass straight through.
    return ng


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
    if isinstance(out, (list, tuple)):
        out = out[-1]
    out = out[:, :, :h, :w]
    return torch.clamp(out, 0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)


def _iter_from_filename(path):
    base = os.path.basename(path)
    for part in base.replace('.pth', '').split('_'):
        try:
            return int(part)
        except ValueError:
            pass
    return 0


def eval_checkpoint(ckpt_path, model, device, lq_paths, gt_paths):
    """Returns (mean_psnr, mean_ssim, per_image_psnr_list, per_image_ssim_list)."""
    from skimage import img_as_ubyte
    _load_state(ckpt_path, model, device)
    model.eval()
    psnr_list, ssim_list = [], []
    for lq_p, gt_p in zip(lq_paths, gt_paths):
        lq   = np.float32(enh_utils.load_img(lq_p)) / 255.
        gt   = np.float32(enh_utils.load_img(gt_p)) / 255.
        pred = _infer(model, lq, device)
        psnr_list.append(float(enh_utils.PSNR(gt, pred)))
        ssim_list.append(float(enh_utils.calculate_ssim(
            img_as_ubyte(gt), img_as_ubyte(pred))))
    return (float(np.mean(psnr_list)), float(np.mean(ssim_list)),
            psnr_list, ssim_list)


# ── Dataset roots for known presets ───────────────────────────────────── #
_DATASET_ROOTS = {
    'LOL_v1':        'data/LOLv1/Test',
    'LOL_v2_real':   'data/LOL_v2_real/Test',
    'LOL_v2_synth':  'data/LOL_v2_synthetic/Test',
}


# ── CLI ────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(
        description='Robust multi-checkpoint PSNR/SSIM evaluation.\n'
                    'Accepts either (--arch + --ckpt_dir) OR (--config + --exp_dir).')

    # Mode A: explicit arch + ckpt dir
    p.add_argument('--arch',
                   choices=['FD2RT_V1', 'FD2RT_A2', 'FD2RT_A4', 'FD2RT_ICNF',
                            'RetinexFormer'],
                   help='Architecture name (Mode A)')
    p.add_argument('--ckpt_dir',
                   help='Directory containing net_g_*.pth files (Mode A)')

    # Mode B: derive arch + paths from a training config
    p.add_argument('--config',
                   help='Path to training YAML config (Mode B). '
                        'Arch and data paths are read from the config.')
    p.add_argument('--exp_dir',
                   help='Experiment root directory, e.g. '
                        'experiments/train_FD2RT_A4_LOL_v1_fixed (Mode B). '
                        'Checkpoints are loaded from <exp_dir>/models/.')

    # Shared
    p.add_argument('--label',    default=None,
                   help='Short label for output files (default: arch name or config name)')
    p.add_argument('--data_root', default=None,
                   help='Test data root with input/ and target/ subdirs. '
                        'Overrides --dataset and config data paths.')
    p.add_argument('--dataset',  default=None, choices=list(_DATASET_ROOTS.keys()),
                   help='Named dataset preset (overrides config val paths)')
    p.add_argument('--out_dir',   default='results/robust_eval')
    p.add_argument('--n_ckpts',   type=int, default=10,
                   help='Number of latest checkpoints to average (default 10)')
    p.add_argument('--best_only', action='store_true',
                   help='Only evaluate net_g_best.pth (overrides --n_ckpts)')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


def _arch_from_config(config_path):
    """Read arch type string from a basicsr YAML config."""
    import yaml
    with open(config_path) as f:
        opt = yaml.safe_load(f)
    return opt['network_g']['type']


def _data_root_from_config(config_path):
    """Read val data root from a basicsr YAML config."""
    import yaml
    with open(config_path) as f:
        opt = yaml.safe_load(f)
    gt_root = opt.get('datasets', {}).get('val', {}).get('dataroot_gt', '')
    # Strip /target or /input suffix to get the Test root
    for suffix in ['/target', '/input']:
        if gt_root.endswith(suffix):
            return gt_root[:-len(suffix)]
    return os.path.dirname(gt_root)


def main():
    args = parse_args()

    # Resolve mode
    net_extra = None
    if args.config and args.exp_dir:
        arch_name = _arch_from_config(args.config)
        net_extra = _net_kwargs_from_config(args.config)
        ckpt_dir  = os.path.join(args.exp_dir, 'models')
        label     = args.label or os.path.basename(args.exp_dir)
        if args.data_root:
            data_root = args.data_root
        elif args.dataset:
            data_root = _DATASET_ROOTS[args.dataset]
        else:
            data_root = _data_root_from_config(args.config)
        out_dir = args.out_dir or os.path.join(args.exp_dir, 'robust_eval')
    elif args.arch and args.ckpt_dir:
        arch_name = args.arch
        ckpt_dir  = args.ckpt_dir
        label     = args.label or args.arch
        data_root = (args.data_root or
                     (_DATASET_ROOTS[args.dataset] if args.dataset else 'data/LOLv1/Test'))
        out_dir   = args.out_dir
    else:
        print('ERROR: provide either (--arch + --ckpt_dir) or (--config + --exp_dir).')
        return 1

    device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda')
    print(f'\nDevice  : {device}')
    print(f'Arch    : {arch_name}')
    print(f'Label   : {label}')
    print(f'Ckpt dir: {ckpt_dir}')
    print(f'Data    : {data_root}')

    # Patch args for rest of function
    args.arch     = arch_name
    args.ckpt_dir = ckpt_dir
    args.out_dir  = out_dir
    args.data_root = data_root

    # Discover checkpoints
    if args.best_only:
        best_path = os.path.join(args.ckpt_dir, 'net_g_best.pth')
        if not os.path.isfile(best_path):
            print(f'ERROR: net_g_best.pth not found in {args.ckpt_dir}')
            return 1
        ckpts = [best_path]
    else:
        all_ckpts = sorted(
            glob(os.path.join(args.ckpt_dir, 'net_g_*.pth')),
            key=_iter_from_filename,
        )
        if not all_ckpts:
            print(f'ERROR: no net_g_*.pth found in {args.ckpt_dir}')
            return 1
        ckpts = all_ckpts[-args.n_ckpts:]

    print(f'\nCheckpoints found : {len(glob(os.path.join(args.ckpt_dir, "net_g_*.pth")))}')
    print(f'Checkpoints used  : {len(ckpts)}')
    for c in ckpts:
        it = _iter_from_filename(c)
        label_s = f'iter {it:>7d}' if it else 'best'
        print(f'  {label_s}  {os.path.basename(c)}')

    # Test images. Different LOL variants name the subdirs differently:
    #   LOL-v1        : input/ (lq)  target/ (gt)
    #   LOL-v2-real   : Low/   (lq)  Normal/ (gt)
    # Auto-detect so the same command works on both.
    def _find(root, names):
        for n in names:
            hits = natsorted(glob(os.path.join(root, n, '*.png')) +
                             glob(os.path.join(root, n, '*.jpg')) +
                             glob(os.path.join(root, n, '*.bmp')))
            if hits:
                return hits, n
        return [], names[0]
    lq_paths, lq_sub = _find(args.data_root, ['input', 'Low', 'low'])
    gt_paths, gt_sub = _find(args.data_root, ['target', 'Normal', 'normal', 'high'])
    if len(lq_paths) == 0:
        print(f'ERROR: no test images under {args.data_root}/ '
              f'(looked for input|Low and target|Normal subdirs)')
        return 1
    print(f'  lq subdir: {lq_sub}   gt subdir: {gt_sub}   ({len(lq_paths)} images)')
    if len(lq_paths) != len(gt_paths):
        print(f'ERROR: image count mismatch: lq={len(lq_paths)} gt={len(gt_paths)}')
        return 1
    print(f'Test images : {len(lq_paths)}')

    # Build model once, swap weights per checkpoint
    model = _build_model(args.arch, extra_kwargs=net_extra).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Params      : {n_params:,}')

    # Evaluate
    rows = []
    print(f'\n{"Ckpt":>12}  {"PSNR":>8}  {"SSIM":>8}  {"Time":>7}')
    print('─' * 44)
    t0 = time.perf_counter()

    for c in ckpts:
        it = _iter_from_filename(c)
        tc = time.perf_counter()
        mp, ms, psnr_per_img, ssim_per_img = eval_checkpoint(
            c, model, device, lq_paths, gt_paths)
        elapsed_c = time.perf_counter() - tc
        label_s = f'{it:>12d}' if it else '        best'
        print(f'{label_s}  {mp:>8.4f}  {ms:>8.4f}  {elapsed_c:>6.1f}s')
        rows.append({
            'ckpt':       os.path.basename(c),
            'iter':       it,
            'psnr':       round(mp, 4),
            'ssim':       round(ms, 4),
            'psnr_per_image': [round(v, 4) for v in psnr_per_img],
            'ssim_per_image': [round(v, 4) for v in ssim_per_img],
        })

    psnr_vals = [r['psnr'] for r in rows]
    ssim_vals = [r['ssim'] for r in rows]
    mean_p = float(np.mean(psnr_vals))
    std_p  = float(np.std(psnr_vals))
    best_p = float(np.max(psnr_vals))
    best_iter = rows[int(np.argmax(psnr_vals))]['iter']
    mean_s = float(np.mean(ssim_vals))
    std_s  = float(np.std(ssim_vals))
    total_t = time.perf_counter() - t0

    print(f'\n{"─"*44}')
    print(f'  Model      : {label} ({args.arch})')
    print(f'  N ckpts    : {len(rows)}')
    print(f'  N images   : {len(lq_paths)}')
    print(f'  PSNR mean  : {mean_p:.4f} ± {std_p:.4f} dB')
    print(f'  PSNR best  : {best_p:.4f} dB  (iter {best_iter})')
    print(f'  SSIM mean  : {mean_s:.4f} ± {std_s:.4f}')
    print(f'  Wall time  : {total_t:.1f}s')

    # Per-image PSNR averaged across all checkpoints
    n_img = len(lq_paths)
    per_img_means = []
    for i in range(n_img):
        vals = [r['psnr_per_image'][i] for r in rows]
        per_img_means.append(round(float(np.mean(vals)), 4))
    worst3  = sorted(range(n_img), key=lambda i: per_img_means[i])[:3]
    best3   = sorted(range(n_img), key=lambda i: per_img_means[i])[-3:][::-1]
    print(f'\n  Per-image PSNR (avg over {len(rows)} ckpts):')
    print(f'    Best  3: ' + '  '.join(
        f'img{i:02d}={per_img_means[i]:.4f}' for i in best3))
    print(f'    Worst 3: ' + '  '.join(
        f'img{i:02d}={per_img_means[i]:.4f}' for i in worst3))

    # Save JSON
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f'{label}_robust.json')
    with open(json_path, 'w') as f:
        json.dump({
            'label':          label,
            'arch':           args.arch,
            'ckpt_dir':       args.ckpt_dir,
            'n_images':       len(lq_paths),
            'n_checkpoints':  len(rows),
            'psnr_mean':      round(mean_p, 4),
            'psnr_std':       round(std_p,  4),
            'psnr_best':      round(best_p, 4),
            'psnr_best_iter': best_iter,
            'ssim_mean':      round(mean_s, 4),
            'ssim_std':       round(std_s,  4),
            'per_image_psnr_mean': per_img_means,
            'per_checkpoint': rows,
            'timestamp':      datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    print(f'\nSaved → {json_path}')

    # Append to comparison CSV
    csv_path = os.path.join(args.out_dir, 'comparison_table.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['Label', 'Arch', 'N_ckpts', 'N_images',
                        'PSNR_mean', 'PSNR_std', 'PSNR_best', 'PSNR_best_iter',
                        'SSIM_mean', 'SSIM_std', 'Timestamp'])
        w.writerow([label, args.arch, len(rows), len(lq_paths),
                    round(mean_p, 4), round(std_p, 4),
                    round(best_p, 4), best_iter,
                    round(mean_s, 4), round(std_s, 4),
                    datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')])
    print(f'Appended → {csv_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
