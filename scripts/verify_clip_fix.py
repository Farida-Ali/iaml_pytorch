"""
TASK 2 — 2000-iteration verification run after clip_grad_norm fix.

Runs a short training loop (2000 iters) on LOL-v1 train set using the A4
config, logging per-iteration:
  • l_pix loss
  • pre-clip gradient norm (before clip_grad_norm is applied)
  • freq branch L2 norm at two specific DDA_Block_Dual locations:
      – first encoder block  (encoder_layers.0.0)
      – bottleneck           (bottleneck)

PASS/FAIL criteria:
  A. Loss smoothness: no spike with l_pix > 3× running_median at any point
     after the first 200 iters (warm-up excluded).
  B. Freq branch activation: mean freq_norm > 0.0 for at least one
     of the two monitored blocks by iter 2000 (branch must wake up).
  C. Gradient scale: mean pre-clip grad norm across all iters > 0.05
     (confirms clip is no longer zeroing 99% of gradient information).

Usage (GPU machine, from /workspace/fd2rt/Retinexformer):
    python scripts/verify_clip_fix.py \
        --config Options/train_FD2RT_A4_LOL_v1.yml \
        --out_dir results/verify_clip \
        [--pretrain experiments/FD2RT_A4_LOL_v1/models/net_g_best.pth]

Output:
  results/verify_clip/verify_log.csv   — per-iter metrics
  results/verify_clip/verify_summary.json — PASS/FAIL verdict + gate values

DO NOT launch full training. Stop and report after 2000 iters.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from natsort import natsorted


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',    default='Options/train_FD2RT_A4_LOL_v1.yml')
    p.add_argument('--out_dir',   default='results/verify_clip')
    p.add_argument('--pretrain',  default=None,
                   help='Optional: path to net_g checkpoint to start from')
    p.add_argument('--n_iter',    type=int, default=2000,
                   help='Number of iterations to run (default 2000)')
    p.add_argument('--seed',      type=int, default=100)
    p.add_argument('--cpu',       action='store_true')
    return p.parse_args()


def _running_median(arr, window=50):
    out = np.empty(len(arr))
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = np.median(arr[lo:hi])
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda')
    print(f'\nDevice : {device}')
    print(f'Config : {args.config}')
    print(f'Iters  : {args.n_iter}')

    # ── Load config ──────────────────────────────────────────────────────── #
    import yaml
    with open(args.config) as f:
        opt = yaml.safe_load(f)

    clip_val = opt['train'].get('clip_grad_norm', 1.0)
    use_clip  = opt['train'].get('use_grad_clip', True)
    print(f'\nGradient clip: use_grad_clip={use_clip}, clip_grad_norm={clip_val}')
    if use_clip and clip_val <= 0.05:
        print(f'  WARNING: clip_grad_norm={clip_val} is still very aggressive. '
              f'Expected 1.0 after fix.')

    # ── Build model ──────────────────────────────────────────────────────── #
    import basicsr  # noqa
    from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4, DDA_MSA, Freq_MSA

    LOL_KWARGS = dict(in_channels=3, out_channels=3, n_feat=40,
                      stage=1, num_blocks=[1, 2, 2])
    model = FD2RT_A4(**LOL_KWARGS).to(device)

    if args.pretrain and os.path.isfile(args.pretrain):
        ckpt  = torch.load(args.pretrain, map_location=device)
        state = ckpt.get('params', ckpt.get('state_dict', ckpt))
        miss, unexp = model.load_state_dict(state, strict=False)
        print(f'Pretrain: {os.path.basename(args.pretrain)}  '
              f'missing={len(miss)}  unexpected={len(unexp)}')
    else:
        print('No pretrain checkpoint — starting from random init.')

    # ── Static: gate values ───────────────────────────────────────────────── #
    stage = model.body[0].denoiser
    gate_values = {}
    for mod_path, dda_mod in [
        ('encoder_layers.0.0', stage.encoder_layers[0][0]),
        ('encoder_layers.1.0', stage.encoder_layers[1][0]),
        ('bottleneck',         stage.bottleneck),
        ('decoder_layers.0.2', stage.decoder_layers[0][2]),
        ('decoder_layers.1.2', stage.decoder_layers[1][2]),
    ]:
        for i, g in enumerate(dda_mod.gates):
            gate_values[f'{mod_path}.inner{i}'] = round(torch.sigmoid(g).item(), 4)

    print('\n── Gate values (start of run) ──────────────────────────────────')
    for k, v in gate_values.items():
        print(f'  {k:45s}  {v:.4f}')

    # ── Freq branch hooks ────────────────────────────────────────────────── #
    freq_norms_enc0 = []   # encoder_layers.0.0.freq_blocks
    freq_norms_btn  = []   # bottleneck.freq_blocks

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, Freq_MSA):
            if 'encoder_layers.0' in name:
                def _h_enc(mod, inp, out, store=freq_norms_enc0):
                    store.append(out.detach().norm().item())
                hooks.append(m.register_forward_hook(_h_enc))
            elif 'bottleneck' in name:
                def _h_btn(mod, inp, out, store=freq_norms_btn):
                    store.append(out.detach().norm().item())
                hooks.append(m.register_forward_hook(_h_btn))

    # ── Optimizer ────────────────────────────────────────────────────────── #
    lr = opt['train']['optim_g'].get('lr', 2e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 betas=tuple(opt['train']['optim_g'].get('betas', [0.9, 0.999])))
    loss_fn = nn.L1Loss()

    # ── Dataset ───────────────────────────────────────────────────────────── #
    data_root_gt = opt['datasets']['train']['dataroot_gt']
    data_root_lq = opt['datasets']['train']['dataroot_lq']

    gt_paths = natsorted(glob(os.path.join(data_root_gt, '*.png')) +
                         glob(os.path.join(data_root_gt, '*.jpg')))
    lq_paths = natsorted(glob(os.path.join(data_root_lq, '*.png')) +
                         glob(os.path.join(data_root_lq, '*.jpg')))
    assert len(gt_paths) == len(lq_paths) > 0, \
        f'Dataset mismatch: gt={len(gt_paths)} lq={len(lq_paths)}'
    print(f'\nDataset: {len(gt_paths)} image pairs from {data_root_gt}')

    from PIL import Image
    import torchvision.transforms.functional as TF

    PATCH = 128
    BS    = 8

    def _load_pair_patch(gt_p, lq_p):
        gt = TF.to_tensor(Image.open(gt_p).convert('RGB'))
        lq = TF.to_tensor(Image.open(lq_p).convert('RGB'))
        _, H, W = gt.shape
        if H < PATCH or W < PATCH:
            gt = TF.resize(gt, (max(H, PATCH), max(W, PATCH)))
            lq = TF.resize(lq, (max(H, PATCH), max(W, PATCH)))
            _, H, W = gt.shape
        r = torch.randint(0, H - PATCH + 1, (1,)).item()
        c = torch.randint(0, W - PATCH + 1, (1,)).item()
        return lq[:, r:r+PATCH, c:c+PATCH], gt[:, r:r+PATCH, c:c+PATCH]

    # ── Training loop ─────────────────────────────────────────────────────── #
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'verify_log.csv')

    log_rows = []
    model.train()
    indices = list(range(len(gt_paths)))

    print(f'\n{"Iter":>6}  {"l_pix":>10}  {"grad_norm":>10}  '
          f'{"freq_enc0":>10}  {"freq_btn":>10}')
    print('─' * 58)

    t0 = time.perf_counter()
    enc0_ptr = 0
    btn_ptr  = 0
    iter_num = 0

    while iter_num < args.n_iter:
        # Build batch
        np.random.shuffle(indices)
        batch_lq, batch_gt = [], []
        for idx in indices[:BS]:
            lq_t, gt_t = _load_pair_patch(gt_paths[idx], lq_paths[idx])
            batch_lq.append(lq_t)
            batch_gt.append(gt_t)
        lq_b = torch.stack(batch_lq).to(device)
        gt_b = torch.stack(batch_gt).to(device)

        # Forward
        freq_norms_enc0_before = len(freq_norms_enc0)
        freq_norms_btn_before  = len(freq_norms_btn)

        pred = model(lq_b)
        # pred may be a tensor or tuple depending on stage output
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        loss = loss_fn(pred, gt_b)

        optimizer.zero_grad()
        loss.backward()

        # Pre-clip grad norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        if use_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

        optimizer.step()

        l_pix_val = loss.item()

        # Freq norms logged by hooks during this forward pass
        new_enc0 = freq_norms_enc0[freq_norms_enc0_before:]
        new_btn  = freq_norms_btn[freq_norms_btn_before:]
        fn_enc0  = float(np.mean(new_enc0)) if new_enc0 else 0.0
        fn_btn   = float(np.mean(new_btn))  if new_btn  else 0.0

        iter_num += 1
        row = {
            'iter':      iter_num,
            'l_pix':     round(l_pix_val, 6),
            'grad_norm': round(grad_norm, 4),
            'freq_enc0': round(fn_enc0,   6),
            'freq_btn':  round(fn_btn,    6),
        }
        log_rows.append(row)

        if iter_num % 100 == 0 or iter_num <= 10:
            elapsed = time.perf_counter() - t0
            print(f'{iter_num:>6d}  {l_pix_val:>10.5f}  {grad_norm:>10.4f}  '
                  f'{fn_enc0:>10.6f}  {fn_btn:>10.6f}  ({elapsed:.0f}s)')

    for h in hooks:
        h.remove()

    # ── Save CSV ──────────────────────────────────────────────────────────── #
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['iter', 'l_pix', 'grad_norm',
                                          'freq_enc0', 'freq_btn'])
        w.writeheader()
        w.writerows(log_rows)
    print(f'\nLog saved → {csv_path}')

    # ── PASS/FAIL evaluation ─────────────────────────────────────────────── #
    l_pix_arr    = np.array([r['l_pix']     for r in log_rows])
    grad_arr     = np.array([r['grad_norm'] for r in log_rows])
    freq_enc0arr = np.array([r['freq_enc0'] for r in log_rows])
    freq_btnarr  = np.array([r['freq_btn']  for r in log_rows])

    # Criterion A: no spike > 3× running median after first 200 iters
    warm_up_skip = min(200, args.n_iter // 10)
    post_warm_l  = l_pix_arr[warm_up_skip:]
    rm           = _running_median(post_warm_l, window=50)
    spike_mask   = post_warm_l > 3.0 * rm
    n_spikes     = int(spike_mask.sum())
    crit_a       = n_spikes == 0

    # Criterion B: at least one freq block has mean norm > 0.0
    mean_enc0 = float(freq_enc0arr.mean())
    mean_btn  = float(freq_btnarr.mean())
    crit_b    = (mean_enc0 > 0.0) or (mean_btn > 0.0)

    # Criterion C: mean pre-clip grad norm > 0.05
    mean_grad = float(grad_arr.mean())
    crit_c    = mean_grad > 0.05

    overall = 'PASS' if (crit_a and crit_b and crit_c) else 'FAIL'

    print('\n' + '=' * 60)
    print('VERIFICATION RESULTS')
    print('=' * 60)
    print(f'\n  clip_grad_norm applied : {clip_val}')
    print(f'\n  Criterion A — Loss smoothness (no spike >3× median after iter {warm_up_skip}):')
    print(f'    Spikes detected : {n_spikes}')
    print(f'    Result          : {"PASS" if crit_a else "FAIL"}')

    print(f'\n  Criterion B — Freq branch activation (mean norm > 0):')
    print(f'    freq_enc0 mean  : {mean_enc0:.6f}')
    print(f'    freq_btn  mean  : {mean_btn:.6f}')
    print(f'    Result          : {"PASS" if crit_b else "FAIL"}')

    print(f'\n  Criterion C — Gradient scale (mean grad norm > 0.05):')
    print(f'    Mean grad norm  : {mean_grad:.4f}')
    print(f'    Result          : {"PASS" if crit_c else "FAIL"}')

    print(f'\n  ── OVERALL : {overall} ──')

    print('\n── Gate values at START of run ─────────────────────────────────')
    for k, v in gate_values.items():
        print(f'  {k:45s}  {v:.4f}')

    # ── Save JSON ─────────────────────────────────────────────────────────── #
    summary = {
        'config':          args.config,
        'pretrain':        args.pretrain,
        'n_iter':          args.n_iter,
        'clip_grad_norm':  clip_val,
        'use_grad_clip':   use_clip,
        'criteria': {
            'A_loss_smoothness': {
                'n_spikes':      n_spikes,
                'warm_up_iters': warm_up_skip,
                'pass':          bool(crit_a),
            },
            'B_freq_activation': {
                'freq_enc0_mean': round(mean_enc0, 6),
                'freq_btn_mean':  round(mean_btn,  6),
                'pass':           bool(crit_b),
            },
            'C_gradient_scale': {
                'mean_grad_norm': round(mean_grad, 4),
                'pass':           bool(crit_c),
            },
        },
        'overall': overall,
        'gate_values_at_start': gate_values,
        'loss_stats': {
            'mean': round(float(l_pix_arr.mean()), 6),
            'std':  round(float(l_pix_arr.std()),  6),
            'min':  round(float(l_pix_arr.min()),  6),
            'max':  round(float(l_pix_arr.max()),  6),
        },
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    json_path = os.path.join(args.out_dir, 'verify_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary saved → {json_path}')
    print('\n*** DO NOT launch full training. Report results and wait for approval. ***')
    return 0 if overall == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
