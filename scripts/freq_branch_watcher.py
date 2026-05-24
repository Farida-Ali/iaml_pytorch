"""
Checkpoint watcher for freq branch diagnostics during A4 training.

Runs in a SEPARATE terminal alongside training. Polls the experiment's
models/ directory for new checkpoints (saved every 5K iters). When a
new checkpoint appears, it:
  1. Loads the model from the checkpoint
  2. Reads gate values (static from weights)
  3. Runs a tiny forward pass on one test image to record branch norms
  4. Parses the training log for l_pix at that iteration
  5. Appends one row to freq_branch_log.csv

Exits automatically when no new checkpoint appears within --timeout
minutes (default 60) — i.e., training has ended.

Usage (in a second terminal, from /workspace/fd2rt/Retinexformer):
    python scripts/freq_branch_watcher.py \
        --exp_dir experiments/train_FD2RT_A4_LOL_v1_fixed \
        --data_root data/LOLv1/Test \
        --poll_sec 120

Outputs:
    experiments/train_FD2RT_A4_LOL_v1_fixed/freq_branch_log.csv
"""

import sys, os, csv, time, argparse, re
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


LOL_V1_KWARGS = dict(in_channels=3, out_channels=3, n_feat=40,
                     stage=1, num_blocks=[1, 2, 2])

_LOG_RE = re.compile(r'iter:\s*(\d+).*?l_pix:\s*([\d.e+\-]+)')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir',   required=True,
                   help='Experiment root, e.g. experiments/train_FD2RT_A4_LOL_v1_fixed')
    p.add_argument('--data_root', default='data/LOLv1/Test',
                   help='Test data root with input/ subdir (one image is enough)')
    p.add_argument('--poll_sec',  type=int, default=120,
                   help='Seconds between polls (default 120)')
    p.add_argument('--timeout',   type=int, default=60,
                   help='Minutes without a new checkpoint before watcher exits (default 60)')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


def _iter_from_filename(path):
    base = os.path.basename(path)
    for part in base.replace('.pth', '').split('_'):
        try:
            return int(part)
        except ValueError:
            pass
    return -1


def _load_model(ckpt_path, device):
    import basicsr  # noqa
    from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4
    model = FD2RT_A4(**LOL_V1_KWARGS).to(device).eval()
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('params', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    return model


def _get_gates(model):
    stage = model.body[0].denoiser
    gate_vals = []
    for dda_mod in [
        stage.encoder_layers[0][0],
        stage.encoder_layers[1][0],
        stage.bottleneck,
        stage.decoder_layers[0][2],
        stage.decoder_layers[1][2],
    ]:
        for g in dda_mod.gates:
            gate_vals.append(torch.sigmoid(g).item())
    return gate_vals


def _get_branch_norms(model, lq_tensor, device):
    from basicsr.models.archs.fd2rt_a4_arch import DDA_MSA, Freq_MSA

    spatial_norms = []
    freq_norms    = []
    hooks = []

    def _s_hook(m, inp, out):
        spatial_norms.append(out.detach().norm().item())

    def _f_hook(m, inp, out):
        freq_norms.append(out.detach().norm().item())

    for name, m in model.named_modules():
        if isinstance(m, DDA_MSA):
            hooks.append(m.register_forward_hook(_s_hook))
        elif isinstance(m, Freq_MSA):
            hooks.append(m.register_forward_hook(_f_hook))

    with torch.inference_mode():
        model(lq_tensor)

    for h in hooks:
        h.remove()

    s_mean = float(np.mean(spatial_norms)) if spatial_norms else 0.0
    f_mean = float(np.mean(freq_norms))    if freq_norms    else 0.0
    return s_mean, f_mean


def _parse_lpix_from_log(log_dir, target_iter):
    """Return l_pix value logged at or just before target_iter, or None."""
    log_files = sorted(glob(os.path.join(log_dir, '*.log')))
    if not log_files:
        return None
    best_iter, best_val = -1, None
    for log_path in log_files:
        try:
            with open(log_path) as f:
                for line in f:
                    m = _LOG_RE.search(line)
                    if m:
                        it = int(m.group(1))
                        if it <= target_iter and it > best_iter:
                            best_iter = it
                            best_val  = float(m.group(2))
        except Exception:
            pass
    return best_val


def main():
    args   = parse_args()
    device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda')

    models_dir = os.path.join(args.exp_dir, 'models')
    csv_path   = os.path.join(args.exp_dir, 'freq_branch_log.csv')

    # One test image for branch-norm forward pass
    lq_paths = natsorted(
        glob(os.path.join(args.data_root, 'input', '*.png')) +
        glob(os.path.join(args.data_root, 'input', '*.jpg')))
    if not lq_paths:
        print(f'ERROR: no images under {args.data_root}/input/ — needed for branch norms.')
        return 1
    lq_img_path = lq_paths[0]

    # Pre-load one test image
    sys.path.insert(0, os.path.join(_ROOT, 'Enhancement'))
    import utils as enh_utils
    lq_np = np.float32(enh_utils.load_img(lq_img_path)) / 255.
    lq_t  = torch.from_numpy(lq_np).permute(2, 0, 1).unsqueeze(0).to(device)
    h, w  = lq_t.shape[2], lq_t.shape[3]
    padh, padw = (-h % 4), (-w % 4)
    if padh or padw:
        lq_t = F.pad(lq_t, (0, padw, 0, padh), 'reflect')

    print(f'\nWatcher started')
    print(f'  Experiment : {args.exp_dir}')
    print(f'  CSV output : {csv_path}')
    print(f'  Test image : {os.path.basename(lq_img_path)}')
    print(f'  Device     : {device}')
    print(f'  Poll every : {args.poll_sec}s')
    print(f'  Timeout    : {args.timeout}min after last new checkpoint\n')

    # Set up CSV
    csv_exists = os.path.isfile(csv_path)
    csv_file   = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(['iter', 'l_pix', 'gate_mean', 'freq_norm_mean',
                             'spatial_norm_mean', 'timestamp'])
        csv_file.flush()

    processed   = set()
    last_new_ck = time.time()
    timeout_sec = args.timeout * 60

    # Re-scan already existing checkpoints on startup (in case watcher is
    # started mid-training or restarted)
    existing = sorted(glob(os.path.join(models_dir, 'net_g_*.pth')),
                      key=_iter_from_filename)
    if existing:
        print(f'Found {len(existing)} existing checkpoints — processing all.')

    try:
        while True:
            all_ckpts = sorted(glob(os.path.join(models_dir, 'net_g_*.pth')),
                               key=_iter_from_filename)
            new_ckpts = [c for c in all_ckpts if c not in processed]

            if new_ckpts:
                last_new_ck = time.time()

            for ckpt_path in new_ckpts:
                it = _iter_from_filename(ckpt_path)
                print(f'[{datetime.now().strftime("%H:%M:%S")}] '
                      f'Processing checkpoint iter {it} ...')

                try:
                    model = _load_model(ckpt_path, device)

                    gates    = _get_gates(model)
                    gate_mean = round(float(np.mean(gates)), 4)

                    s_mean, f_mean = _get_branch_norms(model, lq_t, device)

                    l_pix = _parse_lpix_from_log(args.exp_dir, it)

                    csv_writer.writerow([
                        it,
                        round(l_pix, 6) if l_pix is not None else '',
                        gate_mean,
                        round(f_mean, 6),
                        round(s_mean, 6),
                        datetime.now(timezone.utc).isoformat(),
                    ])
                    csv_file.flush()

                    print(f'  iter={it:>7d}  l_pix={l_pix:.5f if l_pix else "N/A":>10}  '
                          f'gate={gate_mean:.4f}  '
                          f'freq_norm={f_mean:.4f}  spatial_norm={s_mean:.4f}')

                    del model
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f'  ERROR processing {os.path.basename(ckpt_path)}: {e}')

                processed.add(ckpt_path)

            # Check timeout
            idle_sec = time.time() - last_new_ck
            if idle_sec > timeout_sec and processed:
                print(f'\nNo new checkpoint in {args.timeout}min — training likely complete.')
                break

            time.sleep(args.poll_sec)

    finally:
        csv_file.close()

    # Print summary
    print(f'\n── Freq branch log summary ─────────────────────────────────────')
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if rows:
        iters     = [int(r['iter'])              for r in rows if r['iter']]
        gate_vals = [float(r['gate_mean'])        for r in rows if r['gate_mean']]
        freq_vals = [float(r['freq_norm_mean'])   for r in rows if r['freq_norm_mean']]
        spat_vals = [float(r['spatial_norm_mean'])for r in rows if r['spatial_norm_mean']]

        print(f'  Checkpoints logged : {len(rows)}')
        print(f'  Iter range         : {iters[0]} → {iters[-1]}')
        print(f'  Gate mean          : start={gate_vals[0]:.4f}  end={gate_vals[-1]:.4f}  '
              f'min={min(gate_vals):.4f}  max={max(gate_vals):.4f}')
        print(f'  Freq norm mean     : start={freq_vals[0]:.4f}  end={freq_vals[-1]:.4f}  '
              f'min={min(freq_vals):.4f}  max={max(freq_vals):.4f}')
        print(f'  Spatial norm mean  : start={spat_vals[0]:.4f}  end={spat_vals[-1]:.4f}')

        gate_delta = gate_vals[-1] - gate_vals[0]
        freq_delta = freq_vals[-1] - freq_vals[0]
        if abs(gate_delta) < 0.02 and abs(freq_delta) < 0.1:
            print('\n  *** Gate and freq norm barely moved — freq branch may have plateaued.')
        elif gate_vals[-1] > gate_vals[0]:
            print('\n  Gate increased — network learned to rely more on freq branch.')
        else:
            print('\n  Gate decreased — network suppressed freq branch over training.')

    print(f'\nCSV saved → {csv_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
