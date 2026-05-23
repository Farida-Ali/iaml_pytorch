"""
TASK 2 — Post-hoc frequency branch diagnostics for A4.

For each of the 15 LOL-v1 test images, at every DDA block, records:
  • gate value: sigmoid(gate_param)
  • L2 norm of A_spatial  (DDA_MSA output)
  • L2 norm of A_freq     (Freq_MSA output, before gate scaling)
  • ratio = norm(A_freq) / norm(A_spatial)

Interpretation:
  ratio ≈ 0  everywhere → freq branch never activated (collapsed to IG-MSA)
  ratio >> 1 → freq branch output dwarfs spatial; gate must suppress it
  gate < 0.1 → gate is suppressing the freq branch despite possible activity

Usage (GPU machine):
    python scripts/freq_branch_diagnostics.py \
        --weights experiments/FD2RT_A4_LOL_v1/models/net_g_best.pth \
        --data_root data/LOLv1/Test \
        --out_dir   results/diag
"""

import sys, os, json, argparse
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

import basicsr  # noqa
from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4, DDA_MSA, Freq_MSA

sys.path.insert(0, os.path.join(_ROOT, 'Enhancement'))
import utils as enh_utils


LOL_V1_KWARGS = dict(in_channels=3, out_channels=3, n_feat=40,
                     stage=1, num_blocks=[1, 2, 2])

BLOCK_LABELS = [
    ('enc0_block0',  'encoder_layers.0.0'),
    ('enc1_block0',  'encoder_layers.1.0'),
    ('enc1_block1',  'encoder_layers.1.0'),   # 2 inner blocks at enc1
    ('btn_block0',   'bottleneck'),
    ('btn_block1',   'bottleneck'),
    ('dec0_block0',  'decoder_layers.0.2'),
    ('dec0_block1',  'decoder_layers.0.2'),
    ('dec1_block0',  'decoder_layers.1.2'),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights',   required=True)
    p.add_argument('--data_root', default='data/LOLv1/Test')
    p.add_argument('--out_dir',   default='results/diag')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        'cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda')
    print(f'\nDevice : {device}')

    # ── Build & load model ────────────────────────────────────────────── #
    model = FD2RT_A4(**LOL_V1_KWARGS)
    ckpt  = torch.load(args.weights, map_location=device)
    state = ckpt.get('params', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    print(f'Checkpoint: {os.path.basename(args.weights)}')
    print(f'  Missing keys  : {len(missing)}')
    print(f'  Unexpected    : {len(unexpected)}')

    # ── Static: gate values (no inference needed) ─────────────────────── #
    print('\n── Gate values (static, from checkpoint weights) ──────────────')
    stage    = model.body[0].denoiser
    all_gates = {}

    for mod_path, dda_mod in [
        ('encoder_layers.0.0', stage.encoder_layers[0][0]),
        ('encoder_layers.1.0', stage.encoder_layers[1][0]),
        ('bottleneck',         stage.bottleneck),
        ('decoder_layers.0.2', stage.decoder_layers[0][2]),
        ('decoder_layers.1.2', stage.decoder_layers[1][2]),
    ]:
        for i, g in enumerate(dda_mod.gates):
            gate_val = torch.sigmoid(g).item()
            key = f'{mod_path}.inner{i}'
            all_gates[key] = gate_val
            print(f'  {key:45s}  gate = {gate_val:.4f}')

    # ── Dynamic: branch norms via forward hooks ────────────────────────── #
    spatial_norms = {}   # {(mod_path, inner_idx): [norm per image]}
    freq_norms    = {}

    # We instrument by walking DDA_Block_Dual and injecting per-call counters.
    # Since num_blocks may be >1, we need to track which inner block we're in.
    from basicsr.models.archs.fd2rt_a4_arch import DDA_Block_Dual

    hooks = []

    def _make_spatial_hook(label):
        def hook(m, inp, out):
            if label not in spatial_norms:
                spatial_norms[label] = []
            spatial_norms[label].append(out.detach().norm().item())
        return hook

    def _make_freq_hook(label):
        def hook(m, inp, out):
            if label not in freq_norms:
                freq_norms[label] = []
            freq_norms[label].append(out.detach().norm().item())
        return hook

    # Walk named modules; instrument each DDA_MSA and Freq_MSA
    for name, m in model.named_modules():
        if isinstance(m, DDA_MSA):
            hooks.append(m.register_forward_hook(_make_spatial_hook(name)))
        elif isinstance(m, Freq_MSA):
            hooks.append(m.register_forward_hook(_make_freq_hook(name)))

    # ── Test images ───────────────────────────────────────────────────── #
    lq_paths = natsorted(
        glob(os.path.join(args.data_root, 'input', '*.png')) +
        glob(os.path.join(args.data_root, 'input', '*.jpg')))
    gt_paths = natsorted(
        glob(os.path.join(args.data_root, 'target', '*.png')) +
        glob(os.path.join(args.data_root, 'target', '*.jpg')))
    n_img = len(lq_paths)
    print(f'\nTest images: {n_img}')

    psnr_list = []
    for lq_p, gt_p in zip(lq_paths, gt_paths):
        lq = np.float32(enh_utils.load_img(lq_p)) / 255.
        gt = np.float32(enh_utils.load_img(gt_p)) / 255.
        t  = torch.from_numpy(lq).permute(2, 0, 1).unsqueeze(0).to(device)
        h, w = t.shape[2], t.shape[3]
        padh, padw = (-h % 4), (-w % 4)
        if padh or padw:
            t = F.pad(t, (0, padw, 0, padh), 'reflect')
        with torch.inference_mode():
            out = model(t)[:, :, :h, :w]
        pred = torch.clamp(out, 0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)
        psnr_list.append(float(enh_utils.PSNR(gt, pred)))

    for h in hooks:
        h.remove()

    mean_psnr = float(np.mean(psnr_list))
    print(f'PSNR on this checkpoint: {mean_psnr:.4f} dB')

    # ── Report ────────────────────────────────────────────────────────── #
    print('\n── Branch norm statistics (mean over all test images) ──────────')
    print(f'{"Block / inner":50s}  {"Gate":>6}  {"S_norm":>10}  '
          f'{"F_norm":>10}  {"Ratio F/S":>10}  {"Assessment"}')
    print('─' * 115)

    # Align keys: spatial and freq modules share the same path prefix
    # spatial keys: body.0.denoiser.encoder_layers.0.0.blocks.0.0
    # freq keys:    body.0.denoiser.encoder_layers.0.0.freq_blocks.0
    rows = []
    for skey in sorted(spatial_norms.keys()):
        fkey = skey.replace('.blocks.', '.freq_blocks.') \
                   .replace('.0', '', 1)     # remove the trailing .0 (DDA_MSA index)
        # Reconstruct: body.0.denoiser.X.blocks.I.0 → body.0.denoiser.X.freq_blocks.I
        parts = skey.split('.')
        # find 'blocks' position
        try:
            bi = parts.index('blocks')
            inner_idx = parts[bi + 1]
            fkey = '.'.join(parts[:bi]) + f'.freq_blocks.{inner_idx}'
        except ValueError:
            fkey = None

        gate_key = '.'.join(skey.split('.')[:- 2])  # strip '.blocks.I.0'
        # find gate value
        gate_val = None
        for gk, gv in all_gates.items():
            if gk.startswith(gate_key.replace('body.0.denoiser.', '')):
                gate_val = gv
                break

        s_norms = spatial_norms[skey]
        f_norms = freq_norms.get(fkey, [0.0] * len(s_norms))
        s_mean  = float(np.mean(s_norms))
        f_mean  = float(np.mean(f_norms))
        ratio   = f_mean / (s_mean + 1e-8)

        if ratio < 0.01:
            assess = 'DEAD — freq branch never activated'
        elif ratio < 0.1:
            assess = 'weak — marginal contribution'
        elif ratio < 0.5:
            assess = 'active — moderate contribution'
        elif ratio < 2.0:
            assess = 'active — balanced with spatial'
        else:
            assess = 'DOMINANT — freq >> spatial'

        # Short label for the block
        label = skey.replace('body.0.denoiser.', '')
        print(f'{label:50s}  '
              f'{gate_val if gate_val is not None else float("nan"):>6.4f}  '
              f'{s_mean:>10.4f}  {f_mean:>10.4f}  {ratio:>10.4f}  {assess}')
        rows.append({
            'block': label,
            'gate': round(gate_val or 0, 4),
            'spatial_norm_mean': round(s_mean, 4),
            'freq_norm_mean':    round(f_mean, 4),
            'ratio_freq_spatial': round(ratio, 4),
            'assessment': assess,
        })

    # Summary
    ratios = [r['ratio_freq_spatial'] for r in rows]
    print(f'\n  Freq/Spatial norm ratio — min: {min(ratios):.4f}  '
          f'max: {max(ratios):.4f}  mean: {np.mean(ratios):.4f}')
    gates = list(all_gates.values())
    print(f'  Gate values          — min: {min(gates):.4f}  '
          f'max: {max(gates):.4f}  mean: {np.mean(gates):.4f}')
    print(f'\n  PSNR on this ckpt: {mean_psnr:.4f} dB')

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'freq_branch_diag.json')
    with open(out_path, 'w') as f:
        json.dump({
            'checkpoint': args.weights,
            'n_images': n_img,
            'psnr': round(mean_psnr, 4),
            'gate_stats': {
                'min': round(min(gates), 4),
                'max': round(max(gates), 4),
                'mean': round(float(np.mean(gates)), 4),
                'values': {k: round(v, 4) for k, v in all_gates.items()},
            },
            'ratio_stats': {
                'min':  round(float(min(ratios)),        4),
                'max':  round(float(max(ratios)),        4),
                'mean': round(float(np.mean(ratios)),    4),
            },
            'per_block': rows,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    print(f'\nSaved → {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
