"""
Single-image debug script — validates that RetinexFormer pretrained weights
load and run correctly before doing full-dataset evaluation.

Usage:
    python3 test_single_image_debug.py \
        --opt Options/RetinexFormer_LOL_v1.yml \
        --weights pretrained_weights/LOL_v1.pth \
        --input  data/LOLv1/Test/input/00001.png \
        --target data/LOLv1/Test/target/00001.png \
        --out_dir debug_output
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# ── args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--opt',     required=True)
parser.add_argument('--weights', required=True)
parser.add_argument('--input',   required=True)
parser.add_argument('--target',  required=True)
parser.add_argument('--out_dir', default='debug_output')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── load YAML → model type ────────────────────────────────────────────────────
raw = yaml.load(open(args.opt), Loader=Loader)
model_type = raw['network_g']['type']
print(f"Model type from YAML: {model_type}")

# ── load model + weights ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if model_type == 'IAMLNet':
    from basicsr.archs.iaml_arch import IAMLNet
    ckpt = torch.load(args.weights, map_location='cpu')
    if   'params'           in ckpt: sd = ckpt['params']
    elif 'model_state_dict' in ckpt: sd = ckpt['model_state_dict']
    elif 'state_dict'       in ckpt: sd = ckpt['state_dict']
    else:                            sd = ckpt
    sd = {k: v for k, v in sd.items() if not k.startswith('teacher_decoder')}
    model = IAMLNet()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  IAMLNet — missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    run = model.inference
else:
    from basicsr.utils.options import parse
    from basicsr.models import create_model
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False
    model = create_model(opt).net_g
    ckpt = torch.load(args.weights, map_location='cpu')
    try:
        model.load_state_dict(ckpt['params'])
        print(f"  {model_type} — weights loaded via 'params' key")
    except Exception:
        prefixed = {'module.' + k: v for k, v in ckpt['params'].items()}
        model.load_state_dict(prefixed)
        print(f"  {model_type} — weights loaded via 'module.' prefix")
    run = model

model.cuda()
model.eval()

# ── load images ───────────────────────────────────────────────────────────────
def load_rgb_f32(path):
    img = cv2.imread(path)
    assert img is not None, f"Cannot read: {path}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

inp_np  = load_rgb_f32(args.input)
tgt_np  = load_rgb_f32(args.target)
print(f"Input shape: {inp_np.shape}  range [{inp_np.min():.3f}, {inp_np.max():.3f}]")
print(f"Target shape: {tgt_np.shape}")

# ── inference ─────────────────────────────────────────────────────────────────
inp_t = torch.from_numpy(inp_np).permute(2, 0, 1).unsqueeze(0).cuda()

factor = 4
h, w = inp_t.shape[2], inp_t.shape[3]
ph = (factor - h % factor) % factor
pw = (factor - w % factor) % factor
if ph or pw:
    inp_t = F.pad(inp_t, (0, pw, 0, ph), 'reflect')

with torch.no_grad():
    out_t = run(inp_t)

out_t = out_t[:, :, :h, :w]
out_np = torch.clamp(out_t, 0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()

# ── metrics ───────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Enhancement'))
import utils as enh_utils
from skimage import img_as_ubyte

psnr = enh_utils.PSNR(tgt_np, out_np)
ssim = enh_utils.calculate_ssim(img_as_ubyte(tgt_np), img_as_ubyte(out_np))
print(f"\nSingle-image result:")
print(f"  PSNR: {psnr:.4f} dB")
print(f"  SSIM: {ssim:.6f}")
print(f"  Expected for a working model: 20–30 dB")

# ── save side-by-side ─────────────────────────────────────────────────────────
def to_u8(arr):
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

row = np.concatenate([to_u8(inp_np), to_u8(out_np), to_u8(tgt_np)], axis=1)
row_bgr = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
out_path = os.path.join(args.out_dir, 'debug_input_enhanced_target.png')
cv2.imwrite(out_path, row_bgr)
print(f"\nSaved side-by-side (input | enhanced | target): {out_path}")
