"""
Compute FLOPs, parameters, and inference time for IAML and RetinexFormer.

Addresses reviewer comment on computational complexity.

Usage (CPU, for parameter counting only):
    python3 compute_complexity.py

Usage (GPU, for full timing benchmark):
    python3 compute_complexity.py --gpu

Output example:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                  Complexity Report @ 600×400 input                  │
    ├────────────────┬────────────┬────────────┬──────────┬───────────────┤
    │ Model          │ Train Params│ Infer Params│ GMACs   │ Infer Time ms │
    ├────────────────┼────────────┼────────────┼──────────┼───────────────┤
    │ RetinexFormer  │   1.61 M   │   1.61 M   │  15.57 G │         X ms  │
    │ IAML (ours)    │   X.XX M   │   X.XX M   │  XX.XX G │         X ms  │
    └────────────────┴────────────┴────────────┴──────────┴───────────────┘
"""
import argparse
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',    action='store_true', help='Use CUDA for timing')
parser.add_argument('--height', type=int, default=400, help='Input height (default: 400)')
parser.add_argument('--width',  type=int, default=600, help='Input width  (default: 600)')
parser.add_argument('--warmup', type=int, default=10,  help='Warm-up iterations for timing')
parser.add_argument('--reps',   type=int, default=50,  help='Timing repetitions')
args = parser.parse_args()

device = torch.device('cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu')
H, W = args.height, args.width
print(f"\nDevice: {device}   Input: {H}×{W}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: count parameters
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model):
    """Returns (total, trainable) parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
# Helper: GMACs via thop
# Measures only the inference-time forward path.
# ─────────────────────────────────────────────────────────────────────────────

def measure_gmacs_thop(inference_module, h, w, dev):
    """Use thop to count MACs for one (1, 3, H, W) forward pass."""
    from thop import profile as thop_profile
    x = torch.randn(1, 3, h, w).to(dev)
    macs, _ = thop_profile(inference_module, inputs=(x,), verbose=False)
    return macs / 1e9   # GMACs


# ─────────────────────────────────────────────────────────────────────────────
# Helper: inference time (ms / image)
# ─────────────────────────────────────────────────────────────────────────────

def measure_time_ms(inference_fn, h, w, dev, warmup=10, reps=50):
    """Measures mean wall-clock time (ms) for inference_fn(tensor)."""
    x = torch.randn(1, 3, h, w).to(dev)

    # pad to multiple of 32 if needed (IAML requirement)
    ph = (32 - h % 32) % 32
    pw = (32 - w % 32) % 32
    if ph or pw:
        x_padded = F.pad(x, (0, pw, 0, ph), 'reflect')
    else:
        x_padded = x

    # warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = inference_fn(x_padded)

    if dev.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(reps):
            if dev.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                inference_fn(x_padded)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                inference_fn(x_padded)
                times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


# ─────────────────────────────────────────────────────────────────────────────
# IAML
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("IAML")
print("=" * 60)

from basicsr.archs.iaml_arch import IAMLNet

iaml = IAMLNet().to(device).eval()

total_iaml, trainable_iaml = count_params(iaml)

# Inference uses encoder + student_decoder only (teacher_decoder excluded)
infer_iaml = sum(
    p.numel() for name, p in iaml.named_parameters()
    if not name.startswith('teacher_decoder')
)

print(f"  Total params (encoder+student+teacher): {total_iaml/1e6:.3f} M")
print(f"  Trainable params (encoder+student):     {trainable_iaml/1e6:.3f} M")
print(f"  Inference params (encoder+student):     {infer_iaml/1e6:.3f} M")
print(f"  Teacher-decoder params (frozen, not used at inference): "
      f"{(total_iaml - infer_iaml)/1e6:.3f} M")

# Wrap inference path for thop (pad handled inside inference())
class IAMLInferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder         = model.encoder
        self.student_decoder = model.student_decoder

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        out, _ = self.student_decoder(e1, e2, e3, e4, e5, x)
        return out

# Pad H,W to multiple of 32 for the thop wrapper
h32 = H + (32 - H % 32) % 32
w32 = W + (32 - W % 32) % 32
iaml_wrapper = IAMLInferenceWrapper(iaml).to(device)

try:
    gmacs_iaml = measure_gmacs_thop(iaml_wrapper, h32, w32, device)
    print(f"  GMACs @ {h32}×{w32} (pad-to-32):            {gmacs_iaml:.3f} G")
except Exception as e:
    gmacs_iaml = float('nan')
    print(f"  GMACs: FAILED ({e})")

if args.gpu or device.type == 'cpu':
    t_iaml, s_iaml = measure_time_ms(iaml.inference, H, W, device,
                                     args.warmup, args.reps)
    print(f"  Inference time ({args.reps} reps): {t_iaml:.2f} ± {s_iaml:.2f} ms")
else:
    t_iaml = float('nan')
    print("  Inference time: skipped (run with --gpu for GPU timing)")


# ─────────────────────────────────────────────────────────────────────────────
# RetinexFormer  (uses the same create_model path as the test script)
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("RetinexFormer")
print("=" * 60)

try:
    import yaml
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    from basicsr.utils.options import parse
    from basicsr.models import create_model

    opt = parse('Options/RetinexFormer_LOL_v1.yml', is_train=False)
    opt['dist'] = False
    opt['num_gpu'] = 1 if (args.gpu and torch.cuda.is_available()) else 0
    retinex = create_model(opt).net_g.to(device).eval()

    total_ret, trainable_ret = count_params(retinex)
    print(f"  Total params:     {total_ret/1e6:.3f} M")
    print(f"  Trainable params: {trainable_ret/1e6:.3f} M")

    try:
        gmacs_ret = measure_gmacs_thop(retinex, H, W, device)
        print(f"  GMACs @ {H}×{W}:         {gmacs_ret:.3f} G")
    except Exception as e:
        gmacs_ret = float('nan')
        print(f"  GMACs: FAILED ({e})")

    if args.gpu or device.type == 'cpu':
        t_ret, s_ret = measure_time_ms(retinex, H, W, device,
                                       args.warmup, args.reps)
        print(f"  Inference time ({args.reps} reps): {t_ret:.2f} ± {s_ret:.2f} ms")
    else:
        t_ret = float('nan')
        print("  Inference time: skipped")

except Exception as e:
    print(f"  RetinexFormer skipped: {e}")
    total_ret = trainable_ret = gmacs_ret = t_ret = s_ret = float('nan')


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def fmt_m(v): return f"{v/1e6:.2f} M" if not np.isnan(v) else "  N/A  "
def fmt_g(v): return f"{v:.3f} G"     if not np.isnan(v) else "  N/A  "
def fmt_t(v): return f"{v:.1f} ms"    if not np.isnan(v) else "  N/A  "

print()
print("┌" + "─"*72 + "┐")
print(f"│{'Complexity Report @ ' + str(H) + '×' + str(W) + ' input':^72}│")
print("├" + "─"*18 + "┬" + "─"*13 + "┬" + "─"*13 + "┬" + "─"*12 + "┬" + "─"*13 + "┤")
print(f"│{'Model':<18}│{'Train Params':^13}│{'Infer Params':^13}│{'GMACs':^12}│{'Infer Time':^13}│")
print("├" + "─"*18 + "┼" + "─"*13 + "┼" + "─"*13 + "┼" + "─"*12 + "┼" + "─"*13 + "┤")
print(f"│{'RetinexFormer':<18}│{fmt_m(total_ret):^13}│{fmt_m(total_ret):^13}│"
      f"{fmt_g(gmacs_ret):^12}│{fmt_t(t_ret):^13}│")
print(f"│{'IAML (train)':<18}│{fmt_m(trainable_iaml):^13}│{fmt_m(infer_iaml):^13}│"
      f"{fmt_g(gmacs_iaml):^12}│{fmt_t(t_iaml):^13}│")
print("└" + "─"*18 + "┴" + "─"*13 + "┴" + "─"*13 + "┴" + "─"*12 + "┴" + "─"*13 + "┘")

print()
print("Notes:")
print("  • GMACs = Giga Multiply-Accumulate Operations (1 MAC = 2 FLOPs in some papers)")
print("  • IAML 'Train Params' counts encoder+student (teacher is frozen, zero grad)")
print("  • IAML 'Infer Params' = same; teacher_decoder is absent at inference")
print("  • For FLOPs as reported in some papers: FLOPs ≈ 2 × GMACs")
print(f"  • Timing device: {device}")
if device.type == 'cpu':
    print("  • Re-run with --gpu for GPU timing (relevant for practical deployment)")
