"""
Compute FLOPs, parameters, and inference time for IAML and RetinexFormer.

Uses the SAME my_summary() method as Retinexformer's official repo so that
numbers are directly comparable for paper tables.

Reference: Enhancement/utils.py → my_summary(model, H, W, C, N)
  - FLOPs counted by fvcore.nn.FlopCountAnalysis
  - Divided by 1024**3 → "GMac" (their naming; actually GFlops)

Requires GPU + fvcore:
    pip install fvcore
    python3 compute_complexity.py --gpu

CPU-only (params only, no FLOPs):
    python3 compute_complexity.py
"""
import argparse
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',    action='store_true', help='Use CUDA (required for FLOPs + timing)')
parser.add_argument('--height', type=int, default=256, help='Input height (default: 256, same as Retinexformer paper)')
parser.add_argument('--width',  type=int, default=256, help='Input width  (default: 256)')
parser.add_argument('--warmup', type=int, default=20,  help='Warm-up iterations for timing')
parser.add_argument('--reps',   type=int, default=100, help='Timing repetitions')
args = parser.parse_args()

device = torch.device('cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu')
if args.gpu and device.type == 'cpu':
    print("WARNING: --gpu requested but no CUDA available. Falling back to CPU (no FLOPs).")
H, W = args.height, args.width
print(f"\nDevice: {device}   Input: {H}×{W}\n")

# ── Try to import fvcore (same as my_summary) ────────────────────────────────
try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("WARNING: fvcore not installed → FLOPs will be skipped.")
    print("         Install with: pip install fvcore")
    print("         FLOPs are required for fair comparison with Retinexformer.\n")

# ─────────────────────────────────────────────────────────────────────────────
# my_summary — identical logic to Enhancement/utils.py
# (GPU required; FlopCountAnalysis does not support CPU for all ops)
# ─────────────────────────────────────────────────────────────────────────────

def my_summary(model, H=256, W=256, C=3, N=1):
    """Exact replica of Retinexformer's my_summary() for fair comparison."""
    if not HAS_FVCORE:
        return float('nan')
    if device.type == 'cpu':
        print("  [FLOPs] Skipped — fvcore requires CUDA. Run with --gpu.")
        return float('nan')
    inputs = torch.randn((N, C, H, W)).to(device)
    flops = FlopCountAnalysis(model, inputs)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    gmac = flops.total() / (1024 ** 3)   # matches their exact divisor
    return gmac


def count_params(model):
    total     = sum(p.nelement() for p in model.parameters())
    trainable = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_time_ms(fn, h, w, warmup, reps):
    if device.type == 'cpu':
        return float('nan'), float('nan')
    x = torch.randn(1, 3, h, w).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(x); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    return float(np.mean(times)), float(np.std(times))


# ─────────────────────────────────────────────────────────────────────────────
# IAML
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("IAML (ours)")
print("=" * 60)

from basicsr.archs.iaml_arch import IAMLNet

iaml = IAMLNet().to(device).eval()
total_iaml, trainable_iaml = count_params(iaml)
infer_iaml = sum(p.numel() for n, p in iaml.named_parameters()
                 if not n.startswith('teacher_decoder'))
teacher_iaml = total_iaml - infer_iaml

print(f"  Total params  (enc + student + teacher): {total_iaml:>10,}  ({total_iaml/1e6:.3f} M)")
print(f"  Trainable     (enc + student, teacher frozen): {trainable_iaml:>10,}  ({trainable_iaml/1e6:.3f} M)")
print(f"  Inference     (enc + student, teacher absent): {infer_iaml:>10,}  ({infer_iaml/1e6:.3f} M)")
print(f"  Teacher only  (frozen, not used at inference): {teacher_iaml:>10,}  ({teacher_iaml/1e6:.3f} M)")

# Wrap inference path (no teacher) for FlopCountAnalysis
class IAMLInferenceWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.encoder = m.encoder
        self.student_decoder = m.student_decoder
    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        out, _ = self.student_decoder(e1, e2, e3, e4, e5, x)
        return out

# Pad H,W to multiple of 32 for the inference wrapper
h32 = H + (32 - H % 32) % 32
w32 = W + (32 - W % 32) % 32
iaml_infer = IAMLInferenceWrapper(iaml).to(device)
gmac_iaml = my_summary(iaml_infer, H=h32, W=w32)
if not np.isnan(gmac_iaml):
    print(f"  GFlops (fvcore, {h32}×{w32}): {gmac_iaml:.3f}  [same tool as Retinexformer]")

t_iaml, s_iaml = measure_time_ms(iaml.inference, H, W, args.warmup, args.reps)
if not np.isnan(t_iaml):
    print(f"  Inference time ({args.reps} reps, {H}×{W}): {t_iaml:.2f} ± {s_iaml:.2f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# RetinexFormer — using my_summary exactly as their repo does
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("RetinexFormer")
print("=" * 60)

try:
    from basicsr.utils.options import parse
    from basicsr.models import create_model

    opt = parse('Options/RetinexFormer_LOL_v1.yml', is_train=False)
    opt['dist'] = False
    opt['num_gpu'] = 1 if device.type == 'cuda' else 0
    retinex = create_model(opt).net_g.to(device).eval()

    total_ret, _ = count_params(retinex)
    print(f"  Params: {total_ret:>10,}  ({total_ret/1e6:.3f} M)")

    # This is exactly how Retinexformer's README says to call it:
    #   from utils import my_summary
    #   my_summary(RetinexFormer(), 256, 256, 3, 1)
    gmac_ret = my_summary(retinex, H=H, W=W)
    if not np.isnan(gmac_ret):
        print(f"  GFlops (fvcore, {H}×{W}): {gmac_ret:.3f}  [same tool as Retinexformer]")

    t_ret, s_ret = measure_time_ms(retinex, H, W, args.warmup, args.reps)
    if not np.isnan(t_ret):
        print(f"  Inference time ({args.reps} reps, {H}×{W}): {t_ret:.2f} ± {s_ret:.2f} ms")

except Exception as e:
    total_ret = gmac_ret = t_ret = s_ret = float('nan')
    print(f"  Skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def fm(v): return f"{v/1e6:.2f} M"   if not np.isnan(v) else "  N/A  "
def fg(v): return f"{v:.2f} G"       if not np.isnan(v) else "  N/A  "
def ft(v): return f"{v:.1f} ms"      if not np.isnan(v) else "  N/A  "

print()
print("┌" + "─"*74 + "┐")
print(f"│{'Complexity @ ' + str(H) + '×' + str(W) + '  (fvcore = same tool as Retinexformer)':^74}│")
print("├" + "─"*18 + "┬" + "─"*12 + "┬" + "─"*12 + "┬" + "─"*14 + "┬" + "─"*14 + "┤")
print(f"│{'Model':<18}│{'Infer Params':^12}│{'Train Params':^12}│{'GFlops(fvcore)':^14}│{'Infer Time':^14}│")
print("├" + "─"*18 + "┼" + "─"*12 + "┼" + "─"*12 + "┼" + "─"*14 + "┼" + "─"*14 + "┤")
print(f"│{'RetinexFormer':<18}│{fm(total_ret):^12}│{fm(total_ret):^12}│{fg(gmac_ret):^14}│{ft(t_ret):^14}│")
print(f"│{'IAML (ours)':<18}│{fm(infer_iaml):^12}│{fm(trainable_iaml):^12}│{fg(gmac_iaml):^14}│{ft(t_iaml):^14}│")
print("└" + "─"*18 + "┴" + "─"*12 + "┴" + "─"*12 + "┴" + "─"*14 + "┴" + "─"*14 + "┘")

print()
print("Notes:")
print("  • GFlops measured by fvcore.FlopCountAnalysis (same as Retinexformer repo)")
print("    Retinexformer calls it 'GMac' in utils.py but the tool counts FLOPs.")
print("  • IAML FLOPs measured on inference path only (encoder + student decoder).")
print("    Input padded to multiple of 32 for IAML's encoder requirement.")
print("  • 'Infer Params': weights present and used during inference.")
print("  • 'Train Params': weights with requires_grad=True (teacher is frozen).")
print("  • For the paper: report 'Infer Params' and GFlops in the comparison table.")
if device.type == 'cpu':
    print("\n  *** Re-run with --gpu on your training machine for actual numbers. ***")
    print("  *** fvcore requires CUDA; CPU results show params only.           ***")

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
