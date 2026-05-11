"""
Full pipeline sanity check — 10 checks before real training.
Usage:  python3.11 test_full_pipeline.py
"""

import os
import sys

import numpy as np
import torch
from torch.optim import Adam

from basicsr.archs.iaml_arch import IAMLNet
from basicsr.losses.iaml_loss import TotalLoss
from basicsr.metrics import calculate_psnr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

results = {}   # check_num → ('PASS'/'FAIL', detail_str)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1: Forward pass at training resolution
# ══════════════════════════════════════════════════════════════════════════════

model     = IAMLNet().to(device)
criterion = TotalLoss().to(device)

try:
    model.train()
    lq = torch.rand(8, 3, 256, 256, device=device)
    gt = torch.rand(8, 3, 256, 256, device=device)
    enhanced, pairs = model(lq, gt)

    print("CHECK 1 — shapes:")
    print(f"  enhanced : {tuple(enhanced.shape)}")
    for i, (u, t) in enumerate(pairs):
        print(f"  pairs[{i}][0] (u{i+1}): {tuple(u.shape)}")
        print(f"  pairs[{i}][1] (t{i+1}): {tuple(t.shape)}")

    expected_pairs = [
        (8, 256, 16, 16),    # u1 @ H/16 (first deconv step from bottleneck)
        (8, 128, 32, 32),    # u2 @ H/8
        (8,  64, 64, 64),    # u3 @ H/4
        (8,  32, 128, 128),  # u4 @ H/2
    ]
    shape_ok = enhanced.shape == torch.Size([8, 3, 256, 256])
    range_ok  = enhanced.min() >= 0.0 and enhanced.max() <= 1.0
    npairs_ok = len(pairs) == 4
    pair_shapes_ok = all(
        tuple(pairs[i][0].shape) == expected_pairs[i] for i in range(4)
    )

    fails = []
    if not shape_ok:    fails.append(f"enhanced.shape={tuple(enhanced.shape)} expected (8,3,256,256)")
    if not range_ok:    fails.append(f"range [{enhanced.min():.3f},{enhanced.max():.3f}] not in [0,1]")
    if not npairs_ok:   fails.append(f"len(pairs)={len(pairs)} expected 4")
    if not pair_shapes_ok:
        for i in range(4):
            got = tuple(pairs[i][0].shape)
            exp = expected_pairs[i]
            if got != exp:
                fails.append(f"pairs[{i}][0].shape={got} expected {exp}")

    if fails:
        results[1] = ('FAIL', '; '.join(fails))
    else:
        results[1] = ('PASS', f"enhanced {tuple(enhanced.shape)}, {len(pairs)} pairs, range OK")
except Exception as e:
    results[1] = ('FAIL', str(e))


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2: Loss values valid and correctly structured
# ══════════════════════════════════════════════════════════════════════════════

try:
    loss_dict = criterion(enhanced, gt, pairs, lq)

    mse_v   = loss_dict['mse'].item()
    ssim_v  = loss_dict['ssim'].item()
    iaml_v  = loss_dict['iaml'].item()
    total_v = loss_dict['total'].item()
    expected_total = mse_v + ssim_v + 0.8 * iaml_v

    print(f"\nCHECK 2 — loss values:")
    print(f"  mse={mse_v:.6f}  ssim={ssim_v:.6f}  iaml={iaml_v:.6f}  total={total_v:.6f}")
    print(f"  mse+ssim+0.8*iaml = {expected_total:.6f}  (delta={abs(total_v-expected_total):.2e})")

    keys_ok   = set(loss_dict.keys()) == {'total', 'mse', 'ssim', 'iaml'}
    nan_ok    = not any(torch.isnan(v) for v in loss_dict.values())
    inf_ok    = not any(torch.isinf(v) for v in loss_dict.values())
    mse_rng   = 0.0 < mse_v < 1.0
    ssim_rng  = 0.0 < ssim_v < 1.0
    iaml_rng  = 0.0 < iaml_v < 10.0
    total_rng = 0.0 < total_v < 10.0
    total_eq  = abs(total_v - expected_total) < 1e-4

    fails = []
    if not keys_ok:   fails.append(f"keys={set(loss_dict.keys())}")
    if not nan_ok:    fails.append("NaN in loss")
    if not inf_ok:    fails.append("Inf in loss")
    if not mse_rng:   fails.append(f"mse={mse_v:.6f} not in (0,1)")
    if not ssim_rng:  fails.append(f"ssim={ssim_v:.6f} not in (0,1)")
    if not iaml_rng:  fails.append(f"iaml={iaml_v:.6f} not in (0,10)")
    if not total_rng: fails.append(f"total={total_v:.6f} not in (0,10)")
    if not total_eq:  fails.append(f"total≠mse+ssim+0.8*iaml (delta={abs(total_v-expected_total):.2e})")

    results[2] = ('FAIL', '; '.join(fails)) if fails else (
        'PASS', f"mse={mse_v:.4f} ssim={ssim_v:.4f} iaml={iaml_v:.4f} total={total_v:.4f}")
except Exception as e:
    results[2] = ('FAIL', str(e))


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3: Student receives gradients
# ══════════════════════════════════════════════════════════════════════════════

try:
    optimizer = Adam(model.student_parameters(), lr=2e-4)
    optimizer.zero_grad()

    lq2 = torch.rand(8, 3, 256, 256, device=device)
    gt2 = torch.rand(8, 3, 256, 256, device=device)
    enh2, pairs2 = model(lq2, gt2)
    loss2 = criterion(enh2, gt2, pairs2, lq2)
    loss2['total'].backward()

    enc_grads  = [p.grad for p in model.encoder.parameters()
                  if p.requires_grad and p.grad is not None]
    dec_grads  = [p.grad for p in model.student_decoder.parameters()
                  if p.requires_grad and p.grad is not None]

    enc_norm  = sum(g.norm().item() for g in enc_grads)
    dec_norm  = sum(g.norm().item() for g in dec_grads)

    print(f"\nCHECK 3 — gradient norms:")
    print(f"  encoder grad norm:          {enc_norm:.4f}")
    print(f"  student_decoder grad norm:  {dec_norm:.4f}")

    enc_has  = len(enc_grads) > 0 and enc_norm > 0
    dec_has  = len(dec_grads) > 0 and dec_norm > 0

    fails = []
    if not enc_has: fails.append(f"encoder has no/zero gradients (norm={enc_norm:.4f})")
    if not dec_has: fails.append(f"student_decoder has no/zero gradients (norm={dec_norm:.4f})")

    results[3] = ('FAIL', '; '.join(fails)) if fails else (
        'PASS', f"enc_norm={enc_norm:.4f} dec_norm={dec_norm:.4f}")
except Exception as e:
    results[3] = ('FAIL', str(e))


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4: Teacher receives NO gradients
# ══════════════════════════════════════════════════════════════════════════════

try:
    teacher_params = list(model.teacher_decoder.parameters())
    n_total = len(teacher_params)
    no_grad_count   = sum(1 for p in teacher_params if p.grad is None)
    req_false_count = sum(1 for p in teacher_params if not p.requires_grad)

    print(f"\nCHECK 4 — teacher parameters checked: {n_total}")
    print(f"  .grad is None:        {no_grad_count}/{n_total}")
    print(f"  requires_grad=False:  {req_false_count}/{n_total}")

    fails = []
    if no_grad_count != n_total:
        fails.append(f"{n_total-no_grad_count} teacher params have grad (expected 0)")
    if req_false_count != n_total:
        fails.append(f"{n_total-req_false_count} teacher params have requires_grad=True")

    results[4] = ('FAIL', '; '.join(fails)) if fails else (
        'PASS', f"all {n_total} teacher params frozen, no gradients")
except Exception as e:
    results[4] = ('FAIL', str(e))


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5: EMA update works correctly
# ══════════════════════════════════════════════════════════════════════════════

try:
    opt5 = Adam(model.student_parameters(), lr=2e-4)

    for _ in range(20):
        lq_b = torch.rand(2, 3, 256, 256, device=device)
        gt_b = torch.rand(2, 3, 256, 256, device=device)
        enh_b, pairs_b = model(lq_b, gt_b)
        loss_b = criterion(enh_b, gt_b, pairs_b, lq_b)['total']
        loss_b.backward()
        opt5.step()
        opt5.zero_grad()
        # Do NOT call ema_update() yet

    first_layer_name = list(model.teacher_decoder.state_dict().keys())[0]
    teacher_before = model.teacher_decoder.state_dict()[first_layer_name].clone()
    student_val    = model.student_decoder.state_dict()[first_layer_name].clone()

    model.ema_update()
    teacher_after = model.teacher_decoder.state_dict()[first_layer_name]

    change  = (teacher_after - teacher_before).abs().mean().item()
    stu_diff = (student_val - teacher_before).abs().mean().item()

    print(f"\nCHECK 5 — EMA:")
    print(f"  teacher change after ema_update():  {change:.6e}")
    print(f"  student-teacher difference before:  {stu_diff:.6e}")
    print(f"  layer checked: {first_layer_name}")

    fails = []
    if change <= 0:
        fails.append(f"teacher did not change (change={change:.2e})")
    if change >= stu_diff and stu_diff > 0:
        fails.append(f"teacher moved >= student diff ({change:.2e} >= {stu_diff:.2e})")

    results[5] = ('FAIL', '; '.join(fails)) if fails else (
        'PASS', f"change={change:.2e} < stu_diff={stu_diff:.2e}")
except Exception as e:
    results[5] = ('FAIL', str(e))


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6: Model is trainable (loss changes over iterations)
# ══════════════════════════════════════════════════════════════════════════════

try:
    opt6 = Adam(model.student_parameters(), lr=2e-4)
    losses = []
    grad_norms = []
    nan_hit = False

    for step in range(10):
        xl = torch.rand(2, 3, 256, 256, device=device)
        xc = torch.rand(2, 3, 256, 256, device=device)
        opt6.zero_grad()
        enh6, pairs6 = model(xl, xc)
        ld6 = criterion(enh6, xc, pairs6, xl)
        ld6['total'].backward()

        gn = sum(p.grad.norm().item() for p in model.student_parameters()
                 if p.grad is not None)
        grad_norms.append(gn)

        if torch.isnan(ld6['total']):
            nan_hit = True
        losses.append(ld6['total'].item())
        opt6.step()
        model.ema_update()

    print(f"\nCHECK 6 — loss over 10 iterations:")
    print(f"  iter 1 : {losses[0]:.6f}  (grad_norm={grad_norms[0]:.4f})")
    print(f"  iter 5 : {losses[4]:.6f}  (grad_norm={grad_norms[4]:.4f})")
    print(f"  iter 10: {losses[9]:.6f}  (grad_norm={grad_norms[9]:.4f})")

    loss_changed   = abs(losses[-1] - losses[0]) > 1e-6
    no_nan         = not nan_hit
    grads_nonzero  = all(g > 0 for g in grad_norms)

    fails = []
    if not loss_changed:   fails.append(f"loss constant iter1={losses[0]:.6f} iter10={losses[-1]:.6f}")
    if not no_nan:         fails.append("NaN loss occurred")
    if not grads_nonzero:  fails.append("zero gradient norm at some iteration")

    results[6] = ('FAIL', '; '.join(fails)) if fails else (
        'PASS', f"iter1={losses[0]:.4f} iter5={losses[4]:.4f} iter10={losses[9]:.4f}")
except Exception as e:
    results[6] = ('FAIL', str(e))


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 7: Inference at LOL-v1 native resolution
# ══════════════════════════════════════════════════════════════════════════════

try:
    x_test = torch.rand(1, 3, 600, 400, device=device)
    model.eval()
    with torch.no_grad():
        out = model.inference(x_test)

    print(f"\nCHECK 7 — inference (600×400):")
    print(f"  input  : {tuple(x_test.shape)}")
    print(f"  output : {tuple(out.shape)}")
    print(f"  range  : [{out.min():.4f}, {out.max():.4f}]")
    print(f"  has NaN: {torch.isnan(out).any().item()}")

    shape_ok = out.shape == torch.Size([1, 3, 600, 400])
    range_ok  = out.min() >= 0.0 and out.max() <= 1.0
    nan_ok    = not torch.isnan(out).any()

    fails = []
    if not shape_ok: fails.append(f"shape={tuple(out.shape)} expected (1,3,600,400)")
    if not range_ok: fails.append(f"range [{out.min():.4f},{out.max():.4f}] not in [0,1]")
    if not nan_ok:   fails.append("NaN in output")

    results[7] = ('FAIL', '; '.join(fails)) if fails else (
        'PASS', f"shape {tuple(out.shape)}, range [{out.min():.4f},{out.max():.4f}]")
except Exception as e:
    results[7] = ('FAIL', str(e))


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 8: Padding and cropping logic
# ══════════════════════════════════════════════════════════════════════════════

def _padded_size(H, W, m=32):
    return ((H + m - 1) // m) * m, ((W + m - 1) // m) * m

try:
    test_cases = [
        (1, 3, 600, 400),
        (1, 3, 601, 401),
        (1, 3, 720, 1280),
    ]
    expected_padded = [
        (608, 416),    # 600→608, 400→416
        (608, 416),    # 601→608, 401→416
        (736, 1280),   # 720→736 (720 is NOT ÷32), 1280→1280 (is ÷32)
    ]

    print("\nCHECK 8 — pad/crop logic:")
    all_ok = True
    details = []

    model.eval()
    for (B, C, H, W), (pH, pW) in zip(test_cases, expected_padded):
        x_in = torch.rand(B, C, H, W, device=device)
        with torch.no_grad():
            out8 = model.inference(x_in)
        calc_pH, calc_pW = _padded_size(H, W)
        shape_matches = out8.shape == torch.Size([B, C, H, W])
        padded_ok = (calc_pH == pH and calc_pW == pW)
        print(f"  ({H}×{W}) → padded ({calc_pH}×{calc_pW}) → output {tuple(out8.shape[2:])}")
        if not shape_matches:
            all_ok = False
            details.append(f"({H}×{W}) output {tuple(out8.shape)} ≠ expected ({B},{C},{H},{W})")
        if not padded_ok:
            all_ok = False
            details.append(f"({H}×{W}) padded to ({calc_pH}×{calc_pW}) expected ({pH}×{pW})")

    results[8] = ('FAIL', '; '.join(details)) if not all_ok else (
        'PASS', "all 3 resolutions: output shape == input shape")
except Exception as e:
    results[8] = ('FAIL', str(e))


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 9: PSNR function from basicsr
# ══════════════════════════════════════════════════════════════════════════════

try:
    # basicsr calculate_psnr expects uint8-scale images [0, 255]
    # Case A: identical images
    img_a = (np.random.rand(400, 600, 3) * 255).astype(np.float32)
    psnr_identical = calculate_psnr(img_a, img_a, crop_border=0, test_y_channel=True)

    # Case B: all-zeros vs all-ones (in [0,255] scale)
    img_zero = np.zeros((400, 600, 3), dtype=np.float32)
    img_one  = np.full((400, 600, 3), 255.0, dtype=np.float32)
    psnr_different = calculate_psnr(img_zero, img_one, crop_border=0, test_y_channel=True)

    # Case C: slightly different (σ≈2.9 on 0–255 scale → PSNR ≈ 39 dB)
    noise = (np.random.rand(400, 600, 3) * 10).astype(np.float32)
    img_b = np.clip(img_a + noise, 0, 255)
    psnr_realistic = calculate_psnr(img_a, img_b, crop_border=0, test_y_channel=True)

    print(f"\nCHECK 9 — PSNR:")
    print(f"  Case A (identical):   {psnr_identical:.2f} dB")
    print(f"  Case B (0 vs 1):      {psnr_different:.2f} dB")
    print(f"  Case C (slight diff): {psnr_realistic:.2f} dB")

    fails = []
    if psnr_identical <= 60:
        fails.append(f"identical PSNR={psnr_identical:.2f} not >60 dB")
    if not (20 < psnr_realistic < 60):
        fails.append(f"realistic PSNR={psnr_realistic:.2f} not in (20,60) dB")

    results[9] = ('FAIL', '; '.join(fails)) if fails else (
        'PASS', f"A={psnr_identical:.1f} B={psnr_different:.1f} C={psnr_realistic:.1f} dB")
except Exception as e:
    results[9] = ('FAIL', str(e))


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 10: All required files exist with non-zero size
# ══════════════════════════════════════════════════════════════════════════════

required_files = [
    'basicsr/archs/iaml_arch.py',
    'basicsr/losses/iaml_loss.py',
    'options/train/train_IAML_LOLv1.yml',
    'options/train/train_IAML_LOLv2_real.yml',
    'options/train/train_IAML_LOLv2_synthetic.yml',
    'options/test/test_IAML_LOLv1.yml',
    'options/test/test_IAML_LOLv2_real.yml',
    'options/test/test_IAML_LOLv2_synthetic.yml',
    'train_iaml.py',
    'compute_extra_metrics.py',
]

try:
    print("\nCHECK 10 — required files:")
    missing = []
    for fpath in required_files:
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  {'OK':>4}  {size:>7} bytes  {fpath}")
            if size == 0:
                missing.append(f"{fpath} (0 bytes)")
        else:
            print(f"  {'MISS':>4}            {fpath}")
            missing.append(f"{fpath} (missing)")

    results[10] = ('FAIL', '; '.join(missing)) if missing else (
        'PASS', f"all {len(required_files)} files present and non-empty")
except Exception as e:
    results[10] = ('FAIL', str(e))


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

labels = {
    1: 'Forward pass      ',
    2: 'Loss values       ',
    3: 'Student gradients ',
    4: 'Teacher frozen    ',
    5: 'EMA update        ',
    6: 'Model trainable   ',
    7: 'Native resolution ',
    8: 'Pad/crop logic    ',
    9: 'PSNR function     ',
    10:'All files exist   ',
}

all_pass = all(v[0] == 'PASS' for v in results.values())

print()
print('  ╔══════════════════════════════════════╗')
print('  ║       SANITY CHECK SUMMARY           ║')
print('  ╠══════════════════════════════════════╣')
for i in range(1, 11):
    status, detail = results.get(i, ('SKIP', 'not reached'))
    mark = 'PASS' if status == 'PASS' else 'FAIL'
    print(f'  ║ Check {i:>2}  {labels[i]}  {mark} ║')
print('  ╠══════════════════════════════════════╣')
if all_pass:
    print('  ║  RESULT: READY FOR TRAINING     ✅  ║')
else:
    print('  ║  RESULT: SOME CHECKS FAILED     ❌  ║')
print('  ╚══════════════════════════════════════╝')

if not all_pass:
    print('\nFailed checks:')
    for i in range(1, 11):
        status, detail = results.get(i, ('SKIP', 'not reached'))
        if status != 'PASS':
            print(f'  Check {i}: {status} — {detail}')
    sys.exit(1)
