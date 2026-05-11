"""
IAML training script.

Usage:
    python train_iaml.py --data_root data/LOLv1
    python train_iaml.py --data_root data/LOLv1 --resume checkpoints/best_model.pth
    python train_iaml.py --smoke_test
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from basicsr.archs.iaml_arch import IAMLNet
from basicsr.data import create_dataset, create_dataloader
from basicsr.losses.iaml_loss import TotalLoss


# ── Helpers ───────────────────────────────────────────────────────────────────

def rgb_to_y(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image (H,W,3) in [0,1] to Y channel of YCbCr (H,W)."""
    return (16.0 / 255.0
            + (65.481 / 255.0) * img_rgb[:, :, 0]
            + (128.553 / 255.0) * img_rgb[:, :, 1]
            + (24.966 / 255.0)  * img_rgb[:, :, 2])


def pad_to_multiple(x: torch.Tensor, multiple: int = 32) -> tuple:
    """Pad (B,C,H,W) tensor to nearest multiple with reflection, return (padded, pad_h, pad_w)."""
    _, _, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode='reflect'), pad_h, pad_w


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model: IAMLNet, val_loader,
             device: torch.device) -> tuple:
    """Compute mean PSNR and SSIM on Y channel at full native resolution."""
    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for batch in val_loader:
            lq = batch['lq'].to(device)
            gt = batch['gt'].to(device)

            _, _, H, W = lq.shape
            lq_padded, pad_h, pad_w = pad_to_multiple(lq)

            enhanced = model.inference(lq_padded)
            enhanced = enhanced[:, :, :H, :W].clamp(0.0, 1.0)

            enhanced_np = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
            clean_np    = gt.squeeze(0).permute(1, 2, 0).cpu().numpy()

            enh_y   = rgb_to_y(enhanced_np)
            clean_y = rgb_to_y(clean_np)

            psnr_list.append(
                peak_signal_noise_ratio(clean_y, enh_y, data_range=1.0))
            ssim_list.append(
                structural_similarity(clean_y, enh_y, data_range=1.0))

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs('checkpoints', exist_ok=True)

    # Datasets and loaders via BasicSR pipeline
    train_dataset_opt = {
        'name': 'LOLv1_train',
        'type': 'Dataset_PairedImage',
        'dataroot_gt': os.path.join(args.data_root, 'Train', 'target'),
        'dataroot_lq': os.path.join(args.data_root, 'Train', 'input'),
        'geometric_augs': True,
        'filename_tmpl': '{}',
        'io_backend': {'type': 'disk'},
        'gt_size': 256,
        'batch_size_per_gpu': 8,
        'num_worker_per_gpu': 4,
        'dataset_enlarge_ratio': 1,
        'prefetch_mode': None,
        'phase': 'train',
        'scale': 1,
    }
    val_dataset_opt = {
        'name': 'LOLv1_val',
        'type': 'Dataset_PairedImage',
        'dataroot_gt': os.path.join(args.data_root, 'Test', 'target'),
        'dataroot_lq': os.path.join(args.data_root, 'Test', 'input'),
        'io_backend': {'type': 'disk'},
        'phase': 'val',
        'scale': 1,
    }

    train_ds = create_dataset(train_dataset_opt)
    val_ds   = create_dataset(val_dataset_opt)

    train_loader = create_dataloader(
        train_ds, train_dataset_opt, num_gpu=1, dist=False, sampler=None, seed=100)
    val_loader   = create_dataloader(
        val_ds, val_dataset_opt, num_gpu=1, dist=False, sampler=None, seed=100)

    # Model, loss, optimiser, scheduler
    model     = IAMLNet().to(device)
    criterion = TotalLoss().to(device)
    optimizer = Adam(
        list(model.encoder.parameters()) +
        list(model.student_decoder.parameters()),
        lr=2e-4, betas=(0.9, 0.999),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-7)

    start_epoch = 1
    best_ssim   = 0.0
    best_psnr   = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_ssim   = ckpt['best_ssim']
        best_psnr   = ckpt['best_psnr']
        print(f"Resumed from epoch {ckpt['epoch']} "
              f"(best SSIM {best_ssim:.4f})")

    iteration = 0

    for epoch in range(start_epoch, 501):
        model.train()

        for batch in train_loader:
            lq = batch['lq'].to(device)
            gt = batch['gt'].to(device)

            optimizer.zero_grad()
            enhanced, pairs = model(lq, gt)
            loss_dict = criterion(enhanced, gt, pairs, lq)
            loss_dict['total'].backward()
            optimizer.step()
            model.ema_update()   # EMA after every optimizer.step()

            iteration += 1
            if iteration % 100 == 0:
                print(f"Epoch {epoch:03d} | Iter {iteration:06d} | "
                      f"Total: {loss_dict['total'].item():.4f} | "
                      f"MSE: {loss_dict['mse'].item():.4f} | "
                      f"SSIM: {loss_dict['ssim'].item():.4f} | "
                      f"IAML: {loss_dict['iaml'].item():.4f}")

        scheduler.step()   # once per epoch, after all batches

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_psnr, val_ssim = validate(model, val_loader, device)

            print(f"Epoch {epoch:03d} | Val PSNR: {val_psnr:.2f} | "
                  f"Val SSIM: {val_ssim:.4f}")

            if val_ssim > best_ssim:
                best_ssim = val_ssim
                best_psnr = val_psnr
                torch.save({
                    'epoch':                epoch,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_ssim':            best_ssim,
                    'best_psnr':            best_psnr,
                }, 'checkpoints/best_model.pth')
                print(f"  ✅ Saved best model — SSIM: {best_ssim:.4f}")

            model.train()

    print(f"\nDone. Best PSNR: {best_psnr:.2f}  Best SSIM: {best_ssim:.4f}")


# ── Smoke test ────────────────────────────────────────────────────────────────

def smoke_test():
    """8-check smoke test — verifies the full pipeline without real data."""
    import sys

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Smoke test on: {device}\n")

    model     = IAMLNet().to(device)
    criterion = TotalLoss().to(device)
    optimizer = Adam(
        list(model.encoder.parameters()) +
        list(model.student_decoder.parameters()),
        lr=2e-4, betas=(0.9, 0.999),
    )

    results = {}

    # ── Check 1: Forward pass with batch_size=8 ───────────────────────────────
    try:
        model.train()
        x_low   = torch.rand(8, 3, 256, 256, device=device)
        x_clean = torch.rand(8, 3, 256, 256, device=device)
        enhanced, pairs = model(x_low, x_clean)
        assert enhanced.shape == torch.Size([8, 3, 256, 256])
        assert len(pairs) == 4
        results[1] = ('PASS', f'enhanced {tuple(enhanced.shape)}, {len(pairs)} pairs')
    except Exception as e:
        results[1] = ('FAIL', str(e))

    # ── Check 2: All loss values positive and not NaN ─────────────────────────
    try:
        loss_dict = criterion(enhanced, x_clean, pairs, x_low)
        for k, v in loss_dict.items():
            assert not torch.isnan(v), f"{k} is NaN"
            assert v.item() > 0,       f"{k} is not positive ({v.item():.6f})"
        vals = {k: f"{v.item():.4f}" for k, v in loss_dict.items()}
        results[2] = ('PASS', str(vals))
    except Exception as e:
        results[2] = ('FAIL', str(e))

    # ── Check 3: total = MSE + SSIM + 0.8×IAML ───────────────────────────────
    try:
        expected = loss_dict['mse'] + loss_dict['ssim'] + 0.8 * loss_dict['iaml']
        assert torch.allclose(loss_dict['total'], expected, atol=1e-5), (
            f"total={loss_dict['total'].item():.6f} expected={expected.item():.6f}")
        results[3] = ('PASS', f"total={loss_dict['total'].item():.6f}")
    except Exception as e:
        results[3] = ('FAIL', str(e))

    # ── Check 4: Student parameters have gradients after backward ─────────────
    try:
        optimizer.zero_grad()
        enhanced2, pairs2 = model(x_low, x_clean)
        loss_dict2 = criterion(enhanced2, x_clean, pairs2, x_low)
        loss_dict2['total'].backward()

        no_grad_params = [
            n for n, p in model.encoder.named_parameters()
            if p.requires_grad and p.grad is None
        ] + [
            n for n, p in model.student_decoder.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert len(no_grad_params) == 0, f"Missing grads: {no_grad_params[:3]}"
        results[4] = ('PASS', 'all student/encoder params have grad')
    except Exception as e:
        results[4] = ('FAIL', str(e))

    # ── Check 5: Teacher decoder parameters have NO gradients ────────────────
    try:
        grad_params = [
            n for n, p in model.teacher_decoder.named_parameters()
            if p.grad is not None
        ]
        assert len(grad_params) == 0, f"Unexpected grads in teacher: {grad_params[:3]}"
        results[5] = ('PASS', 'teacher decoder has no gradients')
    except Exception as e:
        results[5] = ('FAIL', str(e))

    # ── Check 6: EMA changes teacher weights (after optimizer.step()) ─────────
    try:
        teacher_w_before = list(model.teacher_decoder.state_dict().values())[0].clone()
        optimizer.step()       # student weights now differ from teacher
        model.ema_update()
        teacher_w_after = list(model.teacher_decoder.state_dict().values())[0]
        assert not torch.allclose(teacher_w_before, teacher_w_after), \
            "EMA did not change teacher weights"
        delta = (teacher_w_after - teacher_w_before).abs().mean().item()
        results[6] = ('PASS', f"teacher weights changed (mean Δ={delta:.2e})")
    except Exception as e:
        results[6] = ('FAIL', str(e))

    # ── Check 7: Loss changes across 5 iterations ─────────────────────────────
    try:
        totals = []
        for _ in range(5):
            xl = torch.rand(2, 3, 256, 256, device=device)
            xc = torch.rand(2, 3, 256, 256, device=device)
            optimizer.zero_grad()
            enh, pr = model(xl, xc)
            ld = criterion(enh, xc, pr, xl)
            ld['total'].backward()
            optimizer.step()
            model.ema_update()
            totals.append(ld['total'].item())
        assert len(set(round(v, 6) for v in totals)) > 1, \
            f"Loss constant across 5 iters: {totals}"
        results[7] = ('PASS', f"losses: {[f'{v:.4f}' for v in totals]}")
    except Exception as e:
        results[7] = ('FAIL', str(e))

    # ── Check 8: Validation on fake 600×400 image returns 600×400 ─────────────
    try:
        model.eval()

        class FakeValLoader:
            def __iter__(self):
                yield {'lq': torch.rand(1, 3, 400, 600, device=device),
                       'gt': torch.rand(1, 3, 400, 600, device=device)}

        val_psnr, val_ssim = validate(model, FakeValLoader(), device)
        assert isinstance(val_psnr, float) and isinstance(val_ssim, float)
        assert not np.isnan(val_psnr) and not np.isnan(val_ssim)

        # Verify pad/crop logic directly
        fake = torch.rand(1, 3, 400, 600, device=device)
        padded, pad_h, pad_w = pad_to_multiple(fake)
        assert padded.shape[2] % 32 == 0 and padded.shape[3] % 32 == 0
        out = model.inference(padded)
        out = out[:, :, :400, :600]
        assert out.shape == torch.Size([1, 3, 400, 600]), \
            f"Expected (1,3,400,600), got {tuple(out.shape)}"

        results[8] = ('PASS',
                      f"600×400 → padded {tuple(padded.shape[2:])} → "
                      f"cropped back (400,600) | "
                      f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}")
    except Exception as e:
        results[8] = ('FAIL', str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("═" * 65)
    all_pass = True
    for i in range(1, 9):
        status, detail = results[i]
        icon = '✅' if status == 'PASS' else '❌'
        print(f"  {icon} Check {i}: {status}  —  {detail}")
        if status != 'PASS':
            all_pass = False
    print("═" * 65)
    if all_pass:
        print("🎉 All 8 checks passed. Pipeline is ready for training.")
    else:
        print("❌ Some checks failed. See details above.")
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/LOLv1',
                        help='Root of LOL dataset (contains Train/ and Test/)')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run smoke test instead of training')
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
    else:
        train(args)
