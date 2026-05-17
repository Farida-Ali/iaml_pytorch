"""
IAML training script — BasicSR YAML-config convention.

Usage:
    python train_iaml.py -opt options/train/train_IAML_LOLv1.yml
    python train_iaml.py -opt options/train/train_IAML_LOLv1.yml --resume checkpoints/IAML_LOLv1/latest.pth
    python train_iaml.py --smoke_test
"""

import argparse
import os

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from skimage import img_as_ubyte
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from basicsr.archs.iaml_arch import IAMLNet
from basicsr.data import create_dataset, create_dataloader
from basicsr.losses.iaml_loss import TotalLoss


# ── Helpers ───────────────────────────────────────────────────────────────────

def _psnr_rgb(img1: np.ndarray, img2: np.ndarray) -> float:
    """RGB PSNR on [0,1] float images — identical to Enhancement/utils.py::PSNR."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def _ssim_channel(img1: np.ndarray, img2: np.ndarray) -> float:
    """Single-channel SSIM on [0,255] uint8 — identical to Enhancement/utils.py::ssim."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq  = mu1 ** 2
    mu2_sq  = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12   = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
               (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def _ssim_rgb(img1: np.ndarray, img2: np.ndarray) -> float:
    """Per-channel-average SSIM on [0,255] uint8 — identical to Enhancement/utils.py::calculate_ssim."""
    return float(np.mean([_ssim_channel(img1[:, :, i], img2[:, :, i]) for i in range(3)]))


def pad_to_multiple(x: torch.Tensor, multiple: int = 32) -> tuple:
    """Pad (B,C,H,W) to nearest multiple with reflection; return (padded, pad_h, pad_w)."""
    _, _, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode='reflect'), pad_h, pad_w


def load_opt(opt_path: str) -> dict:
    """Load and return the YAML config as a dict."""
    with open(opt_path, 'r') as f:
        return yaml.safe_load(f)


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model: IAMLNet, val_loader, device: torch.device) -> tuple:
    """Compute mean PSNR and SSIM matching test_IAML_from_dataset.py exactly.

    PSNR: utils.PSNR on [0,1] float RGB (all 3 channels).
    SSIM: utils.calculate_ssim on [0,255] uint8 RGB (per-channel average).
    """
    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for batch in val_loader:
            lq = batch['lq'].to(device)
            gt = batch['gt'].to(device)

            _, _, H, W = lq.shape
            lq_padded, pad_h, pad_w = pad_to_multiple(lq)

            enhanced = model.inference(lq_padded)
            enhanced = enhanced[:, :, :H, :W].clamp(0.0, 1.0)

            enhanced_np = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
            clean_np    = gt.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

            psnr_list.append(_psnr_rgb(clean_np, enhanced_np))
            ssim_list.append(_ssim_rgb(img_as_ubyte(clean_np), img_as_ubyte(enhanced_np)))

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    opt = load_opt(args.opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: {args.opt}  ({opt.get('name', 'unknown')})\n")

    # ── Dataset opts from YAML ────────────────────────────────────────────────
    train_opt = opt['datasets']['train']
    val_opt   = opt['datasets']['val']

    # Ensure required BasicSR fields that may be absent from the YAML section
    global_scale = opt.get('scale', 1)
    train_opt.setdefault('phase',      'train')
    train_opt.setdefault('scale',      global_scale)
    train_opt.setdefault('io_backend', {'type': 'disk'})
    val_opt.setdefault('phase',      'val')
    val_opt.setdefault('scale',      global_scale)
    val_opt.setdefault('io_backend', {'type': 'disk'})

    train_ds = create_dataset(train_opt)
    val_ds   = create_dataset(val_opt)

    seed = opt.get('manual_seed', 100)
    train_loader = create_dataloader(
        train_ds, train_opt, num_gpu=1, dist=False, sampler=None, seed=seed)
    val_loader   = create_dataloader(
        val_ds, val_opt, num_gpu=1, dist=False, sampler=None, seed=seed)

    # ── Training hyperparams from YAML ────────────────────────────────────────
    train_cfg  = opt['train']
    total_iter = int(train_cfg['total_iter'])

    optim_cfg  = train_cfg['optim_g']
    lr         = float(optim_cfg['lr'])
    betas      = tuple(optim_cfg.get('betas', [0.9, 0.999]))

    sched_cfg  = train_cfg['scheduler']
    T_max      = int(sched_cfg['T_max'])
    eta_min    = float(sched_cfg['eta_min'])

    val_cfg    = opt.get('val', {})
    val_freq   = int(val_cfg.get('val_freq', total_iter // 10))

    log_cfg    = opt.get('logger', {})
    print_freq = int(log_cfg.get('print_freq', 100))
    save_freq  = int(log_cfg.get('save_checkpoint_freq', val_freq))

    exp_name   = opt.get('name', 'iaml')
    ckpt_dir   = os.path.join('checkpoints', exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"total_iter={total_iter}  lr={lr}  val_freq={val_freq}  "
          f"print_freq={print_freq}  ckpt_dir={ckpt_dir}\n")

    # ── Model, loss, optimiser, scheduler ─────────────────────────────────────
    model     = IAMLNet().to(device)
    criterion = TotalLoss().to(device)
    optimizer = Adam(model.student_parameters(), lr=lr, betas=betas)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    start_iter = 0
    best_ssim  = 0.0
    best_psnr  = 0.0

    resume_path = args.resume or opt.get('path', {}).get('resume_state')
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_iter = ckpt['iteration']
        best_ssim  = ckpt['best_ssim']
        best_psnr  = ckpt['best_psnr']
        print(f"Resumed from iter {start_iter} (best SSIM {best_ssim:.4f})\n")

    # ── Iteration-based training loop ─────────────────────────────────────────
    iteration = start_iter
    epoch     = 0

    while iteration < total_iter:
        epoch += 1
        model.train()

        for batch in train_loader:
            if iteration >= total_iter:
                break

            lq = batch['lq'].to(device)
            gt = batch['gt'].to(device)

            optimizer.zero_grad()
            enhanced, pairs = model(lq, gt)
            loss_dict = criterion(enhanced, gt, pairs, lq)
            loss_dict['total'].backward()
            optimizer.step()
            model.ema_update()

            iteration += 1

            if iteration % print_freq == 0 or iteration == 1:
                print(f"Epoch {epoch:04d} | Iter {iteration:06d}/{total_iter} | "
                      f"Total: {loss_dict['total'].item():.4f} | "
                      f"MSE: {loss_dict['mse'].item():.4f} | "
                      f"SSIM: {loss_dict['ssim'].item():.4f} | "
                      f"IAML: {loss_dict['iaml'].item():.4f}")

            if iteration % val_freq == 0 or iteration == total_iter:
                model.eval()
                val_psnr, val_ssim = validate(model, val_loader, device)
                print(f"\n[Val] Iter {iteration:06d} | "
                      f"PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}")

                if val_ssim > best_ssim:
                    best_ssim = val_ssim
                    best_psnr = val_psnr
                    torch.save({
                        'iteration':            iteration,
                        'model_state_dict':     model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_ssim':            best_ssim,
                        'best_psnr':            best_psnr,
                    }, os.path.join(ckpt_dir, 'best_model.pth'))
                    print(f"  Saved best model — PSNR: {best_psnr:.2f}  SSIM: {best_ssim:.4f}")

                print()
                model.train()

            if iteration % save_freq == 0:
                torch.save({
                    'iteration':            iteration,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_ssim':            best_ssim,
                    'best_psnr':            best_psnr,
                }, os.path.join(ckpt_dir, 'latest.pth'))

        scheduler.step()   # once per epoch

    print(f"\nTraining complete — {total_iter} iterations over {epoch} epochs.")
    print(f"Best PSNR: {best_psnr:.2f} dB   Best SSIM: {best_ssim:.4f}")


# ── Smoke test ────────────────────────────────────────────────────────────────

def smoke_test():
    """8-check smoke test — verifies the full pipeline without real data."""
    import sys

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Smoke test on: {device}\n")

    model     = IAMLNet().to(device)
    criterion = TotalLoss().to(device)
    optimizer = Adam(model.student_parameters(), lr=2e-4, betas=(0.9, 0.999))

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
        optimizer.step()
        model.ema_update()
        teacher_w_after = list(model.teacher_decoder.state_dict().values())[0]
        assert not torch.allclose(teacher_w_before, teacher_w_after), \
            "EMA did not change teacher weights"
        delta = (teacher_w_after - teacher_w_before).abs().mean().item()
        results[6] = ('PASS', f"teacher weights changed (mean delta={delta:.2e})")
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

        fake = torch.rand(1, 3, 400, 600, device=device)
        padded, pad_h, pad_w = pad_to_multiple(fake)
        assert padded.shape[2] % 32 == 0 and padded.shape[3] % 32 == 0
        out = model.inference(padded)
        out = out[:, :, :400, :600]
        assert out.shape == torch.Size([1, 3, 400, 600]), \
            f"Expected (1,3,400,600), got {tuple(out.shape)}"

        results[8] = ('PASS',
                      f"600x400 padded {tuple(padded.shape[2:])} cropped back (400,600) | "
                      f"PSNR={val_psnr:.2f} SSIM={val_ssim:.4f}")
    except Exception as e:
        results[8] = ('FAIL', str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 65)
    all_pass = True
    for i in range(1, 9):
        status, detail = results[i]
        icon = 'OK' if status == 'PASS' else 'FAIL'
        print(f"  [{icon}] Check {i}: {status}  --  {detail}")
        if status != 'PASS':
            all_pass = False
    print("=" * 65)
    if all_pass:
        print("All 8 checks passed. Pipeline is ready for training.")
    else:
        print("Some checks failed. See details above.")
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IAML training script')
    parser.add_argument('-opt', type=str, default=None,
                        help='Path to YAML training config (e.g. options/train/train_IAML_LOLv1.yml)')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from (overrides path.resume_state in YAML)')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run smoke test instead of training (no -opt needed)')
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
    else:
        if args.opt is None:
            parser.error('-opt is required for training (e.g. -opt options/train/train_IAML_LOLv1.yml)')
        train(args)
