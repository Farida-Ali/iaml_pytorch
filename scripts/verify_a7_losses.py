"""
scripts/verify_a7_losses.py
───────────────────────────
Step-2 verification for FD²RT A7. Loads the trained A4 checkpoint into the
A4 architecture, attaches the LL-capture hook exactly as FD2RT_A7_Model does,
and runs ONE forward + backward on a dummy batch.

Confirms:
  • L_freq is computed and non-zero
  • L_tv_illum is computed and non-zero
  • captured LL has shape [B, 3, H/2, W/2]
  • no errors, no NaN, gradients flow

Run from repo root:
  python scripts/verify_a7_losses.py
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4
from basicsr.models.losses import FrequencyAwareLoss, IlluminationTVLoss

CKPT = ('/root/.claude/uploads/65d5d802-aad9-4807-87d2-a69bf439e318/'
        'e8684ae8-best_psnr_23.71_126000.pth')


def main():
    torch.manual_seed(0)
    device = 'cpu'

    kwargs = dict(in_channels=3, out_channels=3, n_feat=40,
                  stage=1, num_blocks=[1, 2, 2])
    net = FD2RT_A4(**kwargs).to(device)

    # Load real A4 weights (freq_blocks/gates are new → strict=False).
    ckpt = torch.load(CKPT, map_location=device)
    state = ckpt.get('params', ckpt)
    missing, unexpected = net.load_state_dict(state, strict=False)
    new_only = [k for k in missing if ('freq_blocks' in k or 'gates' in k)]
    other_missing = [k for k in missing if k not in new_only]
    print(f'[ckpt] loaded A4 weights; new-only missing={len(new_only)}, '
          f'other_missing={len(other_missing)}, unexpected={len(unexpected)}')
    assert not other_missing and not unexpected, 'unexpected key mismatch'

    # Attach LL-capture hook on the W-IE dwt (same matching rule as the model).
    captured = {}
    target_name = None
    for name, module in net.named_modules():
        if name.endswith('estimator.dwt'):
            target_name = name
            module.register_forward_hook(
                lambda m, i, o: captured.__setitem__('LL', o[0]))
            break
    print(f'[hook] attached to "{target_name}"')
    assert target_name is not None

    cri_freq = FrequencyAwareLoss(loss_weight=0.1, w_low=1.0, w_high=2.0)
    cri_tv = IlluminationTVLoss(loss_weight=0.01)

    # Dummy batch (B=2, 128x128 like training patches).
    lq = torch.rand(2, 3, 128, 128, device=device, requires_grad=False)
    gt = torch.rand(2, 3, 128, 128, device=device)

    net.train()
    out = net(lq)
    LL = captured.get('LL', None)

    print(f'[forward] output shape   : {tuple(out.shape)}')
    print(f'[forward] captured LL     : '
          f'{tuple(LL.shape) if LL is not None else None}')

    assert LL is not None, 'LL not captured'
    B, C, h, w = LL.shape
    assert (B, C, h, w) == (2, 3, 64, 64), f'unexpected LL shape {LL.shape}'

    l_pix = torch.nn.functional.l1_loss(out, gt)
    l_freq = cri_freq(out, gt)
    l_tv = cri_tv(LL)
    l_total = l_pix + l_freq + l_tv

    print()
    print(f'  l_pix              : {l_pix.item():.6f}')
    print(f'  l_freq  (weighted) : {l_freq.item():.6f}')
    print(f'  l_tv    (weighted) : {l_tv.item():.6f}')
    print(f'  l_total            : {l_total.item():.6f}')
    print(f'  l_freq / l_pix     : {100*l_freq.item()/l_pix.item():.2f}%')
    print(f'  l_tv   / l_pix     : {100*l_tv.item()/l_pix.item():.2f}%')

    # Backward — confirm gradients flow and nothing is NaN.
    l_total.backward()
    nan_grad = any((p.grad is not None and not torch.isfinite(p.grad).all())
                   for p in net.parameters())
    has_grad = any((p.grad is not None and p.grad.abs().sum() > 0)
                   for p in net.parameters())

    print()
    ok = True
    def check(label, cond):
        nonlocal ok
        ok &= cond
        print(f'  {"PASS" if cond else "FAIL"}  {label}')
    check('L_freq non-zero',            l_freq.item() > 0)
    check('L_tv   non-zero',            l_tv.item() > 0)
    check('LL shape == [2,3,64,64]',    (B, C, h, w) == (2, 3, 64, 64))
    check('no NaN/Inf in losses',
          all(torch.isfinite(t).all() for t in (l_pix, l_freq, l_tv)))
    check('no NaN/Inf in gradients',    not nan_grad)
    check('gradients flow',             has_grad)

    print()
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
