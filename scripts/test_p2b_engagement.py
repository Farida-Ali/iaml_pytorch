"""
Phase-2b: does the ICNF mechanism actually ENGAGE during training?

Motivation. At initialisation the gate is closed (~0.02) and the frequency
branch is small, so gradient reaching the gate and the sensor parameters is
~1e-8 while the spatial branch sees ~1e+1. Adam is scale-invariant, so in
principle that still trains -- but "in principle" is how A4 ended up with a
scalar gate that may never have moved off its initialisation. This test settles
it empirically instead of assuming.

We train a small FD2RT_ICNF on synthetic Retinex scenes whose textured half is
genuinely recoverable, and watch:
  * the mean gate value (does it open?)
  * gradient magnitude reaching the gate and the sensor params (does it grow?)
  * spatial informativeness of the gate (does it separate texture from flat?)
  * the loss (does the thing actually learn?)

  python scripts/test_p2b_engagement.py [--iters 300]
"""
import sys, os, math, argparse
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

from basicsr.models.archs.fd2rt_icnf_arch import FD2RT_ICNF, ICNF_Block_Dual

RESULTS = []


def check(label, cond, detail=''):
    RESULTS.append((label, bool(cond), detail))
    print(f'  {"PASS" if cond else "FAIL"}  {label}' + (f'  {detail}' if detail else ''))
    return bool(cond)


def texture(H, W, beta=0.4):
    fy = torch.fft.fftfreq(H).view(-1, 1); fx = torch.fft.fftfreq(W).view(1, -1)
    f = torch.sqrt(fy**2 + fx**2); f[0, 0] = 1e-6
    amp = 1.0 / (f ** beta); amp[0, 0] = 0
    img = torch.fft.ifft2(amp * torch.exp(1j * torch.rand(H, W) * 2 * math.pi)).real
    return (img - img.mean()) / (img.std() + 1e-8)


def batch(bs, S=64, a=0.02, b=1e-5):
    """Retinex scenes: flat left half, textured right half, illumination ramp.
    Returns (noisy low-light input, clean well-lit target)."""
    lq, gt = [], []
    for _ in range(bs):
        R = torch.full((S, S), 0.5)
        R[:, S//2:] = (0.5 + 0.30 * texture(S, S//2)).clamp(0.05, 0.95)
        yy = torch.linspace(0, 1, S).view(-1, 1).expand(S, S)
        L = (0.3 + 1.4 * yy)
        L = L * (0.15 / L.mean())
        clean = (L * R).clamp(1e-4, 1)
        var = a * clean + b
        obs = (clean + var.sqrt() * torch.randn(S, S)).clamp(1e-4, 1)
        lq.append(obs.expand(3, S, S).clone())
        gt.append(R.expand(3, S, S).clone())     # target = reflectance (well-lit)
    return torch.stack(lq), torch.stack(gt)


def gate_stats(net, x):
    """Mean gate, and AUC of the gate against the known texture mask."""
    ev = net.body[0].icnf(x)
    blk = next(m for m in net.modules() if isinstance(m, ICNF_Block_Dual))
    with torch.no_grad():
        g = blk.gates[0](ev)
    S = x.shape[-1]
    label = torch.zeros_like(g); label[:, :, :, S//2:] = 1.0
    s = g.flatten(); y = label.flatten().bool()
    order = torch.argsort(s)
    ranks = torch.empty_like(s); ranks[order] = torch.arange(len(s), dtype=s.dtype)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    auc = ((ranks[y].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)).item()
    return g.mean().item(), auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--gate_bias', type=float, default=6.0)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(0)
    net = FD2RT_ICNF(in_channels=3, out_channels=3, n_feat=16,
                     stage=1, num_blocks=[1, 1, 1], gate_bias=args.gate_bias)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    gate_params = [p for n, p in net.named_parameters() if '.gates.' in n]
    icnf_params = [p for n, p in net.named_parameters() if 'icnf.log_' in n]

    probe_x, _ = batch(2)
    print('=' * 76)
    print(f'P2b — mechanism engagement over {args.iters} Adam steps')
    print('=' * 76)
    print(f'{"iter":>6} {"loss":>9} {"gate mean":>11} {"gate AUC":>10} '
          f'{"|g|gates":>11} {"|g|sensor":>11} {"a":>9}')
    print('-' * 76)

    hist = []
    for it in range(args.iters + 1):
        lq, gt = batch(4)
        opt.zero_grad()
        loss = F.l1_loss(net(lq), gt)
        loss.backward()
        gg = sum(p.grad.abs().sum().item() for p in gate_params if p.grad is not None)
        gi = sum(p.grad.abs().sum().item() for p in icnf_params if p.grad is not None)
        opt.step()

        if it % max(1, args.iters // 10) == 0 or it == args.iters:
            gm, gauc = gate_stats(net, probe_x)
            a_val = net.body[0].icnf.a.item()
            hist.append(dict(it=it, loss=loss.item(), gate=gm, auc=gauc,
                             gg=gg, gi=gi, a=a_val))
            print(f'{it:>6} {loss.item():>9.5f} {gm:>11.5f} {gauc:>10.4f} '
                  f'{gg:>11.3e} {gi:>11.3e} {a_val:>9.5f}')

    print()
    first, last = hist[0], hist[-1]

    check('loss decreases', last['loss'] < first['loss'],
          f"({first['loss']:.5f} -> {last['loss']:.5f})")
    check('gate parameters actually move (mechanism is not frozen)',
          abs(last['gate'] - first['gate']) > 1e-4,
          f"(gate mean {first['gate']:.5f} -> {last['gate']:.5f})")
    check('sensor parameter a moves',
          abs(last['a'] - first['a']) > 1e-6,
          f"(a {first['a']:.5f} -> {last['a']:.5f})")
    grew = max(h['gg'] for h in hist) > 5 * hist[0]['gg'] if hist[0]['gg'] > 0 else True
    check('gradient reaching the gate grows as the freq branch trains', grew,
          f"({hist[0]['gg']:.2e} -> max {max(h['gg'] for h in hist):.2e})")
    check('gate stays spatially informative (AUC > 0.5)', last['auc'] > 0.5,
          f"(AUC {last['auc']:.4f})")
    check('no NaN in parameters',
          all(torch.isfinite(p).all() for p in net.parameters()))

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print('\n' + '=' * 76)
    print(f'PHASE 2b: {n_pass}/{len(RESULTS)} passed')
    print('=' * 76)
    for lbl, ok, det in RESULTS:
        if not ok:
            print(f'  FAILED: {lbl} {det}')
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
