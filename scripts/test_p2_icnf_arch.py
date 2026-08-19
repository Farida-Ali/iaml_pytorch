"""
Phase-2 tests: ICNF integrated into the architecture.

Bar: the new model must (1) run, (2) be gradient-healthy everywhere,
(3) reduce to A4/A1 behaviour at initialisation, (4) load A4 checkpoints with
only the expected key differences, and (5) actually gate spatially rather than
collapsing to a constant.

  python scripts/test_p2_icnf_arch.py
"""
import sys, os, traceback
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch

from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4
from basicsr.models.archs.fd2rt_icnf_arch import FD2RT_ICNF, ICNF_Block_Dual

RESULTS = []
KW = dict(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])


def check(label, cond, detail=''):
    RESULTS.append((label, bool(cond), detail))
    print(f'  {"PASS" if cond else "FAIL"}  {label}' + (f'  {detail}' if detail else ''))
    return bool(cond)


def test_forward():
    print('\n' + '=' * 74)
    print('P2.1 — forward pass, shapes, parameter budget')
    print('=' * 74)
    torch.manual_seed(0)
    net = FD2RT_ICNF(**KW)
    x = torch.rand(2, 3, 128, 128)
    y = net(x)
    check('output shape matches input', tuple(y.shape) == (2, 3, 128, 128),
          f'{tuple(y.shape)}')
    check('output is finite', torch.isfinite(y).all())

    a4 = FD2RT_A4(**KW)
    n_a4 = sum(p.numel() for p in a4.parameters())
    n_ic = sum(p.numel() for p in net.parameters())
    print(f'      A4        : {n_a4:>10,}')
    print(f'      FD2RT_ICNF: {n_ic:>10,}   ({n_ic - n_a4:+,})')
    # Replaces per-block scalar gates with (slope,bias) pairs, adds ICNF's
    # (a,b). Should be a rounding error against 2.17M.
    check('parameter delta vs A4 is negligible', abs(n_ic - n_a4) < 500,
          f'({n_ic - n_a4:+d} params)')

    # Non-square and odd-ish sizes (DWT needs even dims).
    for shape in [(1, 3, 64, 96), (2, 3, 96, 64)]:
        out = net(torch.rand(*shape))
        check(f'handles shape {shape}', tuple(out.shape) == shape)


def test_gradients():
    print('\n' + '=' * 74)
    print('P2.2 — every parameter group trains')
    print('=' * 74)
    from scripts.gradient_flow import assert_loss_is_live

    torch.manual_seed(0)
    net = FD2RT_ICNF(**KW)
    x, gt = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)

    live, g = assert_loss_is_live(
        net, lambda: torch.nn.functional.l1_loss(net(x), gt))
    check('L1 through the full model is live', live, f'|grad|={g:.4f}')

    net.zero_grad()
    torch.nn.functional.l1_loss(net(x), gt).backward()

    groups = {'icnf sensor (a,b)': 'icnf.log_',
              'gate slope/bias': '.gates.',
              'freq branch': 'freq_blocks.',
              'spatial branch': '.blocks.',
              'estimator': 'estimator.'}
    for label, needle in groups.items():
        ps = [p for n, p in net.named_parameters() if needle in n]
        tot = sum(p.grad.abs().sum().item() for p in ps if p.grad is not None)
        check(f'{label} receives gradient', len(ps) > 0 and tot > 1e-14,
              f'({len(ps)} tensors, |grad|={tot:.3e})')

    check('no NaN/Inf in any gradient',
          all(torch.isfinite(p.grad).all()
              for p in net.parameters() if p.grad is not None))


def test_init_reduces_to_a4():
    print('\n' + '=' * 74)
    print('P2.3 — at init the frequency path is ~off (reduces to A1 behaviour)')
    print('=' * 74)
    torch.manual_seed(0)
    net = FD2RT_ICNF(**KW).eval()
    x = torch.rand(2, 3, 64, 64)

    gates = []
    for m in net.modules():
        if isinstance(m, ICNF_Block_Dual):
            ev = net.body[0].icnf(x)
            for g in m.gates:
                gates.append(g(ev).mean().item())
    # The property that matters is that the model FUNCTION starts close to A1,
    # which comes from the small out_proj init -- not from the gate being shut.
    # A shut gate would starve the mechanism (13x less gradient at bias 6.0 vs
    # 2.0), so we assert the gate is responsive instead.
    check('gates start responsive, not saturated',
          0.02 < max(gates) < 0.95,
          f'(max mean gate {max(gates):.4f} over {len(gates)} gates)')

    # The direct test: disabling the frequency branch entirely should barely
    # change the output at initialisation.
    with torch.no_grad():
        y_on = net(x)
        for m in net.modules():
            if isinstance(m, ICNF_Block_Dual):
                for g in m.gates:
                    g.bias.fill_(50.0)          # force gate -> 0
        y_off = net(x)
    rel = (y_on - y_off).abs().max().item() / y_on.abs().max().item()
    check('output barely changes when freq branch is forced off at init',
          rel < 1e-3, f'(max rel diff {rel:.3e})')


def test_checkpoint_compat():
    print('\n' + '=' * 74)
    print('P2.4 — A4 checkpoints load; only gate keys differ')
    print('=' * 74)
    torch.manual_seed(0)
    a4 = FD2RT_A4(**KW)
    net = FD2RT_ICNF(**KW)

    missing, unexpected = net.load_state_dict(a4.state_dict(), strict=False)
    bad_missing = [k for k in missing if '.gates.' not in k and 'icnf.' not in k]
    bad_unexpected = [k for k in unexpected if '.gates.' not in k]

    print(f'      missing   : {len(missing)} (gates/icnf: '
          f'{len(missing) - len(bad_missing)})')
    print(f'      unexpected: {len(unexpected)} (gates: '
          f'{len(unexpected) - len(bad_unexpected)})')
    check('no unexpected missing keys beyond gates/icnf', not bad_missing,
          f'{bad_missing[:3]}')
    check('no unexpected extra keys beyond gates', not bad_unexpected,
          f'{bad_unexpected[:3]}')

    # The transplanted weights must actually be the A4 weights.
    a4_sd, ic_sd = a4.state_dict(), net.state_dict()
    shared = [k for k in a4_sd if k in ic_sd and ic_sd[k].shape == a4_sd[k].shape]
    same = all(torch.equal(a4_sd[k], ic_sd[k]) for k in shared)
    check('all shared weights copied bit-exactly', same,
          f'({len(shared)} tensors)')


def test_gate_is_spatial():
    print('\n' + '=' * 74)
    print('P2.5 — the gate really varies across space (not a scalar in disguise)')
    print('=' * 74)
    torch.manual_seed(0)
    net = FD2RT_ICNF(**KW)

    # Half flat, half textured, under a strong illumination ramp.
    x = torch.rand(1, 3, 128, 128) * 0.02 + 0.1
    x[:, :, :, 64:] += 0.15 * torch.rand(1, 3, 128, 64)
    ramp = torch.linspace(0.3, 1.0, 128).view(1, 1, -1, 1)
    x = (x * ramp).clamp(1e-3, 1)

    ev = net.body[0].icnf(x)
    check('evidence varies spatially', ev.std().item() > 1e-4,
          f'(std {ev.std().item():.4f}, range [{ev.min():.3f}, {ev.max():.3f}])')

    blk = next(m for m in net.modules() if isinstance(m, ICNF_Block_Dual))
    with torch.no_grad():
        blk.gates[0].bias.fill_(1.0)            # open the operating point
        g = blk.gates[0](ev)
    check('gate varies spatially', g.std().item() > 1e-3,
          f'(std {g.std().item():.4f}, range [{g.min():.3f}, {g.max():.3f}])')
    check('gate is higher on the textured half',
          g[:, :, :, 64:].mean().item() > g[:, :, :, :64].mean().item(),
          f'(textured {g[:, :, :, 64:].mean():.4f} vs '
          f'flat {g[:, :, :, :64].mean():.4f})')


def test_ablation_arms():
    print('\n' + '=' * 74)
    print('P2.6 — ablation arms all construct and run')
    print('=' * 74)
    x = torch.rand(1, 3, 64, 64)
    for src in ('input', 'illu_map'):
        for mode in ('subtract', 'ratio'):
            net = FD2RT_ICNF(**KW, illum_source=src, icnf_mode=mode)
            y = net(x)
            check(f'illum_source={src}, icnf_mode={mode}',
                  torch.isfinite(y).all() and y.shape == x.shape)
    try:
        FD2RT_ICNF(**KW, illum_source='bogus')
        check('invalid illum_source rejected', False)
    except ValueError:
        check('invalid illum_source rejected', True)


if __name__ == '__main__':
    for fn in (test_forward, test_gradients, test_init_reduces_to_a4,
               test_checkpoint_compat, test_gate_is_spatial, test_ablation_arms):
        try:
            fn()
        except Exception:
            traceback.print_exc()
            RESULTS.append((fn.__name__, False, 'raised'))

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print('\n' + '=' * 74)
    print(f'PHASE 2 TESTS: {n_pass}/{len(RESULTS)} passed')
    print('=' * 74)
    for lbl, ok, det in RESULTS:
        if not ok:
            print(f'  FAILED: {lbl} {det}')
    sys.exit(0 if n_pass == len(RESULTS) else 1)
