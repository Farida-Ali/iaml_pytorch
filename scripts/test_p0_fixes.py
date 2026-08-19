"""
Phase-0 regression tests. Run before ANY training launch.

Covers the three defects that made earlier results unmeasurable:
  P0.1  silent auto-resume hijacking a run
  P0.2  FrequencyAwareLoss band normalisation (1:2 acting as 1:38)
  P0.3  IlluminationTVLoss supervising a constant (zero gradient)
  P0.4  a general gradient-flow harness so a dead loss can never ship again

  python scripts/test_p0_fixes.py
"""
import sys, os, shutil, tempfile, traceback
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch

RESULTS = []


def check(label, cond, detail=''):
    RESULTS.append((label, bool(cond), detail))
    print(f'  {"PASS" if cond else "FAIL"}  {label}' + (f'  {detail}' if detail else ''))
    return bool(cond)


# ───────────────────────────────────────────────────────────────────────── #
def test_auto_resume():
    print('\n' + '=' * 72)
    print('P0.1 — auto-resume is opt-in and refuses dangerous cases')
    print('=' * 72)
    from basicsr.train import resolve_auto_resume

    tmp = tempfile.mkdtemp()
    try:
        name = 'train_FD2RT_test'
        sd = os.path.join(tmp, name, 'training_states')
        os.makedirs(sd)

        base = {'name': name, 'train': {'total_iter': 250000}}

        # 1. No states at all -> start fresh, no error.
        empty = {'name': 'nonexistent_exp', 'train': {'total_iter': 250000}}
        check('no state dir -> returns None (fresh start)',
              resolve_auto_resume(empty, exp_root=tmp) is None)

        # 2. Stale state present, auto_resume OFF -> must RAISE.
        #    This is the exact A7-lite1 scenario.
        open(os.path.join(sd, '250000.state'), 'w').close()
        opt = dict(base, auto_resume=False)
        try:
            resolve_auto_resume(opt, exp_root=tmp)
            check('stale state + auto_resume off -> raises', False,
                  '(it returned instead of raising)')
        except RuntimeError as e:
            check('stale state + auto_resume off -> raises', True,
                  f'({str(e).splitlines()[0][:60]}...)')

        # 3. auto_resume ON but already at total_iter -> must RAISE.
        #    Resuming here trains zero iterations, which is what happened.
        opt = dict(base, auto_resume=True)
        try:
            resolve_auto_resume(opt, exp_root=tmp)
            check('auto_resume on, iter >= total_iter -> raises', False,
                  '(it returned instead of raising)')
        except RuntimeError as e:
            check('auto_resume on, iter >= total_iter -> raises', True,
                  f'({str(e).splitlines()[0][:60]}...)')

        # 4. auto_resume ON, genuine mid-run resume -> returns the path.
        os.remove(os.path.join(sd, '250000.state'))
        open(os.path.join(sd, '120000.state'), 'w').close()
        open(os.path.join(sd, '95000.state'), 'w').close()
        got = resolve_auto_resume(dict(base, auto_resume=True), exp_root=tmp)
        check('auto_resume on, mid-run -> resumes from NEWEST state',
              got is not None and got.endswith('120000.state'), f'({got})')

        # 5. Config default (key absent) behaves as OFF.
        try:
            resolve_auto_resume(dict(base), exp_root=tmp)
            check('auto_resume absent defaults to OFF', False)
        except RuntimeError:
            check('auto_resume absent defaults to OFF', True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ───────────────────────────────────────────────────────────────────────── #
def test_freq_loss():
    print('\n' + '=' * 72)
    print('P0.2 — FrequencyAwareLoss band weighting is honoured')
    print('=' * 72)
    from basicsr.models.losses import FrequencyAwareLoss

    torch.manual_seed(0)
    t = torch.rand(4, 3, 128, 128)
    p = t + 0.05 * torch.randn_like(t)

    low_only = FrequencyAwareLoss(loss_weight=1., w_low=1., w_high=0.)
    high_only = FrequencyAwareLoss(loss_weight=1., w_low=0., w_high=1.)
    l_lo, l_hi = low_only(p, t).item(), high_only(p, t).item()
    ratio = l_hi / max(l_lo, 1e-12)
    # With per-band normalisation and equal unit weights, the two bands measure
    # comparable per-coefficient error. White-ish residual => ratio near 1.
    check('per-band normalised: equal weights give comparable bands',
          0.2 < ratio < 5.0, f'(high/low = {ratio:.2f}, was ~19.5 before fix)')

    # And the configured weighting should now be respected.
    both = FrequencyAwareLoss(loss_weight=1., w_low=1., w_high=2.)
    expected = 1.0 * l_lo + 2.0 * l_hi
    got = both(p, t).item()
    check('L = w_low*L_low + w_high*L_high exactly',
          abs(got - expected) < 1e-6, f'(got {got:.6f}, expected {expected:.6f})')

    check('identical inputs -> 0', both(t, t.clone()).item() < 1e-6)

    leaf = torch.rand(2, 3, 64, 64, requires_grad=True)
    both(leaf, torch.rand(2, 3, 64, 64)).backward()
    check('gradient flows', leaf.grad is not None and leaf.grad.abs().sum() > 0)


# ───────────────────────────────────────────────────────────────────────── #
def test_tv_loss_target():
    print('\n' + '=' * 72)
    print('P0.3 — illumination TV loss supervises a LEARNED tensor')
    print('=' * 72)
    from basicsr.models.archs.fd2rt_a4_arch import FD2RT_A4
    from basicsr.models.losses import IlluminationTVLoss

    torch.manual_seed(0)
    net = FD2RT_A4(in_channels=3, out_channels=3, n_feat=40,
                   stage=1, num_blocks=[1, 2, 2])

    # The OLD target: DWT of the raw input. Proven dead.
    dead = {}
    for n, m in net.named_modules():
        if n.endswith('estimator.dwt'):
            m.register_forward_hook(lambda mo, i, o: dead.__setitem__('t', o[0]))
            break
    # The NEW target: the learned illumination map (estimator.conv2 output).
    live = {}
    for n, m in net.named_modules():
        if n.endswith('estimator.conv2'):
            m.register_forward_hook(lambda mo, i, o: live.__setitem__('t', o))
            break

    net(torch.rand(2, 3, 64, 64))
    check('old target (dwt of input) has no grad_fn -- confirms the bug',
          dead.get('t') is not None and dead['t'].grad_fn is None)
    check('new target (learned illu_map) IS in the autograd graph',
          live.get('t') is not None and live['t'].grad_fn is not None)

    net.zero_grad()
    IlluminationTVLoss(loss_weight=0.01)(live['t']).backward()
    tot = sum(p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None)
    check('TV on the new target produces NONZERO parameter gradient',
          tot > 0, f'(total |grad| = {tot:.6f})')


# ───────────────────────────────────────────────────────────────────────── #
def test_gradient_flow_harness():
    print('\n' + '=' * 72)
    print('P0.4 — gradient-flow harness catches any dead loss term')
    print('=' * 72)
    from scripts.gradient_flow import assert_loss_is_live

    net = torch.nn.Conv2d(3, 3, 3, padding=1)
    x = torch.rand(2, 3, 32, 32)

    live_ok, live_grad = assert_loss_is_live(
        net, lambda: torch.nn.functional.l1_loss(net(x), torch.rand(2, 3, 32, 32)))
    check('harness passes a live loss', live_ok, f'(|grad| = {live_grad:.4f})')

    const = torch.rand(2, 3, 16, 16)          # not connected to net at all
    dead_ok, dead_grad = assert_loss_is_live(net, lambda: const.abs().mean())
    check('harness FLAGS a constant (dead) loss', not dead_ok,
          f'(|grad| = {dead_grad:.6f})')


# ───────────────────────────────────────────────────────────────────────── #
if __name__ == '__main__':
    for fn in (test_auto_resume, test_freq_loss, test_tv_loss_target,
               test_gradient_flow_harness):
        try:
            fn()
        except Exception:
            traceback.print_exc()
            RESULTS.append((fn.__name__, False, 'raised'))

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print('\n' + '=' * 72)
    print(f'PHASE 0 TESTS: {n_pass}/{len(RESULTS)} passed')
    print('=' * 72)
    for lbl, ok, det in RESULTS:
        if not ok:
            print(f'  FAILED: {lbl} {det}')
    sys.exit(0 if n_pass == len(RESULTS) else 1)
