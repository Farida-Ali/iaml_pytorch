"""
Pre-flight gate. Run this on the GPU box BEFORE launching any training.

Every check here corresponds to a failure that already cost this project real
GPU time or a real result:

  1. stale experiment state   -- a leftover 250000.state silently hijacked a run,
                                 which then "trained" for one second and reported
                                 a validation number belonging to other weights.
  2. dead loss term           -- the illumination-TV prior contributed exactly
                                 zero gradient for an entire 250K run.
  3. ladder drift             -- the A0 baseline carried a different grad-clip
                                 and LR schedule than A1/A4/A7, so "improvement
                                 over baseline" was unattributable.
  4. data present and sane    -- fail in seconds, not after the first epoch.
  5. model builds and steps   -- catch shape/dtype errors before the queue.

Usage:
    python scripts/preflight.py --opt Options/train_FD2RT_ICNF_LOL_v1.yml
    python scripts/preflight.py --opt <cfg> --skip-data     # code-only check

Exit code 0 means safe to launch. Anything else, do not launch.
"""
import sys, os, glob, argparse, traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import yaml
import torch

RESULTS = []


def check(label, ok, detail=''):
    RESULTS.append((label, bool(ok), detail))
    print(f'  {"PASS" if ok else "FAIL"}  {label}' + (f'  {detail}' if detail else ''))
    return bool(ok)


def section(title):
    print('\n' + '=' * 76)
    print(title)
    print('=' * 76)


# ───────────────────────────────────────────────────────────────────────── #
def check_no_stale_state(opt, exp_root):
    section('1. Experiment directory is clean')
    name = opt['name']
    exp_dir = os.path.join(exp_root, name)
    states = glob.glob(os.path.join(exp_dir, 'training_states', '*.state'))
    ckpts = glob.glob(os.path.join(exp_dir, 'models', '*.pth'))

    if not os.path.isdir(exp_dir):
        return check(f'{exp_root}/{name} does not exist (clean start)', True)

    print(f'      directory exists: {exp_dir}')
    print(f'      states: {len(states)}   checkpoints: {len(ckpts)}')
    if states:
        iters = sorted(int(os.path.basename(s)[:-6]) for s in states)
        total = opt.get('train', {}).get('total_iter', 0)
        detail = f'(newest iter {iters[-1]}, total_iter {total})'
        if opt.get('auto_resume', False):
            return check('stale state present but auto_resume is ON',
                         iters[-1] < total,
                         detail + ' — will resume; must be < total_iter')
        return check('no stale training state', False,
                     detail + ' — move the directory aside or rename the run')
    return check('no stale training state', True, '(no .state files)')


def check_ladder(opt):
    section('2. Ladder is a controlled experiment')
    try:
        from scripts.check_ladder_alignment import main as ladder_main
        rc = ladder_main()
        return check('all controlled hyperparameters match across the ladder',
                     rc == 0)
    except Exception as exc:
        return check('ladder alignment check ran', False, str(exc))


def check_data(opt, skip):
    section('3. Data is present and readable')
    if skip:
        return check('data check skipped (--skip-data)', True)
    ok = True
    for split in ('train', 'val'):
        ds = opt.get('datasets', {}).get(split, {})
        for key in ('dataroot_gt', 'dataroot_lq'):
            root = ds.get(key)
            if not root:
                continue
            path = os.path.join(_ROOT, root)
            n = len(glob.glob(os.path.join(path, '*.png')) +
                    glob.glob(os.path.join(path, '*.jpg')))
            ok &= check(f'{split}.{key} has images', n > 0, f'({n} files in {root})')
    return ok


def build_model(opt):
    from basicsr.models import archs  # noqa: ensures arch modules are importable
    import importlib
    net_opt = dict(opt['network_g'])
    net_type = net_opt.pop('type')

    for mod in ('fd2rt_icnf_arch', 'fd2rt_a4_arch', 'fd2rt_v1_arch',
                'fd2rt_a2_arch', 'RetinexFormer_arch'):
        try:
            m = importlib.import_module(f'basicsr.models.archs.{mod}')
            if hasattr(m, net_type):
                return getattr(m, net_type)(**net_opt), net_type
        except ImportError:
            continue
    raise ValueError(f'could not locate architecture {net_type}')


def check_model_and_losses(opt):
    section('4. Model builds, steps, and every loss term is LIVE')
    from scripts.gradient_flow import audit_model_losses

    net, net_type = build_model(opt)
    n_par = sum(p.numel() for p in net.parameters())
    check(f'{net_type} constructed', True, f'({n_par:,} params)')

    gt_size = opt['datasets']['train'].get('gt_size', 128)
    bs = 2
    lq = torch.rand(bs, 3, gt_size, gt_size)
    gt = torch.rand(bs, 3, gt_size, gt_size)

    net.train()
    out = net(lq)
    if isinstance(out, (list, tuple)):
        out = out[-1]
    check('forward pass produces correct shape', out.shape == gt.shape,
          f'{tuple(out.shape)}')
    check('forward output is finite', torch.isfinite(out).all())

    # Build every loss term the config enables, then prove each one moves
    # parameters. This is the check that would have caught the dead TV prior.
    import importlib
    loss_mod = importlib.import_module('basicsr.models.losses')
    train_opt = opt['train']
    terms = {}

    if train_opt.get('pixel_opt'):
        cfg = dict(train_opt['pixel_opt']); cfg.pop('type')
        cri_pix = torch.nn.L1Loss()
        # Bind per-lambda: Python's late binding would otherwise make every
        # closure below use whichever `cri` was assigned last.
        terms['l_pix'] = lambda c=cri_pix: c(_fwd(net, lq), gt)

    if train_opt.get('freq_opt'):
        cfg = dict(train_opt['freq_opt']); t = cfg.pop('type')
        cri_freq = getattr(loss_mod, t)(**cfg)
        terms['l_freq'] = lambda c=cri_freq: c(_fwd(net, lq), gt)

    if train_opt.get('tv_opt'):
        cfg = dict(train_opt['tv_opt']); t = cfg.pop('type')
        cri_tv = getattr(loss_mod, t)(**cfg)
        cap = {}
        for nm, mod in net.named_modules():
            if nm.endswith('estimator.conv2'):
                mod.register_forward_hook(
                    lambda m, i, o: cap.__setitem__('t', o))
                break

        def _tv():
            cap.clear()
            _fwd(net, lq)
            if 't' not in cap:
                raise RuntimeError('TV hook did not fire')
            return cri_tv(cap['t'])
        terms['l_tv'] = _tv

    print()
    all_live, _ = audit_model_losses(net, terms)
    check('every configured loss term produces gradient', all_live)

    # One real optimiser step, to catch anything that only shows up under update.
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    optim.zero_grad()
    torch.nn.functional.l1_loss(_fwd(net, lq), gt).backward()
    if train_opt.get('use_grad_clip'):
        torch.nn.utils.clip_grad_norm_(
            net.parameters(), train_opt.get('clip_grad_norm', 1.0))
    optim.step()
    check('one optimiser step completes with finite parameters',
          all(torch.isfinite(p).all() for p in net.parameters()))
    return all_live


def _fwd(net, lq):
    o = net(lq)
    return o[-1] if isinstance(o, (list, tuple)) else o


def check_env():
    section('5. Environment')
    cuda = torch.cuda.is_available()
    check('CUDA available', cuda,
          f'({torch.cuda.get_device_name(0)})' if cuda
          else '(CPU only — training will be impractically slow)')
    print(f'      torch {torch.__version__}')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt', required=True, help='training YAML to validate')
    ap.add_argument('--skip-data', action='store_true')
    ap.add_argument('--exp-root', default='experiments')
    args = ap.parse_args()

    with open(args.opt) as f:
        opt = yaml.safe_load(f)

    print('=' * 76)
    print(f'PRE-FLIGHT: {args.opt}')
    print(f'  run name  : {opt["name"]}')
    print(f'  model     : {opt.get("model_type")} / {opt["network_g"]["type"]}')
    print(f'  total_iter: {opt.get("train", {}).get("total_iter")}')
    print('=' * 76)

    check_no_stale_state(opt, args.exp_root)
    check_ladder(opt)
    check_data(opt, args.skip_data)
    try:
        check_model_and_losses(opt)
    except Exception:
        traceback.print_exc()
        RESULTS.append(('model/loss check ran', False, 'raised'))
    check_env()

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    hard_fail = [l for l, ok, _ in RESULTS
                 if not ok and 'CUDA' not in l and 'data check' not in l]

    print('\n' + '=' * 76)
    print(f'PRE-FLIGHT: {n_pass}/{len(RESULTS)} checks passed')
    print('=' * 76)
    for lbl, ok, det in RESULTS:
        if not ok:
            print(f'  FAILED: {lbl} {det}')

    if hard_fail:
        print('\nDO NOT LAUNCH. Fix the failures above first.')
        return 1
    print('\nSAFE TO LAUNCH:')
    print(f'  python -m basicsr.train --opt {args.opt}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
