"""
Gradient-flow verification.

The A7 runs shipped an illumination-TV term that contributed exactly zero
gradient: it was computed on the DWT of the raw input image, and HaarDWT2D has
no parameters, so the term was a constant w.r.t. every network weight. It never
crashed because the total loss stayed differentiable through the other terms,
and the sanity check that was supposed to catch it asserted `l_tv > 0` -- the
VALUE, not the gradient.

This module asserts the thing that actually matters: every loss term the config
enables must move at least one parameter. Run it before any training launch.

  from scripts.gradient_flow import assert_loss_is_live, audit_model_losses
"""
import torch


def assert_loss_is_live(model, loss_fn, tol=0.0):
    """Does `loss_fn()` produce a nonzero gradient on any model parameter?

    Args:
        model: nn.Module whose parameters should receive gradient.
        loss_fn: zero-arg callable returning a scalar loss tensor. It must run
            the forward pass itself so the graph is built fresh.
        tol: gradient magnitude must exceed this to count as live.

    Returns:
        (is_live, total_abs_grad)
    """
    model.zero_grad(set_to_none=True)
    loss = loss_fn()

    if not torch.is_tensor(loss):
        raise TypeError(f'loss_fn returned {type(loss)}, expected a Tensor')
    if loss.numel() != 1:
        raise ValueError(f'loss_fn returned shape {tuple(loss.shape)}, expected a scalar')

    # A term detached from the graph cannot be backwarded at all -- that is the
    # dead case, and it is a result, not an error.
    if not loss.requires_grad or loss.grad_fn is None:
        return False, 0.0

    loss.backward()
    total = sum(p.grad.abs().sum().item()
                for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    return total > tol, total


def audit_model_losses(model, terms, verbose=True):
    """Check a whole dict of {name: zero-arg loss callable}.

    Returns (all_live, {name: (is_live, total_abs_grad)}).
    """
    report = {}
    for name, fn in terms.items():
        try:
            report[name] = assert_loss_is_live(model, fn)
        except Exception as exc:                       # a broken term is a dead term
            report[name] = (False, float('nan'))
            if verbose:
                print(f'  ERROR  {name}: {exc}')

    if verbose:
        width = max((len(n) for n in report), default=4)
        print(f'  {"term".ljust(width)}  {"status":<6}  total |grad|')
        print('  ' + '-' * (width + 24))
        for name, (live, g) in report.items():
            print(f'  {name.ljust(width)}  {"LIVE" if live else "DEAD":<6}  {g:.6f}')

    all_live = all(live for live, _ in report.values())
    if verbose and not all_live:
        dead = [n for n, (live, _) in report.items() if not live]
        print(f'\n  {len(dead)} DEAD term(s): {", ".join(dead)}')
        print('  A dead term adds a constant to the logged loss and trains nothing.')
    return all_live, report
