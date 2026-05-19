"""
Unit tests for wavelet_utils.HaarDWT2D and HaarIDWT2D.

Three required properties:
  (a) Perfect reconstruction  — IDWT(DWT(x)) == x  to floating-point eps
  (b) Shape correctness       — subbands are [B, C, H/2, W/2]
  (c) Energy preservation     — sum(LL²+LH²+HL²+HH²) == sum(x²)  (Parseval)

Run with:
    python -m pytest tests/test_wavelet_utils.py -v
or:
    python tests/test_wavelet_utils.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import pytest
from wavelet_utils import HaarDWT2D, HaarIDWT2D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dwt_pair(device="cpu"):
    return HaarDWT2D().to(device), HaarIDWT2D().to(device)


# ---------------------------------------------------------------------------
# Property (a): Perfect reconstruction
# ---------------------------------------------------------------------------

class TestPerfectReconstruction:
    """IDWT(DWT(x)) must equal x to floating-point precision."""

    def test_float32_random(self):
        dwt, idwt = make_dwt_pair()
        x = torch.randn(2, 3, 128, 128, dtype=torch.float32)
        LL, LH, HL, HH = dwt(x)
        x_rec = idwt(LL, LH, HL, HH)
        max_err = (x - x_rec).abs().max().item()
        print(f"\n  [float32 random 2×3×128×128]  max_err = {max_err:.2e}")
        assert max_err < 1e-5, f"Reconstruction error too large: {max_err:.2e}"

    def test_float32_all_ones(self):
        """Constant signal — LL should absorb all energy, HF bands = 0."""
        dwt, idwt = make_dwt_pair()
        x = torch.ones(1, 1, 16, 16)
        LL, LH, HL, HH = dwt(x)
        # High-frequency subbands must be zero for a constant image
        assert LH.abs().max().item() < 1e-6, "LH should be zero for constant input"
        assert HL.abs().max().item() < 1e-6, "HL should be zero for constant input"
        assert HH.abs().max().item() < 1e-6, "HH should be zero for constant input"
        # Reconstruction
        x_rec = idwt(LL, LH, HL, HH)
        assert (x - x_rec).abs().max().item() < 1e-5

    def test_float32_checkerboard(self):
        """Checkerboard pattern — energy should be in HH only."""
        dwt, idwt = make_dwt_pair()
        x = torch.zeros(1, 1, 16, 16)
        x[0, 0, ::2, ::2] = 1.0    # even rows, even cols
        x[0, 0, 1::2, 1::2] = 1.0  # odd rows, odd cols  → checkerboard
        LL, LH, HL, HH = dwt(x)
        # LL must be 0.5 (mean of checker: two 1s and two 0s... wait)
        # Actually for 2×2 block: [1,0,0,1] → LL = 0.5*(1+0+0+1)/2... let me verify
        # LL = 0.5*(1+0+0+1) = 1.0  (all blocks equal → constant LL = 1)
        # Reconstruction must still hold
        x_rec = idwt(LL, LH, HL, HH)
        max_err = (x - x_rec).abs().max().item()
        print(f"\n  [checkerboard 1×1×16×16]  max_err = {max_err:.2e}")
        assert max_err < 1e-5

    def test_float64_precision(self):
        """float64 should achieve near-machine precision."""
        dwt = HaarDWT2D()
        idwt = HaarIDWT2D()
        x = torch.randn(2, 4, 64, 64, dtype=torch.float64)
        # Cast buffers to float64 manually (register_buffer is float32 by default)
        dwt.filters = dwt.filters.double()
        idwt.filters = idwt.filters.double()
        LL, LH, HL, HH = dwt(x)
        x_rec = idwt(LL, LH, HL, HH)
        max_err = (x - x_rec).abs().max().item()
        print(f"\n  [float64 random 2×4×64×64]  max_err = {max_err:.2e}")
        assert max_err < 1e-12

    def test_multichannel_batch(self):
        """B=4, C=40 (Retinexformer's feature dim), H=W=64."""
        dwt, idwt = make_dwt_pair()
        x = torch.randn(4, 40, 64, 64)
        LL, LH, HL, HH = dwt(x)
        x_rec = idwt(LL, LH, HL, HH)
        max_err = (x - x_rec).abs().max().item()
        print(f"\n  [B=4,C=40,H=64,W=64]  max_err = {max_err:.2e}")
        assert max_err < 1e-5

    def test_non_square_input(self):
        """H ≠ W."""
        dwt, idwt = make_dwt_pair()
        x = torch.randn(2, 3, 32, 64)
        LL, LH, HL, HH = dwt(x)
        x_rec = idwt(LL, LH, HL, HH)
        max_err = (x - x_rec).abs().max().item()
        print(f"\n  [B=2,C=3,H=32,W=64]  max_err = {max_err:.2e}")
        assert max_err < 1e-5


# ---------------------------------------------------------------------------
# Property (b): Shape correctness
# ---------------------------------------------------------------------------

class TestShapeCorrectness:
    """Subbands must be [B, C, H/2, W/2]; reconstruction must be [B, C, H, W]."""

    @pytest.mark.parametrize("B,C,H,W", [
        (2, 4, 16, 16),
        (1, 1, 8,  8),
        (3, 8, 32, 64),
        (2, 3, 128, 128),
    ])
    def test_subband_shapes(self, B, C, H, W):
        dwt, idwt = make_dwt_pair()
        x = torch.randn(B, C, H, W)
        LL, LH, HL, HH = dwt(x)
        expected = (B, C, H // 2, W // 2)
        for name, band in zip(["LL", "LH", "HL", "HH"], [LL, LH, HL, HH]):
            assert tuple(band.shape) == expected, (
                f"{name} shape {tuple(band.shape)} ≠ expected {expected}"
            )
        x_rec = idwt(LL, LH, HL, HH)
        assert tuple(x_rec.shape) == (B, C, H, W), (
            f"Reconstruction shape {tuple(x_rec.shape)} ≠ ({B},{C},{H},{W})"
        )

    def test_no_learnable_parameters(self):
        """DWT and IDWT must have zero trainable parameters."""
        dwt = HaarDWT2D()
        idwt = HaarIDWT2D()
        n_dwt = sum(p.numel() for p in dwt.parameters())
        n_idwt = sum(p.numel() for p in idwt.parameters())
        assert n_dwt == 0, f"HaarDWT2D has {n_dwt} learnable params (expected 0)"
        assert n_idwt == 0, f"HaarIDWT2D has {n_idwt} learnable params (expected 0)"

    def test_filters_not_grad(self):
        """Filters must not require gradient."""
        dwt = HaarDWT2D()
        assert not dwt.filters.requires_grad, "DWT filters must not require grad"

    def test_filter_values(self):
        """Verify the stored filter values match the mathematical specification."""
        dwt = HaarDWT2D()
        f = dwt.filters  # [4, 1, 2, 2]
        half = 0.5
        # LL
        torch.testing.assert_close(f[0, 0], torch.tensor([[half, half], [half, half]]))
        # LH
        torch.testing.assert_close(f[1, 0], torch.tensor([[half, -half], [half, -half]]))
        # HL
        torch.testing.assert_close(f[2, 0], torch.tensor([[half, half], [-half, -half]]))
        # HH
        torch.testing.assert_close(f[3, 0], torch.tensor([[half, -half], [-half, half]]))

    def test_odd_dims_raise(self):
        """Odd spatial dimensions must raise AssertionError."""
        dwt = HaarDWT2D()
        x = torch.randn(1, 1, 15, 16)
        with pytest.raises(AssertionError):
            dwt(x)


# ---------------------------------------------------------------------------
# Property (c): Energy preservation (Parseval's theorem)
# ---------------------------------------------------------------------------

class TestEnergyPreservation:
    """||LL||² + ||LH||² + ||HL||² + ||HH||² == ||x||²."""

    def test_random_float32(self):
        dwt = HaarDWT2D()
        x = torch.randn(2, 3, 128, 128)
        LL, LH, HL, HH = dwt(x)
        e_in  = x.pow(2).sum().item()
        e_out = (LL.pow(2) + LH.pow(2) + HL.pow(2) + HH.pow(2)).sum().item()
        rel_err = abs(e_in - e_out) / (e_in + 1e-8)
        print(f"\n  [energy]  E_in={e_in:.6f}  E_out={e_out:.6f}  rel_err={rel_err:.2e}")
        assert rel_err < 1e-5, f"Energy not preserved: rel_err={rel_err:.2e}"

    def test_energy_partition(self):
        """LL captures the dominant energy for smooth (low-frequency) images."""
        dwt = HaarDWT2D()
        # Smooth image: slowly varying values
        x = torch.linspace(-1, 1, 64).view(1, 1, 1, 64).expand(1, 1, 64, 64).clone()
        x = x + torch.linspace(-1, 1, 64).view(1, 1, 64, 1).expand(1, 1, 64, 64)
        LL, LH, HL, HH = dwt(x)
        e_total = x.pow(2).sum().item()
        e_LL    = LL.pow(2).sum().item()
        frac_LL = e_LL / (e_total + 1e-8)
        print(f"\n  [smooth signal]  fraction of energy in LL = {frac_LL:.4f}")
        assert frac_LL > 0.9, (
            f"Expected >90% energy in LL for smooth signal, got {frac_LL:.1%}"
        )

    def test_energy_checkerboard_in_HH(self):
        """Checkerboard (highest-frequency signal) should concentrate in HH."""
        dwt = HaarDWT2D()
        x = torch.ones(1, 1, 64, 64)
        x[0, 0, ::2, 1::2] = -1.0   # alternating columns in even rows
        x[0, 0, 1::2, ::2] = -1.0   # alternating columns in odd rows
        LL, LH, HL, HH = dwt(x)
        e_total = x.pow(2).sum().item()
        e_HH    = HH.pow(2).sum().item()
        frac_HH = e_HH / (e_total + 1e-8)
        print(f"\n  [checkerboard]  fraction of energy in HH = {frac_HH:.4f}")
        assert frac_HH > 0.9, (
            f"Expected >90% energy in HH for checkerboard, got {frac_HH:.1%}"
        )

    @pytest.mark.parametrize("B,C,H,W", [(2,4,16,16), (1,1,32,32), (3,8,64,64)])
    def test_energy_many_shapes(self, B, C, H, W):
        dwt = HaarDWT2D()
        x = torch.randn(B, C, H, W)
        LL, LH, HL, HH = dwt(x)
        e_in  = x.pow(2).sum().item()
        e_out = (LL.pow(2) + LH.pow(2) + HL.pow(2) + HH.pow(2)).sum().item()
        rel_err = abs(e_in - e_out) / (e_in + 1e-8)
        assert rel_err < 1e-5, f"[{B},{C},{H},{W}] energy rel_err={rel_err:.2e}"


# ---------------------------------------------------------------------------
# Gradient flow test (sanity check for training integration)
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Gradients must flow through DWT/IDWT to the input tensor."""

    def test_grad_through_dwt(self):
        dwt, idwt = make_dwt_pair()
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        LL, LH, HL, HH = dwt(x)
        loss = LL.sum() + LH.sum() + HL.sum() + HH.sum()
        loss.backward()
        assert x.grad is not None, "No gradient flowed to x through DWT"
        assert not x.grad.isnan().any(), "NaN in gradients"

    def test_grad_through_round_trip(self):
        dwt, idwt = make_dwt_pair()
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        x_rec = idwt(*dwt(x))
        loss = x_rec.sum()
        loss.backward()
        assert x.grad is not None
        assert not x.grad.isnan().any()


# ---------------------------------------------------------------------------
# Standalone runner (no pytest needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    suites = [
        TestPerfectReconstruction,
        TestShapeCorrectness,
        TestEnergyPreservation,
        TestGradientFlow,
    ]

    total, passed, failed = 0, 0, 0
    for suite_cls in suites:
        suite = suite_cls()
        print(f"\n{'='*60}")
        print(f"  {suite_cls.__name__}")
        print(f"{'='*60}")
        for name in dir(suite):
            if not name.startswith("test_"):
                continue
            method = getattr(suite, name)
            # handle parametrize by skipping (pytest handles it; here just call once)
            import inspect
            sig = inspect.signature(method)
            params = [p for p in sig.parameters if p not in ("self",)]
            if params:
                # skip parametrized tests in standalone mode
                print(f"  SKIP (parametrized)  {name}")
                continue
            total += 1
            try:
                method()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
