"""
Haar 2-D DWT / IDWT — pure PyTorch, no external libraries.

Physics motivation (FD²RT project):
  Retinex theory states illumination is low-frequency; reflectance and
  noise are high-frequency.  The Haar DWT cleanly separates these in one
  level: LL ≈ illumination proxy, HH ≈ noise/detail proxy.

Mathematical basis:
  1-D Haar analysis filters (length 2, stride 2):
      h (low-pass):  [+1, +1] * 0.5   (mean)
      g (high-pass): [+1, -1] * 0.5   (difference)

  2-D filters are outer products (both with stride=2, no padding):
      LL = h⊗h = [[+1,+1],[+1,+1]] * 0.5   smooth approximation
      LH = h⊗g = [[+1,-1],[+1,-1]] * 0.5   h-row, g-col (horizontal edges)
      HL = g⊗h = [[+1,+1],[-1,-1]] * 0.5   g-row, h-col (vertical edges)
      HH = g⊗g = [[+1,-1],[-1,+1]] * 0.5   diagonal / noise

  Synthesis filters are identical to analysis (Haar is orthogonal and
  symmetric under this normalisation), applied via conv_transpose2d.

  Parseval identity holds exactly:
      ||LL||² + ||LH||² + ||HL||² + ||HH||² = ||x||²

No learnable parameters.  Filters are stored in register_buffer so they
move with .to(device) / .cuda() automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Analysis filters (shared constant — build once at module load time)
# Shape: [4, 1, 2, 2]  (4 filters, 1 input channel, 2×2 kernel)
# ---------------------------------------------------------------------------
_HAAR_FILTERS = torch.tensor(
    [
        [[[ 1.,  1.], [ 1.,  1.]]],   # LL  h⊗h
        [[[ 1., -1.], [ 1., -1.]]],   # LH  h⊗g
        [[[ 1.,  1.], [-1., -1.]]],   # HL  g⊗h
        [[[ 1., -1.], [-1.,  1.]]],   # HH  g⊗g
    ],
    dtype=torch.float32,
) * 0.5
# _HAAR_FILTERS[k] is the k-th 2×2 analysis kernel, scaled by ½


class HaarDWT2D(nn.Module):
    """
    Single-level 2-D Haar Discrete Wavelet Transform.

    Input:  x  [B, C, H, W]   — H and W must be even
    Output: (LL, LH, HL, HH)  — each [B, C, H/2, W/2]

    Implementation:
      1. Merge B and C into a single "batch" axis → [B*C, 1, H, W]
      2. Apply all 4 analysis kernels in one grouped conv2d call with stride=2
         → [B*C, 4, H/2, W/2]
      3. Reshape back to [B, C, 4, H/2, W/2] and split along dim-2.
    """

    def __init__(self) -> None:
        super().__init__()
        # register_buffer: not a parameter (no grad), moves with .to() / .cuda()
        self.register_buffer("filters", _HAAR_FILTERS.clone())
        # self.filters: [4, 1, 2, 2]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor [B, C, H, W]
        Returns:
            LL [B, C, H/2, W/2] — low-frequency (illumination proxy)
            LH [B, C, H/2, W/2] — horizontal edges
            HL [B, C, H/2, W/2] — vertical edges
            HH [B, C, H/2, W/2] — diagonal detail / noise proxy
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, (
            f"HaarDWT2D requires even spatial dims, got H={H}, W={W}"
        )

        # [B, C, H, W] → [B*C, 1, H, W]
        x_flat = x.reshape(B * C, 1, H, W)

        # conv2d: [B*C, 1, H, W] × [4, 1, 2, 2] → [B*C, 4, H/2, W/2]
        out = F.conv2d(x_flat, self.filters, stride=2, padding=0)

        # [B*C, 4, H/2, W/2] → [B, C, 4, H/2, W/2]
        out = out.reshape(B, C, 4, H // 2, W // 2)

        LL = out[:, :, 0]   # [B, C, H/2, W/2]
        LH = out[:, :, 1]
        HL = out[:, :, 2]
        HH = out[:, :, 3]

        return LL, LH, HL, HH


class HaarIDWT2D(nn.Module):
    """
    Single-level 2-D Haar Inverse Discrete Wavelet Transform.

    Input:  (LL, LH, HL, HH)  — each [B, C, H/2, W/2]
    Output: x  [B, C, H, W]

    For the Haar wavelet at this normalisation, synthesis filters equal
    analysis filters.  Each subband is upsampled via conv_transpose2d
    (stride=2) and the four results are summed.

    Implementation:
      For each subband S and its synthesis filter f_S (shape [1,1,2,2]):
          F.conv_transpose2d(S_flat, f_S, stride=2) → [B*C, 1, H, W]
      Output = sum over all four subbands, reshaped to [B, C, H, W].
    """

    def __init__(self) -> None:
        super().__init__()
        # Synthesis filters = analysis filters for Haar
        # Store as [4, 1, 2, 2] for easy indexing
        self.register_buffer("filters", _HAAR_FILTERS.clone())
        # Each self.filters[k] has shape [1, 2, 2];
        # conv_transpose2d needs [in_ch, out_ch, kH, kW] = [1, 1, 2, 2]

    def forward(
        self,
        LL: torch.Tensor,
        LH: torch.Tensor,
        HL: torch.Tensor,
        HH: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            LL, LH, HL, HH: each [B, C, H/2, W/2]
        Returns:
            x: [B, C, H, W]
        """
        B, C, Hd, Wd = LL.shape

        def _synth(subband: torch.Tensor, filt_idx: int) -> torch.Tensor:
            # subband: [B, C, Hd, Wd] → flat: [B*C, 1, Hd, Wd]
            flat = subband.reshape(B * C, 1, Hd, Wd)
            # kernel for conv_transpose2d: [in_ch, out_ch, kH, kW] = [1, 1, 2, 2]
            kernel = self.filters[filt_idx].unsqueeze(0)  # [1, 1, 2, 2]
            # conv_transpose2d: [B*C, 1, Hd, Wd] → [B*C, 1, 2*Hd, 2*Wd]
            return F.conv_transpose2d(flat, kernel, stride=2, padding=0)

        # Sum contributions from all four subbands
        out = _synth(LL, 0) + _synth(LH, 1) + _synth(HL, 2) + _synth(HH, 3)
        # out: [B*C, 1, H, W] → [B, C, H, W]
        return out.reshape(B, C, Hd * 2, Wd * 2)
