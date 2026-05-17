import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class IAMLLoss(nn.Module):
    """Illumination-Aware Mirror Loss.

    Computes weighted L1 between standardized student and teacher projected
    features across 4 decoder scales, with per-pixel emphasis on dark regions.

    Args:
        beta: emphasis strength for dark pixels (default 0.6)
        eps:  standardization epsilon (default 1e-6)
    """

    def __init__(self, beta: float = 0.6, eps: float = 1e-6):
        super().__init__()
        self.beta = beta
        self.eps  = eps

    def forward(self, pairs: list, x_low: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pairs:  [(u1,t1), (u2,t2), (u3,t3), (u4,t4)]
            x_low:  (B, 3, H, W) low-light input in [0,1]
        Returns:
            scalar IAML loss
        """
        # ── Step 1: Luminance map ─────────────────────────────────────────────
        # L: (B, 1, H, W)
        L = (0.299 * x_low[:, 0:1, :, :]
           + 0.587 * x_low[:, 1:2, :, :]
           + 0.114 * x_low[:, 2:3, :, :])

        # ── Step 2: Per-image min-max normalisation ───────────────────────────
        B = x_low.shape[0]
        L_flat = L.view(B, -1)
        L_min  = L_flat.min(dim=1)[0].view(B, 1, 1, 1)
        L_max  = L_flat.max(dim=1)[0].view(B, 1, 1, 1)
        L_norm = (L - L_min) / (L_max - L_min + 1e-7)   # (B, 1, H, W)

        # ── Step 3: Emphasis weight — darker pixels weighted higher ───────────
        # W: (B, 1, H, W), range [1.0, 1.6]
        W = 1.0 + self.beta * (1.0 - L_norm)

        # ── Steps 4 & 5: Per-scale IAML ───────────────────────────────────────
        scale_losses = []

        for u_i, t_i in pairs:
            H_i, W_i = u_i.shape[2], u_i.shape[3]

            # 4a. Resize illumination weight to feature spatial size
            W_i_map = F.interpolate(W, size=(H_i, W_i),
                                    mode='bilinear', align_corners=False)  # (B,1,H_i,W_i)

            # 4b. Standardise student features per sample
            u_flat = u_i.view(B, -1)                           # (B, C*H_i*W_i)
            u_mean = u_flat.mean(dim=1).view(B, 1, 1, 1)
            u_std  = u_flat.std(dim=1).view(B, 1, 1, 1)
            u_norm = (u_i - u_mean) / (u_std + self.eps)       # (B, C, H_i, W_i)

            # 4c. Standardise teacher features per sample, then detach
            t_flat = t_i.view(B, -1)
            t_mean = t_flat.mean(dim=1).view(B, 1, 1, 1)
            t_std  = t_flat.std(dim=1).view(B, 1, 1, 1)
            t_norm = ((t_i - t_mean) / (t_std + self.eps)).detach()  # (B, C, H_i, W_i)

            # 4d-f. Weighted L1, averaged over all dimensions
            diff      = torch.abs(u_norm - t_norm)             # (B, C, H_i, W_i)
            weighted  = W_i_map * diff                          # broadcasts over C
            scale_losses.append(weighted.mean())

        # Step 5: mean over 4 scales
        return sum(scale_losses) / len(scale_losses)


class SSIMLoss(nn.Module):
    """1 - SSIM loss using pytorch_msssim."""

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        ssim_val = ssim(enhanced, clean, data_range=1.0, size_average=True)
        return 1.0 - ssim_val


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-3) -> torch.Tensor:
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


class TotalLoss(nn.Module):
    """L_total = L_Charb + L_SSIM + 0.8 * L_IAML.

    Returns a dict with keys: 'total', 'charb', 'ssim', 'iaml'.
    """

    def __init__(self):
        super().__init__()
        self.iaml_loss = IAMLLoss()
        self.ssim_loss = SSIMLoss()

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor,
                pairs: list, x_low: torch.Tensor) -> dict:
        charb = charbonnier_loss(enhanced, clean, eps=1e-3)
        ssim  = self.ssim_loss(enhanced, clean)
        iaml  = self.iaml_loss(pairs, x_low)
        return {
            'total': charb + ssim + 0.8 * iaml,
            'charb': charb,
            'ssim':  ssim,
            'iaml':  iaml,
        }
