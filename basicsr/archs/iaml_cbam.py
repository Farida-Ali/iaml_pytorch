import torch
import torch.nn as nn


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Channel attention: GlobalAvgPool + GlobalMaxPool → shared MLP → sigmoid → add → multiply input
    Spatial attention: [channel_avg, channel_max] → Conv2d(k=7) → sigmoid → multiply input
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)

        # Shared MLP for channel attention (both avg and max paths use same weights)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
        )

        # Spatial attention conv: 2 input channels (avg + max along channel dim)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Channel attention ─────────────────────────────────────────────────
        # x: (B, C, H, W)
        avg = x.mean(dim=[2, 3])          # (B, C)
        max_ = x.amax(dim=[2, 3])         # (B, C)

        # Both paths through THE SAME MLP weights, then sigmoid
        avg_att = torch.sigmoid(self.shared_mlp(avg))    # (B, C)
        max_att = torch.sigmoid(self.shared_mlp(max_))   # (B, C)

        channel_att = (avg_att + max_att).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        x = x * channel_att

        # ── Spatial attention ─────────────────────────────────────────────────
        avg_spatial = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        max_spatial = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)

        spatial_att = torch.sigmoid(
            self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1))
        )  # (B, 1, H, W)

        return x * spatial_att
