import torch
import torch.nn as nn
from basicsr.archs.iaml_cbam import CBAM


class EncoderBlock(nn.Module):
    """Single encoder level: Conv2d(stride=2) → BN → LeakyReLU(0.2) → optional CBAM."""

    def __init__(self, in_ch: int, out_ch: int, apply_attention: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.cbam = CBAM(out_ch) if apply_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x)))
        if self.cbam is not None:
            x = self.cbam(x)
        return x


class Encoder(nn.Module):
    """5-level U-Net encoder with stride-2 convolutions and CBAM attention.

    Level channels: [64, 128, 256, 512, 512]
    Level 1 has no CBAM; levels 2-5 include CBAM.

    Returns all 5 feature maps for use as skip connections.
    """

    def __init__(self):
        super().__init__()
        self.enc1 = EncoderBlock(3,   64,  apply_attention=False)  # → (B, 64,  H/2,  W/2)
        self.enc2 = EncoderBlock(64,  128, apply_attention=True)   # → (B, 128, H/4,  W/4)
        self.enc3 = EncoderBlock(128, 256, apply_attention=True)   # → (B, 256, H/8,  W/8)
        self.enc4 = EncoderBlock(256, 512, apply_attention=True)   # → (B, 512, H/16, W/16)
        self.enc5 = EncoderBlock(512, 512, apply_attention=True)   # → (B, 512, H/32, W/32)

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        return e1, e2, e3, e4, e5
