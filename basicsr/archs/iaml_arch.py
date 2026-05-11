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


class DecoderBlock(nn.Module):
    """Single decoder level: ConvTranspose(stride=2) → BN → ReLU → Concat(skip) → CBAM.

    Returns concatenated+attended feature map of (out_ch + skip_ch) channels.
    """

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_ch + skip_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.deconv(x)))   # (B, out_ch, H*2, W*2)
        x = torch.cat([x, skip], dim=1)          # (B, out_ch+skip_ch, H*2, W*2)
        return self.cbam(x)


class StudentDecoder(nn.Module):
    """4-level U-Net decoder with CBAM, 1×1 projection convs, and residual output head.

    Channel trace for 256×256 input:
      d1: ConvTranspose(512→512) + cat(e4:512) = 1024ch @ H/16  →  u1: 256ch
      d2: ConvTranspose(1024→256) + cat(e3:256) = 512ch @ H/8   →  u2: 128ch
      d3: ConvTranspose(512→128) + cat(e2:128) = 256ch @ H/4    →  u3: 64ch
      d4: ConvTranspose(256→64) + cat(e1:64) = 128ch @ H/2      →  u4: 32ch
      d5: ConvTranspose(128→32) + cat(img:3) = 35ch @ H         →  out: 3ch

    Returns: (output_image, [u1, u2, u3, u4])
    """

    def __init__(self):
        super().__init__()
        # Decoder blocks
        self.dec1 = DecoderBlock(512,  512, skip_ch=512)   # d1: 1024ch
        self.dec2 = DecoderBlock(1024, 256, skip_ch=256)   # d2: 512ch
        self.dec3 = DecoderBlock(512,  128, skip_ch=128)   # d3: 256ch
        self.dec4 = DecoderBlock(256,  64,  skip_ch=64)    # d4: 128ch

        # 1×1 projection convs — outputs used by IAML loss
        self.proj1 = nn.Conv2d(1024, 256, kernel_size=1)   # u1
        self.proj2 = nn.Conv2d(512,  128, kernel_size=1)   # u2
        self.proj3 = nn.Conv2d(256,  64,  kernel_size=1)   # u3
        self.proj4 = nn.Conv2d(128,  32,  kernel_size=1)   # u4

        # Output head: upsample to full res, concat input image, residual prediction
        self.up5   = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.act5  = nn.ReLU(inplace=True)
        self.conv_out1 = nn.Conv2d(32 + 3, 32, kernel_size=3, padding=1)
        self.act_out1  = nn.ReLU(inplace=True)
        self.conv_out2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, e1, e2, e3, e4, e5, img):
        # (B, C, H, W) shapes for 256×256 input shown in comments
        d1 = self.dec1(e5, e4)          # (B, 1024, H/16, W/16)
        u1 = self.proj1(d1)             # (B, 256,  H/16, W/16)

        d2 = self.dec2(d1, e3)          # (B, 512,  H/8,  W/8)
        u2 = self.proj2(d2)             # (B, 128,  H/8,  W/8)

        d3 = self.dec3(d2, e2)          # (B, 256,  H/4,  W/4)
        u3 = self.proj3(d3)             # (B, 64,   H/4,  W/4)

        d4 = self.dec4(d3, e1)          # (B, 128,  H/2,  W/2)
        u4 = self.proj4(d4)             # (B, 32,   H/2,  W/2)

        # Output head
        d5 = self.act5(self.up5(d4))    # (B, 32,   H,    W)
        d5 = torch.cat([d5, img], dim=1)# (B, 35,   H,    W)
        out = self.act_out1(self.conv_out1(d5))   # (B, 32, H, W)
        out = self.conv_out2(out)                  # (B, 3,  H, W)
        out = torch.sigmoid(out + img)             # residual + sigmoid

        return out, [u1, u2, u3, u4]


# TeacherDecoder is architecturally identical to StudentDecoder.
# Using a class alias keeps the two objects independent (separate parameter sets)
# while sharing the same architecture definition.
TeacherDecoder = StudentDecoder


class IAMLNet(nn.Module):
    """Teacher-student low-light enhancement network with IAML loss support.

    One shared encoder processes both paths.
    Student decoder is trained with gradients.
    Teacher decoder is a frozen EMA copy of the student decoder.

    At inference time only encoder + student_decoder are used.
    """

    EMA_MOMENTUM = 0.999

    def __init__(self):
        super().__init__()
        self.encoder         = Encoder()
        self.student_decoder = StudentDecoder()
        self.teacher_decoder = TeacherDecoder()

        # Teacher starts with exact student weights; frozen from here on.
        self.teacher_decoder.load_state_dict(self.student_decoder.state_dict())
        for param in self.teacher_decoder.parameters():
            param.requires_grad = False

    def forward(self, x_low: torch.Tensor, x_clean: torch.Tensor):
        """Full teacher-student forward pass used during training.

        Student path: encoder(x_low, train=True) → student_decoder → enhanced, [u1..u4]
        Teacher path: encoder(x_clean, eval/no_grad) → teacher_decoder → [t1..t4]

        Returns:
            enhanced:  (B, 3, H, W) enhanced image
            pairs:     [(u1,t1), (u2,t2), (u3,t3), (u4,t4)]
        """
        # ── Student path (gradients flow) ────────────────────────────────────
        e1_s, e2_s, e3_s, e4_s, e5_s = self.encoder(x_low)
        enhanced, student_feats = self.student_decoder(e1_s, e2_s, e3_s, e4_s, e5_s, x_low)

        # ── Teacher path (no gradients; encoder runs in eval/running-stats mode) ──
        with torch.no_grad():
            self.encoder.eval()
            e1_t, e2_t, e3_t, e4_t, e5_t = self.encoder(x_clean)
            self.encoder.train()
            _, teacher_feats = self.teacher_decoder(e1_t, e2_t, e3_t, e4_t, e5_t, x_clean)

        pairs = list(zip(student_feats, teacher_feats))
        return enhanced, pairs

    @torch.no_grad()
    def ema_update(self):
        """EMA update of teacher decoder weights. Call after every optimizer.step().

        W_teacher = 0.999 * W_teacher + 0.001 * W_student
        Covers both parameters and BatchNorm running statistics.
        """
        mu = self.EMA_MOMENTUM
        # state_dict includes both learnable parameters and BN running stats
        for (s_name, s_val), (_, t_val) in zip(
            self.student_decoder.state_dict().items(),
            self.teacher_decoder.state_dict().items(),
        ):
            # num_batches_tracked is an integer counter; copy directly
            if 'num_batches_tracked' in s_name:
                t_val.copy_(s_val)
            else:
                t_val.mul_(mu).add_(s_val, alpha=1.0 - mu)

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Single-image inference using student encoder + decoder only."""
        with torch.no_grad():
            e1, e2, e3, e4, e5 = self.encoder(x)
            enhanced, _ = self.student_decoder(e1, e2, e3, e4, e5, x)
        return enhanced
