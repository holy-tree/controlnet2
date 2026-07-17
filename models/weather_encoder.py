"""Weather Degradation Encoder: 退化感知专用编码器.

将 LQ RGB (512x512) 映射到与 SD2 UNet 对齐的四层多尺度特征金字塔 [F64, F32, F16, F8]:
    F64: (B, 320, 64, 64)   对应 UNet down0 输出
    F32: (B, 640, 32, 32)   对应 UNet down1 输出
    F16: (B, 1280, 16, 16)  对应 UNet down2 输出
    F8:  (B, 1280, 8, 8)    对应 UNet mid 输出 (在 8x8 而不是 16x16; mid 实际是 16x16,
                            但论文表述 F8 表示"第 4 层精炼", 实际空间分辨率由 avg_pool 决定)

设计要点:
    - 浅层 stem 提取低维纹理 (64 ch)
    - 三次 stride=2 下采样 512 -> 64
    - 逐级 1x1 conv 升维到 SD2 UNet down0/down1/down2/mid 通道
    - 后续 avg_pool 派生 F32/F16/F8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherDegradationEncoder(nn.Module):
    """退化感知编码器: LQ RGB -> 四级多尺度特征金字塔.

    输出通道: [320, 640, 1280, 1280] 严格对齐 SD2 UNet down0/down1/down2/mid.
    """

    def __init__(self, in_channels: int = 3, stem_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.stem_channels = stem_channels

        # 浅层低维纹理提取 (3 -> 64 -> 64), 保持分辨率 512x512
        self.stem_head = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, padding=1, stride=1),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, 3, padding=1, stride=1),
            nn.GELU(),
        )

        # 三次 stride=2 下采样: 512 -> 256 -> 128 -> 64
        self.down_blocks = nn.Sequential(
            nn.Conv2d(stem_channels, stem_channels, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, 3, padding=1, stride=2),
            nn.GELU(),
        )

        # 升维到 SD2 UNet 各层通道
        self.proj_64 = nn.Conv2d(stem_channels, 320, 1)
        self.proj_32 = nn.Conv2d(320, 640, 1)
        self.proj_16 = nn.Conv2d(640, 1280, 1)
        self.proj_8 = nn.Conv2d(1280, 1280, 1)

    def forward(self, lq_img: torch.Tensor):
        """
        Args:
            lq_img: [B, 3, H, W] 退化 RGB 图 (H, W 通常为 512, 需为 8 的倍数)
        Returns:
            [f64, f32, f16, f8]: 四级金字塔
                f64: [B, 320, H/8, W/8]
                f32: [B, 640, H/16, W/16]
                f16: [B, 1280, H/32, W/32]
                f8:  [B, 1280, H/64, W/64]
        """
        feat_512 = self.stem_head(lq_img)             # [B, 64, H, W]
        feat_64_base = self.down_blocks(feat_512)     # [B, 64, H/8, W/8]

        f64 = self.proj_64(feat_64_base)              # [B, 320, H/8, W/8]
        f32 = self.proj_32(F.avg_pool2d(f64, 2))      # [B, 640, H/16, W/16]
        f16 = self.proj_16(F.avg_pool2d(f32, 2))      # [B, 1280, H/32, W/32]
        f8 = self.proj_8(F.avg_pool2d(f16, 2))        # [B, 1280, H/64, W/64]

        return [f64, f32, f16, f8]