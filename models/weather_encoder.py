"""Weather Degradation Encoder: 退化感知专用编码器.

将 LQ RGB (512x512) 映射到与 SD2 UNet 对齐的四层多尺度特征金字塔 [F64, F32, F16, F8]:
    F64: (B, 320, 64, 64)   对应 UNet down0 输出
    F32: (B, 640, 32, 32)   对应 UNet down1 输出
    F16: (B, 1280, 16, 16)  对应 UNet down2 输出
    F8:  (B, 1280, 8, 8)    对应 UNet mid 输出 (在 8x8 而不是 16x16; mid 实际是 16x16,
                            但论文表述 F8 表示"第 4 层精炼", 实际空间分辨率由 avg_pool 决定)

设计要点 (Phase 4: 多尺度增强):
    - 浅层 stem 提取低维纹理 (64 ch)
    - 三次 stride=2 下采样 512 -> 64 (保持原结构, 向后兼容老 checkpoint)
    - 新增 F128 输出 (bilinear 上采样到 64x64 注入到 F64) 解决高频细节丢失问题
    - 逐级 1x1 conv 升维到 SD2 UNet down0/down1/down2/mid 通道
    - 后续 avg_pool 派生 F32/F16/F8

向后兼容:
    - 老 checkpoint 里没有 proj_128, strict=False load_state_dict 自动跳过
    - proj_128 从零初始化, 在已有 checkpoint-152000 基础上继续训练即可
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherDegradationEncoder(nn.Module):
    """退化感知编码器: LQ RGB -> 四级多尺度特征金字塔.

    输出通道: [320, 640, 1280, 1280] 严格对齐 SD2 UNet down0/down1/down2/mid.

    Phase 4 增强:
        - 新增 proj_128 (1x1 Conv2d 64→320), 把 128x128 中间层特征投影到 UNet down0 通道
        - 通过 bilinear 上采样到 64x64 加到 F64, 注入高频信息 (雨丝、雪粒)
        - 老权重 (proj_64/32/16/8, stem_head, down_blocks) 完全保留
        - 新 proj_128 初始化后从头学, 老 checkpoint 加载时 strict=False 自动跳过
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
        # 注意: proj_64 输入仍是 stem_channels (从 down_blocks 输出取)
        self.proj_64 = nn.Conv2d(stem_channels, 320, 1)
        # Phase 4 新增: proj_128 把 128x128 特征投影到 320 通道, 上采样注入 F64
        # 关键: weight/bias 初始化为 0 → 训练初期 F128 贡献为 0, 不破坏已学 F64
        # 训练过程中 proj_128 慢慢学到东西, F128 贡献逐渐出现
        self.proj_128 = nn.Conv2d(stem_channels, 320, 1)
        nn.init.zeros_(self.proj_128.weight)
        nn.init.zeros_(self.proj_128.bias)
        self.proj_32 = nn.Conv2d(320, 640, 1)
        self.proj_16 = nn.Conv2d(640, 1280, 1)
        self.proj_8 = nn.Conv2d(1280, 1280, 1)

    def forward(self, lq_img: torch.Tensor):
        """
        Args:
            lq_img: [B, 3, H, W] 退化 RGB 图 (H, W 通常为 512, 需为 8 的倍数)
        Returns:
            [f64, f32, f16, f8]: 四级金字塔 (F64 已含高频注入)
                f64: [B, 320, H/8, W/8]   ← 含 F128 上采样注入
                f32: [B, 640, H/16, W/16]
                f16: [B, 1280, H/32, W/32]
                f8:  [B, 1280, H/64, W/64]
        """
        feat_512 = self.stem_head(lq_img)             # [B, 64, H, W]

        # === Phase 4: 多尺度特征 ===
        # 用同一个 down_blocks 的子模块取中间层 (向后兼容, 不改 down_blocks 结构)
        # Sequential 索引: [Conv, GELU, Conv, GELU, Conv, GELU]
        # 每个 Conv 单独 stride=2, 用 slice 拿到 Conv+GELU pair
        feat_256 = self.down_blocks[0:2](feat_512)     # [B, 64, H/2, W/2]  (Conv0+GELU)
        feat_128 = self.down_blocks[2:4](feat_256)     # [B, 64, H/4, W/4]  (Conv1+GELU)
        feat_64_base = self.down_blocks[4:6](feat_128)  # [B, 64, H/8, W/8]  (Conv2+GELU)

        # F128 → 320ch, 上采样到 64x64, 加到 F64 (高频细节注入)
        f128 = self.proj_128(feat_128)                 # [B, 320, H/4, W/4]
        f64_base = self.proj_64(feat_64_base)         # [B, 320, H/8, W/8]
        f128_up = F.interpolate(f128, size=f64_base.shape[-2:],
                                mode='bilinear', align_corners=False)
        f64 = f64_base + f128_up                       # [B, 320, H/8, W/8]

        f32 = self.proj_32(F.avg_pool2d(f64, 2))      # [B, 640, H/16, W/16]
        f16 = self.proj_16(F.avg_pool2d(f32, 2))      # [B, 1280, H/32, W/32]
        f8 = self.proj_8(F.avg_pool2d(f16, 2))        # [B, 1280, H/64, W/64]

        return [f64, f32, f16, f8]