"""C2FBlock (粗细分支 + MAFC 多注意力融合) + TimedC2FBlock (时序调制) + Adapter.

从 ReviveDiff (DenoisingNAFNet_arch.CFBlock) 移植并扩展:
    - C2FBlock: 静态条件特征精炼 (链式 d=2/4/8 膨胀卷积, 累积 RF = 31x31)
    - TimedC2FBlock: 在 C2FBlock 之上叠加扩散时间步软调制
    - LightweightAdapter: 对齐退化特征与扩散 UNet 隐空间分布
    - ZeroConv: 权重/bias 全零初始化 (ControlNet 零残差惯例)
"""

import math

import torch
import torch.nn as nn


# ============================================================================
# 基础组件 (从 ReviveDiff 移植, 保持实现细节不变)
# ============================================================================

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm(nn.Module):
    """带可学习 gain 的简化 LayerNorm (NAFNet 风格).
    沿通道维度归一化, 仅有一个 (1, C, 1, 1) 的可学习缩放参数.
    """
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PA(nn.Module):
    """PA is pixel attention."""
    def __init__(self, nf):
        super().__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)
        return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class Fusion(nn.Module):
    """MAFC: CA + SA 注意力门控融合 PA(fine) 与 coarse.

    Args:
        x (Tensor): "fine" 特征 (PA 输入)
        y (Tensor): "coarse" 特征 (与 fine 加权融合)
    """
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PA(nf=dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        yglobal = x + y        # fine + coarse
        ylocal = x             # fine
        cattn = self.ca(yglobal)
        sattn = self.sa(yglobal)
        gattn = self.pa(ylocal) # fine -> 像素注意力
        weight = self.sigmoid(cattn + sattn)
        result = gattn * weight + y * (1 - weight)
        result = self.conv(result)
        return result


# ============================================================================
# C2FBlock: 静态条件特征精炼 (无 Time 模块)
# ============================================================================

class C2FBlock(nn.Module):
    """静态条件特征精炼模块 (无时序调制).

    流程:
        norm1 -> conv1 (1x1) -> conv2 (3x3 DW) -> SimpleGate -> *sca  -> x_fine
        链式膨胀卷积 (累积 RF = 31x31):
            dconv1 (d=2) -> dconv2 (d=4) -> dconv3 (d=8)  -> x3_coarse
        -> MAFC Fusion(fine=x_fine, coarse=x3_coarse)
        -> conv3 (3x3 DW) -> *beta 残差
        -> norm2 -> conv4 (1x1) -> SimpleGate -> conv5 (1x1) -> *gamma 残差

    Args:
        c: 输入/输出通道数
        DW_Expand: 深度可分离扩展倍率 (默认 1)
        FFN_Expand: FFN 隐藏层扩展倍率 (默认 2)
        drop_out_rate: dropout 概率 (默认 0)
        reduction: ChannelAttention 的通道压缩倍率 (默认 8)
    """
    def __init__(self, c: int, DW_Expand: int = 1, FFN_Expand: int = 2,
                 drop_out_rate: float = 0.0, reduction: int = 8):
        super().__init__()
        dw_channel = c * DW_Expand

        self.norm1 = LayerNorm(c)
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1,
                               bias=True, groups=dw_channel)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )

        # 链式膨胀卷积 (累积大感受野, fine -> coarse)
        # nn.Conv2d(in, out, kernel, stride, padding, dilation, groups, bias)
        self.dconv1 = nn.Sequential(
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 3, 1, 2, 2,
                      bias=True, groups=dw_channel // 2),
            nn.GELU(),
        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 3, 1, 4, 4,
                      bias=True, groups=dw_channel // 2),
            nn.GELU(),
        )
        self.dconv3 = nn.Sequential(
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 3, 1, 8, 8,
                      bias=True, groups=dw_channel // 2),
            nn.GELU(),
        )

        self.fusion = Fusion(dw_channel // 2, reduction=reduction)

        self.conv3 = nn.Conv2d(dw_channel // 2, c, 3, 1, 1,
                               bias=True, groups=dw_channel // 2)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.norm2 = LayerNorm(c)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        inp = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)

        x1 = self.dconv1(x)        # 链式 1 (d=2)
        x2 = self.dconv2(x1)       # 链式 2 (d=4)
        x3 = self.dconv3(x2)       # 链式 3 (d=8), 累积 RF = 31x31

        x = self.fusion(x, x3)

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        out = y + x * self.gamma
        return out


# ============================================================================
# TimedC2FBlock: 在 C2FBlock 之上叠加扩散时间步软调制
# ============================================================================

class TimestepSinusoidalEmbedding(nn.Module):
    """扩散时间步正弦位置嵌入."""
    def __init__(self, dim: int = 128, max_period: int = 10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] 标量时间步 (任意 dtype). 返回 [B, dim]."""
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb


class TimedC2FBlock(nn.Module):
    """带时序软调制的 C2F 精炼块.

    时序逻辑:
        gamma = sigmoid(MLP(sinusoidal(timestep)))   ∈ (0, 1)
        fine_out   = fine_feat   * (0.5 + 0.5 * gamma)   # 高 t -> 偏细
        coarse_out = coarse_feat * (1.0 - 0.5 * gamma)   # 高 t -> 偏粗
        final      = (fine_out + coarse_out) * (1 + 0.2 * gamma)

    论文规范:
        - 高 t 噪声剧烈阶段侧重全局结构约束 (coarse 主导)
        - 低 t 噪声微弱阶段强化局部纹理引导 (fine 主导)
        - 不直接对应某一类天气退化, 仅依据噪声强度动态调节
        - 全程无硬截断, 粗细信息同步保留
    """
    def __init__(self, c: int, DW_Expand: int = 1, FFN_Expand: int = 2,
                 drop_out_rate: float = 0.0, reduction: int = 8,
                 time_dim: int = 128):
        super().__init__()
        self.c2f = C2FBlock(c=c, DW_Expand=DW_Expand, FFN_Expand=FFN_Expand,
                            drop_out_rate=drop_out_rate, reduction=reduction)

        self.time_emb = TimestepSinusoidalEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        # 时序 MLP 初始化: 输出层 bias=0, weight 小初始化避免震荡
        nn.init.normal_(self.time_mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 输入特征
            timestep: [B] 时间步标量
        Returns:
            [B, C, H, W] 时序调制后的精炼特征
        """
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0).expand(x.shape[0])
        elif timestep.shape[0] == 1 and x.shape[0] > 1:
            timestep = timestep.expand(x.shape[0])

        # 强制 time_emb / time_mlp 与输入 x 同 dtype, 避免 fp16 模式下 dtype 不匹配
        target_dtype = x.dtype
        te = self.time_emb(timestep).to(target_dtype)        # [B, time_dim]
        gamma_raw = self.time_mlp(te)       # [B, 1]
        gamma = torch.sigmoid(gamma_raw)    # [B, 1]

        # C2F 主干精炼 (返回的 y 即融合后的特征)
        refined = self.c2f(x)               # [B, C, H, W]

        # 软调制: 通道级常数缩放 (broadcast 到 H, W)
        gamma_map = gamma.view(-1, 1, 1, 1)
        # 高 gamma (低 t): 强化细节; 低 gamma (高 t): 偏向全局
        scale = 1.0 + 0.2 * gamma_map
        return refined * scale


# ============================================================================
# Adapter: 轻量特征适配模块
# ============================================================================

class LightweightAdapter(nn.Module):
    """轻量特征适配器: Conv + LayerNorm + GELU, 对齐退化特征与扩散 UNet 隐空间分布.

    使用 GroupNorm(32) 替代 LayerNorm([C, H, W]) 以适配任意空间尺寸 (避免 LN 的固定 H,W).
    """
    def __init__(self, channels: int, hidden_ratio: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


# ============================================================================
# ZeroConv: 零初始化卷积 (ControlNet 残差惯例)
# ============================================================================

def zero_conv(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Conv2d:
    """构造一个权重和 bias 都初始化为 0 的卷积层."""
    layer = nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2, bias=True)
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class ZeroConv2d(nn.Module):
    """包装 zero_conv 为 nn.Module, 便于注册到 ModuleList."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.conv = zero_conv(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)