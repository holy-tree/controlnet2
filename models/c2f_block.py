"""C2FBlock: 粗细分支 + MAFC 多注意力融合条件特征精炼模块.

从 ReviveDiff (DenoisingNAFNet_arch.CFBlock) 移植并裁剪:
    - 彻底移除 Time MLP / FiLM 调制相关代码 (静态条件特征精炼)
    - DW_Expand 默认 1, 控制显存
    - 内置 Fusion (CA+SA+PA 多注意力门控) 来自 ReviveDiff fusion.Fusion
"""

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
        链式膨胀 (累积 RF):
            dconv1 (d=2) -> dconv2 (d=4) -> dconv3 (d=8)  -> x3_coarse
        -> MAFC Fusion(fine=x_fine, coarse=x3_coarse)
        -> conv3 (3x3 DW) -> *beta 残差
        -> norm2 -> conv4 (1x1) -> SimpleGate -> conv5 (1x1) -> *gamma 残差

    Args:
        c: 输入/输出通道数 (== cond_emb_feat 通道, ControlNet 标准配置下为 320)
        DW_Expand: 深度可分离扩展倍率 (Stage 1 固定 1)
        FFN_Expand: FFN 隐藏层扩展倍率 (默认 2)
        drop_out_rate: dropout 概率 (默认 0)
        reduction: ChannelAttention 的通道压缩倍率 (默认 8)
    """
    def __init__(self, c: int, DW_Expand: int = 1, FFN_Expand: int = 2,
                 drop_out_rate: float = 0.0, reduction: int = 8):
        super().__init__()
        dw_channel = c * DW_Expand

        # 主干
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

        # 融合
        self.fusion = Fusion(dw_channel // 2, reduction=reduction)

        # 主干到残差门控
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 3, 1, 1,
                               bias=True, groups=dw_channel // 2)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # FFN
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

        # 三路并行
        x1 = self.dconv1(x)        # 链式 1 (d=2)
        x2 = self.dconv2(x1)       # 链式 2 (d=4)
        x3 = self.dconv3(x2)       # 链式 3 (d=8), 累积 RF ~= 27x27

        # MAFC 融合 (fine=x_sca, coarse=x3_chain)
        x = self.fusion(x, x3)

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # FFN
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        out = y + x * self.gamma
        return out