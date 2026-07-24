"""Decoder Skip Connection: LQ → SD2 UNet decoder skip.

用户洞察: encoder 端的细节保留是局部优化, 但 decoder (up_blocks) 才是细节
重建的主战场. SD2 UNet 的 decoder 通过 up_blocks.resnets 接收 down_block 的
skip connections; 但这些 skip 已经被 8x8 → 64x64 的下采样链路磨平, 缺乏 LQ
原始高频细节.

本模块从编码器原生三尺度特征 (feat_256 / feat_128 / feat_64) 直接构造 decoder
各层 skip 残差, 通过 forward pre-hook 注入到 SD2 UNet up_blocks 的
res_hidden_states_tuple. 禁用 feat_512 以降低浅层雨雪噪声被直接复刻的风险.

通道匹配 (SD2 UNet up_blocks 期望的 down_block skip 通道):
    up_block_0 (16x16): 1280ch
    up_block_1 (32x32): 640ch
    up_block_2 (64x64): 320ch

特征映射 (Phase 6 修订):
    feat_64  (64x64,  64ch) → up_block_0 (16x16,  1280ch)  # 4× AvgPool + 1×1 proj
    feat_128 (128x128, 64ch) → up_block_1 (32x32,  640ch)   # 4× AvgPool + 1×1 proj
    feat_256 (256x256, 64ch) → up_block_2 (64x64,  320ch)   # 4× AvgPool + 1×1 proj

输出顺序: dict keys 为 'up_block_0' / 'up_block_1' / 'up_block_2',
与现有 forward_pre_hook 的消费顺序 (block_idx → dict[f'up_block_{idx}']) 一致.

全部 init=0 (ZeroConv 权重/偏置 + alpha 标量均 init=0), 训练初期
DecoderSkipPath 总贡献严格为 0, 不破坏 SD2 预训练.
"""

import torch
import torch.nn as nn

from .c2f_block import ZeroConv2d


class _DecoderBranch(nn.Module):
    """单个 decoder skip 分支: 2× AvgPool (4× 降采样) + 1×1 通道投影 + ZeroConv + alpha.

    Args:
        in_ch:   输入特征通道 (固定 64, 与 WeatherDegradationEncoder 一致)
        out_ch:  输出通道 (匹配对应 up_block 的 skip 通道)
        alpha_init: alpha 标量初始值 (默认 0, 训练初期输出严格为 0)
    """

    def __init__(self, in_ch: int, out_ch: int, alpha_init: float = 0.0):
        super().__init__()
        # 固定 AvgPool: 2× stride=2, 调用两次共 4× 下采样
        # 无可学习参数, 保证分支只有"1×1 通道投影"这一个学习环节
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 1×1 通道投影 (唯一的学习投影, 与 UNet 原生 skip 通道对齐)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        # ZeroConv 包装: 权重/偏置均初始化为 0
        self.zero_conv = ZeroConv2d(out_ch, out_ch)
        # 独立可学习 alpha (init=0, 双重保险)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=True)

        # 严格 init=0 安全保险: 即使 proj 偏离, ZeroConv 仍锁死为 0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # ZeroConv2d 内部已经 zero_init, 这里冗余保险
        nn.init.zeros_(self.zero_conv.conv.weight)
        nn.init.zeros_(self.zero_conv.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 4× 下采样 (调用 pool 两次: 64→32→16 / 128→64→32 / 256→128→64)
        x = self.pool(x)
        x = self.pool(x)
        # 1×1 通道投影
        x = self.proj(x)
        # ZeroConv (init=0)
        x = self.zero_conv(x)
        # alpha 缩放 (init=0)
        x = x * self.alpha
        return x


class DecoderSkipPath(nn.Module):
    """LQ 三尺度浅层特征 → SD2 UNet decoder skip 注入.

    输入 (Phase 6 修订):
        feat_256: (B, 64, H/2,  W/2 )  encoder 256×256 浅层
        feat_128: (B, 64, H/4,  W/4 )  encoder 128×128 浅层
        feat_64 : (B, 64, H/8,  W/8 )  encoder  64×64  浅层

    输出 (与现有 hook 消费顺序一致):
        dict {
            'up_block_0': (B, 1280, H/16, W/16)  注入 SD2 UNet up_blocks[0]
            'up_block_1': (B, 640,  H/32, W/32)  注入 SD2 UNet up_blocks[1]
            'up_block_2': (B, 320,  H/64, W/64)  注入 SD2 UNet up_blocks[2]
        }

    注: SD2 UNet 实际有 4 个 up_blocks, 但最后一个 (up_block_3) 接收的 skip
    来自 up_block_2 输出而非 down_block, 因此 decoder skip 只针对前 3 个.

    全部 init=0 → 训练第一步输出严格为 0 → 等价于无 decoder skip.
    """

    def __init__(self, in_ch: int = 64, alpha_init: float = 0.0):
        super().__init__()
        self.in_ch = in_ch

        # 三路独立分支 (alpha/proj/zero_conv 全部独立参数)
        # 分支命名: branch_0/1/2, 对应 feat_64/128/256 → up_block_0/1/2
        # 这样 named_parameters() 的参数名以 .alpha 结尾, 与 train_controlnet.py
        # 中 endswith('.alpha') 的 alpha 参数收集逻辑天然兼容
        self.branch_0 = _DecoderBranch(in_ch=in_ch, out_ch=1280, alpha_init=alpha_init)  # feat_64  -> up_block_0 (16x16, 1280ch)
        self.branch_1 = _DecoderBranch(in_ch=in_ch, out_ch=640,  alpha_init=alpha_init)  # feat_128 -> up_block_1 (32x32,  640ch)
        self.branch_2 = _DecoderBranch(in_ch=in_ch, out_ch=320,  alpha_init=alpha_init)  # feat_256 -> up_block_2 (64x64,  320ch)

    def forward(self, feat_256: torch.Tensor, feat_128: torch.Tensor,
                feat_64: torch.Tensor) -> dict:
        """前向计算 decoder skip 残差 (三路独立 → 三路输出).

        Args:
            feat_256: (B, 64, H/2,  W/2 )
            feat_128: (B, 64, H/4,  W/4 )
            feat_64 : (B, 64, H/8,  W/8 )
        Returns:
            dict with 3 个 decoder skip 张量 (dtype 与模块权重一致)
        """
        # dtype 策略: 三路输入 dtype 必须一致; 模块权重 dtype 与输入 dtype 一致即可.
        # 无需手动 .float() / .to() 转换, 避免与 bf16 模块冲突.
        # ZeroConv init=0 保证输出严格为 0, 不引入数值精度问题.
        up0 = self.branch_0(feat_64)     # (B, 1280, H/16, W/16)
        up1 = self.branch_1(feat_128)    # (B, 640,  H/32, W/32)
        up2 = self.branch_2(feat_256)    # (B, 320,  H/64, W/64)

        return {
            'up_block_0': up0,
            'up_block_1': up1,
            'up_block_2': up2,
        }


def attach_decoder_skip_hooks(unet, decoder_skip_fn, target_blocks=(0, 1, 2)):
    """为 SD2 UNet 的 up_blocks 注册 forward pre-hook, 注入 decoder skip 残差.

    Args:
        unet: SD2 UNet (frozen)
        decoder_skip_fn: callable, 接收 hidden_states (B, C, H, W), 返回 dict
                        {'up_block_0': Tensor, 'up_block_1': Tensor, 'up_block_2': Tensor}
                        每次 forward 调用时计算
        target_blocks: 要注入的 up_block 索引, 默认 (0, 1, 2)

    Returns:
        list of hooks (用于 detach)
    """
    hooks = []

    for block_idx in target_blocks:
        up_block = unet.up_blocks[block_idx]

        def make_hook(idx):
            def hook(module, args):
                # args[0]: hidden_states (上采样结果)
                # args[1]: res_hidden_states_tuple (down_block skip connections)
                if len(args) < 2:
                    return args
                hidden_states = args[0]
                res_tuple = args[1]
                if not isinstance(res_tuple, (tuple, list)) or len(res_tuple) == 0:
                    return args

                # 计算 decoder skip 残差
                skip_dict = decoder_skip_fn(hidden_states)
                decoder_skip = skip_dict.get(f'up_block_{idx}', None)
                if decoder_skip is None:
                    return args

                # 检查形状匹配: decoder_skip 应与 res_tuple 中每个元素形状一致
                target_shape = res_tuple[0].shape
                if decoder_skip.shape != target_shape:
                    return args  # 形状不匹配, 跳过注入

                # 叠加到所有 resnet 用的 skip (SD2 up_block 内多 resnet 共享 decoder skip)
                # 与 UNet 原生 skip 特征逐元素相加 (禁止通道拼接)
                new_res_tuple = tuple(s + decoder_skip for s in res_tuple)
                return (hidden_states, new_res_tuple) + args[2:]
            return hook

        h = up_block.register_forward_pre_hook(make_hook(block_idx))
        hooks.append(h)

    return hooks


def detach_decoder_skip_hooks(hooks):
    """移除之前注册的 hooks."""
    for h in hooks:
        h.remove()