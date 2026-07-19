"""
ARCA: Adaptive Residual Calibration Adapter (自适应残差校准适配器)
==================================================================
针对 ControlNet 控制残差幅值爆炸、传统 Adapter 跨通道放大、多层特征尺度
不匹配三大问题, 设计 ARCA 替换 LightweightAdapter:

  c2f_feat
      |
  GroupNorm(in_ch)           # 约束 C2F 输出分布
      |
  Depthwise 3x3 Conv          # 局部纹理 (雨丝/雾), 无跨通道幅值放大
      |
  GELU
      |
  1x1 Conv (in_ch -> out_ch)  # 通道融合, 无空间膨胀
      |
  GroupNorm(out_ch)           # 压缩卷积输出波动
      |
  ZeroConv (1x1, 0-init)      # 训练初期整体输出 0
      |
  tanh(alpha) * residual      # 每 stage 独立可学习缩放, 范围 [-1, 1]
      |
  residual (返回, 后续与 UNet 隐特征相加)

每个 ARCA 实例:
  - 持有 1 个 zero_conv (1x1 Conv2d, 0-init)
  - 持有 1 个可学习标量 alpha (init=0, 训练后被 tanh 限到 [-1, 1])

训练约束:
  - alpha 初始 0, 配合 zero_conv 0 初始化, 训练初期残差为 0, 不破坏 SD 预训练
  - 不引入额外 loss / 正则, 仅依靠 LN + DWConv + 分层 alpha 三层保险
  - 监控指标 (训练时打印):
      alpha_i, std(res_i), std(unet_hidden_i), R = std(res) / std(unet_hidden)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 局部工具: 1x1 zero-initialized Conv2d (沿用 c2f_block.zero_conv 约定)
# ----------------------------------------------------------------------------
def _zero_conv(in_channels: int, out_channels: int) -> nn.Conv2d:
    layer = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=True)
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


# ----------------------------------------------------------------------------
# 单个 ARCA stage
# ----------------------------------------------------------------------------
class ARCAResidualCalibrator(nn.Module):
    """
    自适应残差校准适配器 (单 stage).

    通道规则:
      in_channels  : 输入特征通道 (来自 C2F 主干)
      out_channels : 输出 (送入 zero_conv) 通道
                     通常 = in_channels (stage 0/1/2/3)
                     或 != in_channels (down 投影: 640 -> 320, 1280 -> 640)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 ln_groups_max: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1) 第一层 LN (约束 C2F 输出分布)
        self.ln1 = nn.GroupNorm(
            num_groups=min(ln_groups_max, in_channels),
            num_channels=in_channels,
        )

        # 2) Depthwise 3x3 (单通道空间特征, 无跨通道幅值放大)
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            bias=True, groups=in_channels,
        )

        # 3) 1x1 Pointwise 通道融合
        self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

        # 4) 第二层 LN (压缩卷积输出波动)
        self.ln2 = nn.GroupNorm(
            num_groups=min(ln_groups_max, out_channels),
            num_channels=out_channels,
        )

        # 5) ZeroConv (1x1, 0-init)
        self.zero_conv = _zero_conv(out_channels, out_channels)

        # 6) 分层独立可学习缩放 alpha.
        # 论文方案: 初始值 0.1 (而非 0). 原因:
        #   - alpha=0 + zero_conv 0-init 会形成死锁 (residual=0, d_residual/d_alpha=0, 永远不学)
        #   - 初始 0.1 配合 tanh 限到 [-1, 1], 残差幅度受 zero_conv 权重二次约束, 不会污染 SD 预训练
        #   - 但 alpha 梯度链路立刻打通, 训练能持续学
        self.alpha = nn.Parameter(torch.full((1,), 0.1), requires_grad=True)

        # 7) 分层独立可学习门控 gate (直接乘子, 无 sigmoid 包裹).
        # 作用: 在 alpha 之外再叠一层可学习缩放, 用于把 down_arca 等 R>1 的位置压回 0.3~0.5.
        # 初始值 1.0 → residual = 1.0 * scale * h, 与 gate 引入前完全相同, 不破坏已有模型.
        # 设计原因: sigmoid gate 在 init=4.0 时梯度只有 0.018 (sigmoid saturation),
        #          在 init=1.0 时初始行为又破坏模型. 直接乘子避免两个问题:
        #          - 初始行为 100% 等价于原模型 (接续训练无冲击)
        #          - 梯度永远是 gate*grad_loss, 不会饱和
        #          - 学习目标明确: gate 降从 1.0 → ~0.3 (down_arca)
        # 配合 train_controlnet.py 中 gate 单独 param group (lr=5e-4, weight_decay=0),
        # 期望 50k 内 gate 从 1.0 降到 0.5~0.7 (down_arca 进一步降到 0.3).
        self.gate = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # 残差 cache: 训练时序监控需要 std(res), 缓存在 self._last_residual_stats
        # 仅在 self.training 模式下填充, 避免污染推理路径
        self._last_residual_stats: dict | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h = self.dwconv(h)
        h = F.gelu(h)
        h = self.pwconv(h)
        h = self.ln2(h)
        h = self.zero_conv(h)
        scale = torch.tanh(self.alpha)
        gate = self.gate                              # 直接乘子, 不饱和, init=1.0 完全等价于原模型
        residual = gate * scale * h
        if self.training:
            with torch.no_grad():
                self._last_residual_stats = {
                    "abs_mean": residual.detach().float().abs().mean().item(),
                    "std": residual.detach().float().std().item(),
                    "scale": scale.detach().item(),
                    "gate": gate.detach().item(),
                    "effective_scale": (gate * scale).detach().item(),
                }
        return residual

    def extra_repr(self) -> str:
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"alpha_init={float(self.alpha.detach().item()):.4f}")


# ----------------------------------------------------------------------------
# 旧 LightweightAdapter 兼容别名 (旧 checkpoint / 论文消融 baseline 用)
# ----------------------------------------------------------------------------
class LightweightAdapter(nn.Module):
    """兼容旧 baseline: 3x3 Conv + GroupNorm + GELU (无 alpha 限幅)."""

    def __init__(self, channels: int, hidden_ratio: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


# ----------------------------------------------------------------------------
# 训练监控: 把每个 ARCA 的 alpha + 残差 std 整理成可打印表格
# ----------------------------------------------------------------------------
@torch.no_grad()
def collect_arca_monitor(model: nn.Module) -> list[dict]:
    """
    遍历 model 中所有 ARCAResidualCalibrator, 收集 alpha + (可选) residual std.
    返回 list[dict], 顺序按 Module 注册顺序稳定.
    """
    records: list[dict] = []
    for name, mod in model.named_modules():
        if isinstance(mod, ARCAResidualCalibrator):
            r = {
                "name": name,
                "alpha": float(mod.alpha.detach().item()),
                "tanh_alpha": float(torch.tanh(mod.alpha).detach().item()),
                "gate_raw": float(mod.gate.detach().item()),
                "gate": float(torch.sigmoid(mod.gate).detach().item()),
            }
            if mod._last_residual_stats is not None:
                r.update(mod._last_residual_stats)
            records.append(r)
    return records


def format_arca_monitor(records: list[dict], unet_hidden_stds: list[float] | None = None
                        ) -> str:
    """
    渲染监控表:
      header:  stage | alpha | tanh(alpha) | scale | gate | eff_scale | res_std | unet_std | R
    unet_hidden_stds: 与 records 等长的列表, 给出同位置 UNet 隐特征 std.
                      若为 None 或长度不匹配, unet_std / R 列显示 N/A.
                      论文 method 章节做消融时由调用方 hook UNet 提供.
    """
    has_unet = (unet_hidden_stds is not None and len(unet_hidden_stds) == len(records))
    lines = ["# ARCA 监控"]
    if has_unet:
        lines.append(
            f"{'name':<42} {'alpha':>10} {'tanh':>8} {'scale':>8} {'gate':>6} "
            f"{'eff_s':>7} {'res_std':>11} {'unet_std':>11} {'R':>9}"
        )
    else:
        lines.append(
            f"{'name':<42} {'alpha':>10} {'tanh':>8} {'scale':>8} {'gate':>6} "
            f"{'eff_s':>7} {'res_std':>11}  (unet_std/R 需 hook UNet 提供)"
        )
    lines.append("-" * 110)
    for i, r in enumerate(records):
        scale = r.get("scale", float("nan"))
        gate = r.get("gate", float("nan"))
        eff = r.get("effective_scale", scale * gate if (scale == scale and gate == gate) else float("nan"))
        res_std = r.get("std", float("nan"))
        if has_unet:
            unet_std = unet_hidden_stds[i]
            r_ratio = (res_std / unet_std) if (unet_std and unet_std > 0) else float("nan")
            lines.append(
                f"{r['name']:<42} {r['alpha']:>10.4f} {r['tanh_alpha']:>8.4f} {scale:>8.4f} "
                f"{gate:>6.3f} {eff:>7.4f} "
                f"{res_std:>11.4e} {unet_std:>11.4e} {r_ratio:>9.4f}"
            )
        else:
            lines.append(
                f"{r['name']:<42} {r['alpha']:>10.4f} {r['tanh_alpha']:>8.4f} {scale:>8.4f} "
                f"{gate:>6.3f} {eff:>7.4f} {res_std:>11.4e}    N/A                  N/A"
            )
    return "\n".join(lines)
