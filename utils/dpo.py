"""
DPO (Direct Preference Optimization) 工具
==========================================

参考: Diffusion-DPO (Wallace et al., 2023, arXiv:2311.12908)
       整合到 ControlNet 训练框架

提供:
    - build_ref_controlnet:  深拷贝当前 controlnet 作为冻结参考
    - compute_dpo_loss:      计算 Diffusion DPO 损失
    - DPOScheduler:          控制 ref_controlnet 的 EMA / sync 行为 (可选)
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn.functional as F


# ============================================================
# 1. 参考模型构建
# ============================================================
def build_ref_controlnet(controlnet) -> torch.nn.Module:
    """
    深拷贝 controlnet 作为 DPO 的参考策略 ref_controlnet.

    要求:
        - 完全冻结 (requires_grad = False)
        - 处于 eval 模式
        - 不进入 accelerator.prepare / optimizer

    训练时把 ref_controlnet 放到与 controlnet 相同的 device / dtype.
    """
    ref = copy.deepcopy(controlnet)
    ref.requires_grad_(False)
    ref.eval()
    for p in ref.parameters():
        p.detach_()
    return ref


@torch.no_grad()
def ema_update_ref_controlnet(
    controlnet,
    ref_controlnet,
    decay: float = 0.999,
):
    """
    可选: 用 EMA 把 policy controlnet 的权重复制到 ref_controlnet.
    注意: 原版 Diffusion-DPO 不使用 EMA, 因为 ref 与 policy 起点完全相同.
    该函数保留为扩展入口, 默认未启用.
    """
    for p, p_ref in zip(controlnet.parameters(), ref_controlnet.parameters()):
        p_ref.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# ============================================================
# 2. Diffusion DPO 损失
# ============================================================
def compute_dpo_loss(
    model_pred: torch.Tensor,
    ref_pred: torch.Tensor,
    target: torch.Tensor,
    beta_dpo: float = 5000.0,
):
    """
    Diffusion DPO 损失 (Wallace et al., 2023).

    输入形状约定 (与 SFT 保持一致):
        model_pred / ref_pred / target: [2*B, C, H, L]
        - 前 B 维是 winner (y_w) 的噪声预测
        - 后 B 维是 loser  (y_l) 的噪声预测
        B 来自 per-device batch_size; 2*B 来自 winner/loser 沿 batch 维 cat.

    关键不变量 (调用方必须保证):
        1. winner / loser 共享同一 timestep / noise
        2. winner / loser 共享同一 prompt (input_ids)
        3. winner / loser 共享同一 LQ (controlnet_cond)
        4. ref_pred 在 torch.no_grad() 内计算

    返回:
        loss:           scalar tensor (可 backward)
        implicit_acc:   scalar tensor (训练监控: 0~1)
        inside_term:    [B] tensor (可用于更细粒度分析)
    """
    # per-sample MSE (B 维以外的维度都求均值)
    model_losses = (model_pred - target).pow(2).mean(dim=list(range(1, model_pred.ndim)))
    # chunk(2) 沿 batch 维切分 -> 各 [B]
    model_losses_w, model_losses_l = model_losses.chunk(2, dim=0)
    model_diff = model_losses_w - model_losses_l

    with torch.no_grad():
        ref_losses = (ref_pred - target).pow(2).mean(dim=list(range(1, ref_pred.ndim)))
        ref_losses_w, ref_losses_l = ref_losses.chunk(2, dim=0)
        ref_diff = ref_losses_w - ref_losses_l

    inside_term = -0.5 * beta_dpo * (model_diff - ref_diff)
    implicit_acc = (inside_term > 0).float().mean()
    loss = -F.logsigmoid(inside_term).mean()

    return loss, implicit_acc, inside_term


# ============================================================
# 3. 工具: 把 (winner, loser) 沿 batch 维 cat, 强制 timestep / noise 对齐
# ============================================================
def align_timesteps_noise(
    bsz: int,
    noise_scheduler,
    device: torch.device,
):
    """
    为 DPO 准备 (2*B,) 的 timestep 与 noise, 保证 winner/loser 共享.

    返回:
        timesteps: [2*B] long
        noise:     [2*B, C, H, W]
    """
    bsz2 = bsz * 2
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz2,), device=device
    ).long()
    timesteps = timesteps.chunk(2, dim=0)[0].repeat(2)

    return timesteps


def repeat_timesteps(timesteps: torch.Tensor) -> torch.Tensor:
    """把 [B] 的 timestep 复制为 [2B], 与 SFT 时 [B] 噪声 cat 后结构一致."""
    return timesteps.chunk(2, dim=0)[0].repeat(2)


def repeat_noise(noise: torch.Tensor) -> torch.Tensor:
    """把 [B, ...] 的 noise 复制为 [2B, ...]."""
    return noise.chunk(2, dim=0)[0].repeat(2, *([1] * (noise.ndim - 1)))
