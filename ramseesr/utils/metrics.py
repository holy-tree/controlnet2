"""
图像质量评估指标 (PSNR / SSIM)
================================

使用 PyTorch 实现, 支持 batch 输入:
    输入: tensor [B, 3, H, W] 或 [3, H, W], 范围 [0, 1]
    输出: float (标量均值)
"""

import torch
import torch.nn.functional as F


def _to_4d(x: torch.Tensor) -> torch.Tensor:
    """[3,H,W] -> [1,3,H,W]"""
    if x.ndim == 3:
        return x.unsqueeze(0)
    return x


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    计算 PSNR (Peak Signal-to-Noise Ratio)。
    pred / target: [B, 3, H, W] 或 [3, H, W], 范围 [0, 1]
    返回: float (dB)
    """
    pred = _to_4d(pred).detach().float()
    target = _to_4d(target).detach().float()

    mse = F.mse_loss(pred, target, reduction="mean").item()
    if mse <= 1e-12:
        return 100.0  # 完全相同
    return 20.0 * torch.log10(torch.tensor(max_val)).item() - 10.0 * torch.log10(torch.tensor(mse)).item()


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    """生成 1D 高斯核"""
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _create_window(window_size: int, channels: int, device, dtype) -> torch.Tensor:
    """生成 2D 高斯窗口 [channels, 1, ws, ws]"""
    _1d = _gaussian_window(window_size, 1.5, device, dtype).unsqueeze(1)
    _2d = _1d @ _1d.t()
    window = _2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """
    计算 SSIM (Structural Similarity Index)。
    pred / target: [B, 3, H, W] 或 [3, H, W], 范围 [0, 1]
    返回: float (0~1, 越大越好)
    """
    pred = _to_4d(pred).detach().float()
    target = _to_4d(target).detach().float()

    B, C, H, W = pred.shape
    window = _create_window(window_size, C, pred.device, pred.dtype)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def evaluate_batch(pred_list, gt_list):
    """
    对一组 (pred, gt) 对计算平均 PSNR / SSIM。
    pred_list, gt_list: list of tensors in [0, 1]
    返回: (avg_psnr, avg_ssim)
    """
    psnrs, ssims = [], []
    for p, g in zip(pred_list, gt_list):
        psnrs.append(psnr(p, g))
        ssims.append(ssim(p, g))
    if not psnrs:
        return 0.0, 0.0
    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)