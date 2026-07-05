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


# ============================================================
# LPIPS (Learned Perceptual Image Patch Similarity)
# ============================================================
# LPIPS 需要预训练 backbone, 第一次调用时会自动下载权重 (~50MB)
#   - 'alex': 基于 AlexNet, 更快
#   - 'vgg':  基于 VGG, 更准但更慢
_LPIPS_MODEL = None
_LPIPS_NET = None
_LPIPS_DEVICE = None


def _get_lpips_model(net: str = "alex", device=None):
    """
    懒加载 LPIPS 模型 (避免每次调用都重新加载权重)。
    device: 目标设备 (None 则保持 CPU, 等首次调用时再迁移)
    """
    global _LPIPS_MODEL, _LPIPS_NET, _LPIPS_DEVICE
    if _LPIPS_MODEL is None or _LPIPS_NET != net:
        import lpips as lpips_pkg
        _LPIPS_MODEL = lpips_pkg.LPIPS(net=net, verbose=False)
        _LPIPS_MODEL.eval()
        _LPIPS_NET = net
        _LPIPS_DEVICE = None  # 强制首次迁移
    # 必要时把模型迁移到目标 device
    if device is not None and _LPIPS_DEVICE != device:
        _LPIPS_MODEL = _LPIPS_MODEL.to(device)
        _LPIPS_DEVICE = device
    return _LPIPS_MODEL


def lpips(pred: torch.Tensor, target: torch.Tensor, net: str = "alex") -> float:
    """
    计算 LPIPS (Learned Perceptual Image Patch Similarity)。
    pred / target: [B, 3, H, W] 或 [3, H, W], 范围 [0, 1]
    返回: float (越小越好, 0 表示完全相同)

    依赖: pip install lpips
    注意: 第一次调用时会下载预训练权重到 ~/.cache/torch/hub/checkpoints/
          模型会自动迁移到与输入 tensor 相同的 device
    """
    pred = _to_4d(pred).detach().float()
    target = _to_4d(target).detach().float()

    # 让模型跟随输入 tensor 的 device (避免 cuda/cpu 不一致)
    model = _get_lpips_model(net, device=pred.device)

    # LPIPS 内部将 [0,1] 映射到 [-1,1]
    pred = pred * 2.0 - 1.0
    target = target * 2.0 - 1.0

    with torch.no_grad():
        d = model(pred, target)
    return d.mean().item()


# ============================================================
# FID (Frechet Inception Distance)
# ============================================================
# FID 通过预训练 InceptionV3 提取 2048 维特征, 计算两组图像的 Fréchet 距离.
# 越低越好, 表示生成/恢复图像分布越接近真实图像分布.
# 依赖: torchvision (提供 Inception V3 weights)
#       第一次运行会下载 InceptionV3 权重 (~100MB) 到 ~/.cache/torch/hub/checkpoints/
_INCEPTION_MODEL = None
_INCEPTION_DEVICE = None


def _get_inception_model(device=None):
    """懒加载 InceptionV3 (aux_logits=True), 输出 2048 维特征."""
    global _INCEPTION_MODEL, _INCEPTION_DEVICE
    if _INCEPTION_MODEL is None:
        from torchvision.models import inception_v3, Inception_V3_Weights
        _INCEPTION_MODEL = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        _INCEPTION_MODEL.fc = torch.nn.Identity()  # 移除分类头, 输出 2048 维特征
        _INCEPTION_MODEL.eval()
        _INCEPTION_DEVICE = None
    if device is not None and _INCEPTION_DEVICE != device:
        _INCEPTION_MODEL = _INCEPTION_MODEL.to(device)
        _INCEPTION_DEVICE = device
    return _INCEPTION_MODEL


def _inception_features(images: torch.Tensor) -> torch.Tensor:
    """
    提取 InceptionV3 特征.
    images: [N, 3, H, W], 范围 [0, 1]
    返回: [N, 2048] 特征向量
    """
    model = _get_inception_model(device=images.device)
    # InceptionV3 要求 299x299 输入
    x = torch.nn.functional.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    # InceptionV3 预训练权重使用 ImageNet 标准化
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std

    feats = []
    with torch.no_grad():
        for i in range(0, x.size(0), 32):
            batch = x[i:i + 32]
            f = model(batch)
            # aux_logits=True 训练时返回 tuple, eval 时返回 tensor, 但 fc 已替换为 Identity
            if isinstance(f, tuple):
                f = f[0]
            feats.append(f)
    return torch.cat(feats, dim=0)


def fid(pred_list, gt_list) -> float:
    """
    计算 FID (Frechet Inception Distance).
    pred_list, gt_list: list of tensors in [0, 1] (任意尺寸均可, 内部 resize 到 299x299)
    返回: float (越小越好, 表示两组图像分布越接近)

    算法:
        1. 用 InceptionV3 提取两组图像的 2048 维特征
        2. 分别计算均值 mu 与协方差 sigma
        3. FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))
    """
    if not pred_list or not gt_list:
        return float("nan")

    pred_t = torch.stack([(_to_4d(p) if isinstance(p, torch.Tensor) else _to_4d(torch.as_tensor(p))).squeeze(0) for p in pred_list]).float()
    gt_t = torch.stack([(_to_4d(g) if isinstance(g, torch.Tensor) else _to_4d(torch.as_tensor(g))).squeeze(0) for g in gt_list]).float()

    pred_feats = _inception_features(pred_t).cpu().double().numpy()
    gt_feats = _inception_features(gt_t).cpu().double().numpy()

    # 计算 FID
    import numpy as np
    from scipy import linalg

    mu1, sigma1 = pred_feats.mean(axis=0), np.cov(pred_feats, rowvar=False)
    mu2, sigma2 = gt_feats.mean(axis=0), np.cov(gt_feats, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_val = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid_val)