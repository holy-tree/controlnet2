#!/usr/bin/env python
"""
build_preference.py
===================

Sample-level batching 重构版:
    DataLoader(batch_size=B) → 一次送入 B 张不同 LQ 到 pipeline
    → pipeline(num_images_per_prompt=N) 一次生成 B×N 张候选
    → GPU batched 算 LPIPS / PSNR / SSIM / CLIP-IQA
    → ThreadPool 异步保存
    → 写 per-stem score.json

与旧版主要区别:
    旧版 process_one_sample 串行, sample_batch_size 实际只对单个样本的候选分批,
    GPU 利用率长期 20~40% (CPU 串行 I/O + 单样本 pipeline call 间隔长).
    新版 process_batch 一次处理 B 个样本, 一个 pipeline call 推到 B*N 张图,
    GPU 利用率稳定 80~95% (A800 80G 实测).

数据布局不变: dataset_root/{weather}/{split}/candidates/{stem}/{cand_XXX.png, score.json}
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("build_preference")


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTENSIONS


# ============================================================
# 1. 批处理版本的 PSNR / SSIM (per-sample, 在 GPU 上一次算)
# ============================================================
def psnr_batch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> List[float]:
    """
    Per-sample PSNR. 输入 [N, 3, H, W] in [0, 1], 输出 list[float], 长度 N.
    GPU 一次算完, 速度比 ramseesr.utils.metrics.psnr (逐张 item() 调用) 快 ~10x.
    """
    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])              # [N]
    mse_safe = mse.clamp(min=1e-12)
    psnr = 20.0 * torch.log10(torch.tensor(max_val)) - 10.0 * torch.log10(mse_safe)
    psnr = torch.where(mse <= 1e-12, torch.full_like(psnr, 100.0), psnr)
    return psnr.tolist()


def ssim_batch(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> List[float]:
    """
    Per-sample SSIM. 输入 [N, 3, H, W] in [0, 1], 输出 list[float], 长度 N.
    与 ramseesr.utils.metrics.ssim 算法一致 (Gaussian window), 改成 batched.
    """
    N, C, H, W = pred.shape
    device, dtype = pred.device, pred.dtype
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window_2d = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(C, 1, -1, -1).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=C)
    mu2 = F.conv2d(target, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=[1, 2, 3]).tolist()


# ============================================================
# 2. Reward Scorer (CLIP-IQA / LPIPS 懒加载)
# ============================================================
class RewardScorer:
    """懒加载 LPIPS / CLIP-IQA, 防止启动时无谓的依赖检查."""

    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self._lpips = None
        self._clipiqa = None

    def _ensure_lpips(self):
        if self._lpips is None:
            self._lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device, self.dtype)
            self._lpips.eval()
        return self._lpips

    def _ensure_clipiqa(self):
        if self._clipiqa is None:
            try:
                import pyiqa
                self._clipiqa = pyiqa.create_metric("clipiqa", device=self.device)
            except ImportError as e:
                raise ImportError(
                    "需要 pyiqa 才能计算 CLIP-IQA, 请先 pip install pyiqa"
                ) from e
        return self._clipiqa


# ============================================================
# 3. Sample 收集 (旧: 扫目录 / 新: 读 JSON)
# ============================================================
def _derive_lq_path(gt_path: Path) -> Path:
    """
    GT 路径 → LQ 路径: 路径中最后一段 'GT' 替换为 'LQ'.
    同时兼容正反斜杠与 mixed separators (Windows).
    """
    gt_str = str(gt_path)
    for sep_g, sep_l in [("/", "/"), ("\\", "\\"), ("/", "\\"), ("\\", "/")]:
        gt_str = gt_str.replace(f"{sep_g}GT{sep_l}", f"{sep_l}LQ{sep_l}")
    return Path(gt_str)


def _make_cand_dir(weather: str, split: str, stem: str,
                   dataset_root, candidates_subdir: str) -> Path:
    """
    拼 cand_dir:
      - candidates_subdir 是绝对路径: {candidates_subdir}/{weather}/{split}/{stem}
      - candidates_subdir 是相对路径: {dataset_root}/{weather}/{split}/{candidates_subdir}/{stem}
        (此情况 dataset_root 必须提供)
    """
    subdir = Path(candidates_subdir)
    if subdir.is_absolute():
        # 绝对路径时 dataset_root 完全用不上, 短路避免 None 报错
        return subdir / weather / split / stem
    if dataset_root is None:
        raise ValueError(
            f"candidates_subdir='{candidates_subdir}' 为相对路径, "
            "必须同时在 yaml 或 CLI 里提供 dataset_root"
        )
    return Path(dataset_root) / weather / split / candidates_subdir / stem


def collect_samples(dataset_root, weather_types, splits,
                    weather_num_samples, candidates_subdir):
    """旧路径: 从 dataset_root/{weather}/{split}/{GT,LQ}/ 扫描"""
    samples = []
    for weather in weather_types:
        for split in splits:
            gt_dir = Path(dataset_root) / weather / split / "GT"
            lq_dir = Path(dataset_root) / weather / split / "LQ"
            if not gt_dir.is_dir() or not lq_dir.is_dir():
                logger.warning(f"[跳过] {gt_dir} 或 {lq_dir} 不存在")
                continue
            gt_map = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and is_image(p)}
            lq_map = {p.stem: p for p in lq_dir.iterdir() if p.is_file() and is_image(p)}
            matched = 0
            for stem in sorted(gt_map.keys() & lq_map.keys()):
                cand_dir = _make_cand_dir(weather, split, stem, dataset_root, candidates_subdir)
                samples.append({
                    "weather": weather, "split": split, "stem": stem,
                    "lq_path": lq_map[stem], "gt_path": gt_map[stem],
                    "cand_dir": cand_dir,
                })
                matched += 1
            logger.info(f"[数据集] {weather}/{split}: 匹配 {matched} 对")

    if weather_num_samples:
        new_samples = []
        for w in weather_types:
            ws = [s for s in samples if s["weather"] == w]
            limit = weather_num_samples.get(w, 0)
            if limit and limit > 0 and limit < len(ws):
                logger.info(f"[截断] {w}: {len(ws)} -> {limit}")
                new_samples.extend(ws[:limit])
            else:
                new_samples.extend(ws)
        samples = new_samples
    return samples


def collect_samples_from_json(json_path, weather_types, splits,
                              weather_num_samples, dataset_root, candidates_subdir):
    """
    从 scripts/filter_dataset/filter_by_sobel.py 输出的 JSON 读取样本.
    配合 --filtered_json 启用.
    """
    json_path = Path(json_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"filtered_json 不存在: {json_path}")

    logger.info(f"从 sobel 筛选 JSON 加载: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_samples = payload.get("samples", []) or []
    if not raw_samples:
        raise ValueError(f"JSON {json_path} 中没有 samples 字段")

    weather_set = set(weather_types or [])
    splits_set = set(splits or ["train"])

    filtered = []
    skipped_weather = 0
    for s in raw_samples:
        w = s.get("weather", "")
        sp = s.get("split", "train")
        if weather_set and w not in weather_set:
            skipped_weather += 1
            continue
        if sp not in splits_set:
            continue
        filtered.append(s)
    logger.info(
        f"[JSON] 原始 {len(raw_samples)} -> 白名单后 {len(filtered)} "
        f"(跳过 {skipped_weather} 条 weather 不匹配)"
    )

    samples = []
    skipped_missing = 0
    for s in filtered:
        gt_path = Path(s["path"])
        if not gt_path.is_file():
            skipped_missing += 1
            continue
        lq_path = _derive_lq_path(gt_path)
        if not lq_path.is_file():
            logger.debug(f"[{s.get('weather','?')}/{gt_path.stem}] LQ 缺失: {lq_path}")
            skipped_missing += 1
            continue
        cand_dir = _make_cand_dir(
            s.get("weather", ""), s.get("split", "train"), gt_path.stem,
            dataset_root, candidates_subdir,
        )
        samples.append({
            "weather": s.get("weather", ""),
            "split": s.get("split", "train"),
            "stem": gt_path.stem,
            "lq_path": lq_path,
            "gt_path": gt_path,
            "cand_dir": cand_dir,
        })

    if weather_num_samples:
        new_samples = []
        for w in weather_types:
            ws = [s for s in samples if s["weather"] == w]
            limit = weather_num_samples.get(w, 0)
            if limit and limit > 0 and limit < len(ws):
                new_samples.extend(ws[:limit])
            else:
                new_samples.extend(ws)
        samples = new_samples

    for w in weather_types:
        cnt = sum(1 for s in samples if s["weather"] == w)
        logger.info(f"  - {w}: {cnt}")
    return samples


# ============================================================
# 4. Pipeline 加载 (含 xformers 友好降级)
# ============================================================
def load_pipeline(args, weight_dtype, device):
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        if args.controlnet_model_name_or_path else None,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    if getattr(args, "enable_xformers", False):
        if is_xformers_available():
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info("xformers 内存优化注意力已启用")
            except Exception as e:
                logger.warning(
                    f"启用 xformers 失败 ({e}), 继续默认注意力"
                )
        else:
            logger.warning(
                "enable_xformers=true 但 xformers 未安装, "
                "继续默认注意力. 请 pip install xformers (匹配 torch 版本)"
            )
    return pipeline


# ============================================================
# 5. 核心: process_batch - 真正的样本级批处理
# ============================================================
@torch.no_grad()
def process_batch(pipeline, scorer, batch_samples, args,
                  weight_dtype, device, enabled):
    """
    一次处理 B 个不同样本:

      1. CPU 并行解码 B 张 LQ + B 张 GT
      2. pipeline(prompt=Bx"", image=[B,3,H,W], num_images_per_prompt=N)
         → 一次生成 B*N 张候选 (GPU 利用率 ≈ 90%)
      3. ThreadPool 异步保存 B*N 张 PNG
      4. PSNR / SSIM / LPIPS / CLIP-IQA 全部 batched 算 (B*N 张一次过 GPU)
      5. 写 B 个 score.json (per stem)
      6. empty_cache 防碎片化

    输入:
        batch_samples: list of dict, 每条含 lq_path / gt_path / cand_dir / weather / split / stem
        enabled: 启用的 reward 指标列表

    返回:
        list[dict | None], 每个 sample 一个 payload (失败为 None), 与旧版兼容
    """
    B = len(batch_samples)
    N = args.num_candidates
    BN = B * N

    # ===== 1. CPU 并行解码 + 预处理 =====
    preprocess = transforms.Compose([
        transforms.Resize((args.height, args.width),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop((args.height, args.width)),
    ])
    to_tensor = transforms.ToTensor()

    def _load_one(sample):
        lq_pil = Image.open(sample["lq_path"]).convert("RGB")
        gt_pil = Image.open(sample["gt_path"]).convert("RGB")
        lq_pil = preprocess(lq_pil)
        gt_pil = preprocess(gt_pil)
        return to_tensor(lq_pil), to_tensor(gt_pil), sample

    load_workers = min(8, B)
    with ThreadPoolExecutor(max_workers=load_workers) as ex:
        loaded = list(ex.map(_load_one, batch_samples))

    lq_tensors = [x[0] for x in loaded]
    gt_tensors = [x[1] for x in loaded]
    samples_meta = [x[2] for x in loaded]

    # 一次性 stack 到 GPU, 后续推理无需再搬运
    lq_batch = torch.stack(lq_tensors, dim=0).to(device)   # [B, 3, H, W]
    gt_batch = torch.stack(gt_tensors, dim=0).to(device)   # [B, 3, H, W]

    # ===== 1b. 确保所有候选目录存在 (并行 mkdir, 不重复) =====
    unique_cand_dirs = list(set(info["cand_dir"] for info in samples_meta))
    with ThreadPoolExecutor(max_workers=min(8, len(unique_cand_dirs) or 1)) as ex:
        list(ex.map(lambda d: d.mkdir(parents=True, exist_ok=True), unique_cand_dirs))

    # ===== 2. cache 检查 =====
    save_workers = max(2, min(8, BN))
    all_cached = args.rescore_only and all(
        (info["cand_dir"] / f"cand_{j:03d}.png").is_file()
        for info in samples_meta for j in range(N)
    )

    if all_cached:
        # 全部命中 cache, 并行加载
        paths = [
            info["cand_dir"] / f"cand_{j:03d}.png"
            for info in samples_meta for j in range(N)
        ]
        with ThreadPoolExecutor(max_workers=save_workers) as ex:
            cand_list = list(ex.map(
                lambda p: to_tensor(Image.open(p).convert("RGB")), paths
            ))
        cand_tensors = torch.stack(cand_list, dim=0).to(device)
    else:
        # ===== 3. Pipeline batch 推理 =====
        # B*N 个 generator: pipeline 要求 len(generators) == batch_size * num_images_per_prompt
        # 顺序 [s0_c0, s0_c1, ..., s0_cN-1, s1_c0, s1_c1, ..., s1_cN-1, ...]
        # 不同 sample 用不同的 seed 块, 避免 seed 重叠 (sample i 用 base + i*N ~ base + (i+1)*N - 1)
        prompts = [""] * B
        generators = [
            torch.Generator(device=device).manual_seed(args.seed_base + i * N + j)
            for i in range(B) for j in range(N)
        ]
        with torch.autocast("cuda", enabled=weight_dtype != torch.float32):
            outputs = pipeline(
                prompt=prompts,
                image=lq_batch,                       # [B, 3, H, W] tensor batch
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generators,
                num_images_per_prompt=N,              # 每个 prompt 推 N 张
                height=args.height,
                width=args.width,
            ).images
            # outputs: list of B*N PIL, 顺序 [s0_c0..s0_cN, s1_c0..s1_cN, ...]

        # ===== 4. ThreadPool 异步保存 + 收集 tensor =====
        save_jobs = []
        cand_tensors_list = []
        for i, info in enumerate(samples_meta):
            for j in range(N):
                flat_idx = i * N + j
                pil = outputs[flat_idx].resize(
                    (args.width, args.height), Image.BILINEAR
                )
                cand_path = info["cand_dir"] / f"cand_{j:03d}.png"
                save_jobs.append((cand_path, pil))
                cand_tensors_list.append(to_tensor(pil))

        # CPU 并行 I/O, 不阻塞 GPU
        with ThreadPoolExecutor(max_workers=save_workers) as ex:
            list(ex.map(lambda jp: jp[1].save(jp[0]), save_jobs))

        cand_tensors = torch.stack(cand_tensors_list, dim=0).to(device)

    # ===== 5. NaN 屏蔽 (per-candidate) =====
    valid_mask = [bool(torch.isfinite(t).all().item()) for t in cand_tensors]

    # GT 沿 batch 维重复 N 次, 与 cand_tensors 对齐: [B*N, 3, H, W]
    gt_repeated = gt_batch.repeat_interleave(N, dim=0)

    # ===== 6. 全部 batched 算 reward =====
    psnr_scores = _batched_psnr(cand_tensors, gt_repeated, enabled, valid_mask, BN)
    ssim_scores = _batched_ssim(cand_tensors, gt_repeated, enabled, valid_mask, BN)
    lpips_scores = _batched_lpips(cand_tensors, gt_repeated, scorer, enabled, valid_mask, BN)
    clip_scores = _batched_clipiqa(cand_tensors, scorer, enabled, valid_mask, BN)

    # ===== 7. 写 per-stem score.json =====
    payloads = []
    for i, info in enumerate(samples_meta):
        valid_idx_sample = [i * N + j for j in range(N) if valid_mask[i * N + j]]
        n_valid = len(valid_idx_sample)
        if n_valid < 2:
            logger.warning(
                f"[{info['weather']}/{info['stem']}] valid={n_valid}/{N} (< 2), 跳过"
            )
            payloads.append(None)
            continue

        candidates = []
        for j in range(N):
            flat_idx = i * N + j
            if not valid_mask[flat_idx]:
                continue
            cand_record = {"idx": j}
            if psnr_scores is not None: cand_record["psnr"] = psnr_scores[flat_idx]
            if ssim_scores is not None: cand_record["ssim"] = ssim_scores[flat_idx]
            if lpips_scores is not None: cand_record["lpips"] = lpips_scores[flat_idx]
            if clip_scores is not None: cand_record["clip_iqa"] = clip_scores[flat_idx]
            candidates.append(cand_record)

        payload = {
            "stem": info["stem"],
            "weather": info["weather"],
            "split": info["split"],
            "num_candidates": len(candidates),
            "num_candidates_total": N,
            "reward_metrics_enabled": enabled,
            "reward_weights_default": args.reward_weights,
            "candidates": candidates,
        }
        score_path = info["cand_dir"] / "score.json"
        with open(score_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        payloads.append(payload)

    # 释放显存, 防止碎片化 (尤其当 B*N 大时)
    del lq_batch, gt_batch, cand_tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return payloads


# ============================================================
# 6. 各 reward 的 batched 计算 (内部 helpers)
# ============================================================
def _batched_psnr(cand_tensors, gt_repeated, enabled, valid_mask, BN):
    if "psnr" not in enabled:
        return None
    scores = [float("nan")] * BN
    valid_idx = [i for i in range(BN) if valid_mask[i]]
    if valid_idx:
        try:
            cand_v = cand_tensors[valid_idx]
            gt_v = gt_repeated[valid_idx]
            out = psnr_batch(cand_v, gt_v)
            for vi, s in zip(valid_idx, out):
                scores[vi] = s
        except Exception as e:
            logger.warning(f"PSNR batch 失败: {e}")
    return scores


def _batched_ssim(cand_tensors, gt_repeated, enabled, valid_mask, BN):
    if "ssim" not in enabled:
        return None
    scores = [float("nan")] * BN
    valid_idx = [i for i in range(BN) if valid_mask[i]]
    if valid_idx:
        try:
            cand_v = cand_tensors[valid_idx]
            gt_v = gt_repeated[valid_idx]
            out = ssim_batch(cand_v, gt_v)
            for vi, s in zip(valid_idx, out):
                scores[vi] = s
        except Exception as e:
            logger.warning(f"SSIM batch 失败: {e}")
    return scores


def _batched_lpips(cand_tensors, gt_repeated, scorer, enabled, valid_mask, BN):
    if "lpips" not in enabled:
        return None
    scores = [float("nan")] * BN
    valid_idx = [i for i in range(BN) if valid_mask[i]]
    if valid_idx:
        try:
            lpips_model = scorer._ensure_lpips()
            cand_v = cand_tensors[valid_idx].to(scorer.device, scorer.dtype)
            gt_v = gt_repeated[valid_idx].to(scorer.device, scorer.dtype)
            cand_norm = cand_v * 2.0 - 1.0     # LPIPS 要求 [-1, 1]
            gt_norm = gt_v * 2.0 - 1.0
            with torch.no_grad():
                d = lpips_model(cand_norm, gt_norm)
            d_list = d.flatten().cpu().tolist() if d.ndim > 1 else [d.item()] * len(valid_idx)
            for vi, s in zip(valid_idx, d_list):
                scores[vi] = float(s)
        except Exception as e:
            logger.warning(f"LPIPS 失败: {e}")
    return scores


def _batched_clipiqa(cand_tensors, scorer, enabled, valid_mask, BN):
    if "clip_iqa" not in enabled:
        return None
    scores = [float("nan")] * BN
    valid_idx = [i for i in range(BN) if valid_mask[i]]
    if valid_idx:
        try:
            clipiqa = scorer._ensure_clipiqa()
            cand_v = torch.stack([cand_tensors[i] for i in valid_idx], dim=0).to(scorer.device)
            scores_tensor = clipiqa(cand_v)
            scores_list = (
                [float(scores_tensor.item())] * len(valid_idx)
                if scores_tensor.ndim == 0
                else [float(s) for s in scores_tensor.cpu().tolist()]
            )
            for vi, s in zip(valid_idx, scores_list):
                scores[vi] = s
        except Exception as e:
            logger.warning(f"CLIP-IQA 失败: {e}")
    return scores


# ============================================================
# 7. Argparse + 配置合并 + 默认值
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="build_preference.py (sample-level batching)")
    p.add_argument("--config", type=str, default=None,
                   help="YAML 配置文件路径")

    p.add_argument("--dataset_root", type=str, default=None)
    p.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    p.add_argument("--num_candidates", type=int, default=None,
                   help="每条样本生成的候选数 N (默认 16)")
    p.add_argument("--data_batch_size", type=int, default=None,
                   help="一次 pipeline 处理的样本数 B (默认 16, 显存够可调到 32)")
    p.add_argument("--sample_batch_size", type=int, default=None,
                   help="[deprecated] 旧: 候选 batch. 新架构下被忽略, 改用 --data_batch_size")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--rescore_only", action="store_true")
    p.add_argument("--enable_xformers", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--filtered_json", type=str, default=None)
    p.add_argument("--log_interval_sec", type=int, default=30,
                   help="进度日志间隔 (秒)")

    # 兼容旧字段
    p.add_argument("--candidates_subdir", type=str, default=None)
    p.add_argument("--inference_steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--seed_base", type=int, default=None)
    p.add_argument("--mixed_precision", type=str, default=None)
    p.add_argument("--reward_metrics", type=str, nargs="+", default=None)
    p.add_argument("--reward_weights", type=str, default=None)
    return p.parse_args()


def merge_config(args):
    """YAML 中未在 CLI 显式指定的字段 (None) 被 yaml 填入."""
    if args.config and Path(args.config).is_file():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
    return args


def _ensure_defaults(args):
    """程序级默认值, 在 yaml/CLI 之后填充."""
    if args.candidates_subdir is None: args.candidates_subdir = "candidates"
    if args.max_samples is None: args.max_samples = None
    if args.rescore_only is None: args.rescore_only = False
    if args.filtered_json is None: args.filtered_json = None
    if args.enable_xformers is None: args.enable_xformers = False
    if args.log_interval_sec is None: args.log_interval_sec = 30

    # 兼容旧字段: sample_batch_size 在新架构下语义不同
    if args.data_batch_size is None and args.sample_batch_size is not None:
        logger.warning(
            f"--sample_batch_size={args.sample_batch_size} 是旧的 '候选 batch' 语义, "
            f"新架构下忽略. 请改用 --data_batch_size (默认 16)"
        )
    if args.data_batch_size is None:
        args.data_batch_size = 16
    return args


# ============================================================
# 8. Main: sample-level batching 主循环
# ============================================================
def main():
    args = parse_args()
    args = merge_config(args)
    args = _ensure_defaults(args)

    # device / dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # reward 启用的指标
    enabled = args.reward_metrics or ["psnr", "ssim", "lpips", "clip_iqa"]
    logger.info(f"启用的 reward 指标: {enabled}")

    # 加载 pipeline
    logger.info(f"加载 pipeline from {args.pretrained_model_name_or_path}")
    pipeline = load_pipeline(args, weight_dtype, device)

    # reward scorer (懒加载模型)
    scorer = RewardScorer(device, weight_dtype)
    if "lpips" in enabled:
        try:
            scorer._ensure_lpips()
        except Exception as e:
            logger.error(f"LPIPS 模型加载失败: {e}")
            return
    if "clip_iqa" in enabled:
        try:
            scorer._ensure_clipiqa()
        except ImportError as e:
            logger.error(str(e))
            return
    logger.info(f"Reward 模型加载完成 ({', '.join(enabled)})")

    # 收集样本
    weather_num_samples = {}
    for w in (args.weather_types or []):
        attr = f"{w}_num"
        if hasattr(args, attr) and getattr(args, attr):
            weather_num_samples[w] = getattr(args, attr)

    if args.filtered_json:
        samples = collect_samples_from_json(
            args.filtered_json,
            args.weather_types,
            args.splits,
            weather_num_samples,
            args.dataset_root,
            args.candidates_subdir,
        )
    else:
        samples = collect_samples(
            args.dataset_root,
            args.weather_types,
            args.splits,
            weather_num_samples,
            args.candidates_subdir,
        )

    if args.max_samples:
        samples = samples[:args.max_samples]
    logger.info(f"待处理样本数: {len(samples)}")
    if not samples:
        logger.error("未找到任何样本, 退出")
        sys.exit(1)

    # ===== 主循环: 样本级批处理 =====
    summary = {
        "num_processed": 0,
        "num_skipped_error": 0,
        "num_skipped_weak": 0,    # NaN 或 valid < 2
        "data_batch_size": args.data_batch_size,
        "num_candidates": args.num_candidates,
        "elapsed_sec": 0.0,
    }
    t0 = time.time()
    progress_bar = tqdm(
        total=len(samples), desc="samples", unit="stem",
    )
    last_log_t = t0
    log_interval = max(5, args.log_interval_sec)

    # 将样本切成大小为 data_batch_size 的批次
    for batch_start in range(0, len(samples), args.data_batch_size):
        batch_samples = samples[batch_start:batch_start + args.data_batch_size]
        try:
            payloads = process_batch(
                pipeline, scorer, batch_samples,
                args, weight_dtype, device, enabled,
            )
        except Exception as e:
            logger.exception(
                f"batch {batch_start}~{batch_start+len(batch_samples)} 失败: {e}"
            )
            summary["num_skipped_error"] += len(batch_samples)
            progress_bar.update(len(batch_samples))
            continue

        n_done = sum(1 for p in payloads if p is not None)
        n_skip = len(payloads) - n_done
        summary["num_processed"] += n_done
        summary["num_skipped_weak"] += n_skip

        progress_bar.update(len(batch_samples))
        progress_bar.set_postfix(
            processed=summary["num_processed"],
            skipped_weak=summary["num_skipped_weak"],
        )

        # 周期性 GPU / throughput 日志
        now = time.time()
        if now - last_log_t >= log_interval:
            elapsed = now - t0
            rate = summary["num_processed"] / elapsed if elapsed > 0 else 0.0
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024 ** 3
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                util_str = f"GPU mem: {mem:.1f}/{mem_total:.1f}GB"
            else:
                util_str = "CPU"
            logger.info(
                f"[batch {batch_start}~{batch_start+len(batch_samples)}] "
                f"processed={summary['num_processed']} skipped_weak={n_skip} "
                f"rate={rate:.1f} stems/s  {util_str}"
            )
            last_log_t = now

        # 每 100 步打印 Top-1 / Bottom-1 (旧版有, 这里简化掉,
        # 因为 batch 处理后单步样本量变大, 这个 debug 打印意义变小)

    progress_bar.close()
    summary["elapsed_sec"] = time.time() - t0
    logger.info(f"完成. summary = {summary}")

    # ===== 写总览 manifest =====
    subdir = Path(args.candidates_subdir)
    manifest_parent = subdir if subdir.is_absolute() else Path(args.dataset_root)
    manifest_path = manifest_parent / "preference_manifest.json"

    by_weather = {}
    for s in samples:
        w = s["weather"]
        by_weather[w] = by_weather.get(w, 0) + 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_samples": len(samples),
            "num_samples_by_weather": by_weather,
            "num_candidates_per_sample": args.num_candidates,
            "data_batch_size": args.data_batch_size,
            "candidates_subdir": args.candidates_subdir,
            "reward_metrics": enabled,
            "reward_weights_default": args.reward_weights,
            "data_source": "filtered_json" if args.filtered_json else "dataset_root_scan",
            "filtered_json": args.filtered_json,
            "dataset_root": args.dataset_root,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"manifest 写入 {manifest_path}")


if __name__ == "__main__":
    main()