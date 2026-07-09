#!/usr/bin/env python
"""
build_preference.py
===================

离线构建 ControlNet-DPO 偏好对 (preference pairs).

流程:
    1. 遍历 (LQ, GT) 对
    2. 用 SFT 训好的 controlnet 对 LQ 采样 N 个候选恢复图
    3. 用 4 维 reward (PSNR / SSIM / LPIPS / CLIP-IQA) 给每个候选打分
    4. 把 N 张候选图 + 全量分数保存到:
         dataset_root/{weather}/{split}/candidates/{stem}/
             cand_000.png ... cand_{N-1}.png
             score.json
       训练时再按 reward_weights 聚合, 在 Top-k% / Bottom-k% 池内随机抽 winner/loser

使用:
    python scripts/build_preference.py --config config/build_preference.yaml
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

# 让脚本能 import 项目顶层模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

from ramseesr.utils.metrics import (
    psnr as calc_psnr,
    ssim as calc_ssim,
    lpips as calc_lpips,
)

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
# 1. Reward 计算器 (4 维: PSNR / SSIM / LPIPS / CLIP-IQA)
# ============================================================
class RewardScorer:
    """懒加载 lpips 与 clipiqa, 避免重复初始化."""

    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self._lpips = None
        self._clipiqa = None

    def _ensure_lpips(self):
        if self._lpips is None:
            import lpips as lpips_pkg
            self._lpips = lpips_pkg.LPIPS(net="alex", verbose=False).to(self.device, self.dtype)
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

    @torch.no_grad()
    def score_all(self, cand_tensors, gt_tensor, metrics=None):
        """
        输入:
            cand_tensors: list[Tensor[3, H, W]] in [0, 1]
            gt_tensor:    Tensor[3, H, W] in [0, 1]
            metrics:      要计算的指标列表 (None 时默认全部)
                          候选: "psnr" / "ssim" / "lpips" / "clip_iqa"
        返回:
            dict of {metric_name: list[float]}, 长度 = len(cand_tensors)
            含 NaN/Inf 像素的候选: 所有指标都返回 nan, 由 process_one_sample 过滤
            未启用的指标不出现在结果中
        """
        if metrics is None:
            metrics = ["psnr", "ssim", "lpips", "clip_iqa"]
        results = {m: [] for m in metrics}
        n = len(cand_tensors)

        # 预处理: 标记哪些候选含 NaN/Inf 像素 (跳过其 reward 计算, 全部填 nan)
        valid_mask = [bool(torch.isfinite(c).all().item()) for c in cand_tensors]

        # 1) PSNR / SSIM (逐张; 仅对 valid_mask[i]=True 的算)
        if "psnr" in metrics or "ssim" in metrics:
            for i, c in enumerate(cand_tensors):
                if not valid_mask[i]:
                    if "psnr" in metrics: results["psnr"].append(float("nan"))
                    if "ssim" in metrics: results["ssim"].append(float("nan"))
                    continue
                try:
                    if "psnr" in metrics:
                        results["psnr"].append(calc_psnr(c, gt_tensor))
                    if "ssim" in metrics:
                        results["ssim"].append(calc_ssim(c, gt_tensor))
                except Exception as e:
                    logger.debug(f"PSNR/SSIM 失败 (cand {i}): {e}")
                    if "psnr" in metrics: results["psnr"].append(float("nan"))
                    if "ssim" in metrics: results["ssim"].append(float("nan"))

        # 2) LPIPS (逐张; 仅算 valid, 模型失败也写 nan)
        if "lpips" in metrics:
            try:
                lpips_model = self._ensure_lpips()
                gt_in = (gt_tensor.unsqueeze(0) * 2 - 1).to(self.device, self.dtype)
                for i, c in enumerate(cand_tensors):
                    if not valid_mask[i]:
                        results["lpips"].append(float("nan"))
                        continue
                    try:
                        c_in = (c.unsqueeze(0) * 2 - 1).to(self.device, self.dtype)
                        d = lpips_model(c_in, gt_in)
                        results["lpips"].append(float(d.mean().item()))
                    except Exception as e:
                        logger.debug(f"LPIPS 失败 (cand {i}): {e}")
                        results["lpips"].append(float("nan"))
            except Exception as e:
                logger.warning(f"LPIPS 模型加载失败: {e}")
                for _ in range(n):
                    results["lpips"].append(float("nan"))

        # 3) CLIP-IQA (无参考; 仅对 valid 的 batch 处理)
        if "clip_iqa" in metrics:
            iqa_scores = [float("nan")] * n
            valid_indices = [i for i in range(n) if valid_mask[i]]
            if valid_indices:
                try:
                    clipiqa = self._ensure_clipiqa()
                    batch = torch.stack(
                        [cand_tensors[i] for i in valid_indices], dim=0
                    ).to(self.device)
                    scores_tensor = clipiqa(batch)
                    scores_list = (
                        [float(scores_tensor.item())] * len(valid_indices)
                        if scores_tensor.ndim == 0
                        else [float(s) for s in scores_tensor.cpu().tolist()]
                    )
                    for idx, s in zip(valid_indices, scores_list):
                        iqa_scores[idx] = s
                except Exception as e:
                    logger.warning(f"CLIP-IQA 失败: {e}")
                    # iqa_scores 已默认全 nan
            results["clip_iqa"] = iqa_scores

        # 严格只保留 metrics 中列出的 (避免意外多余键)
        return {m: results[m] for m in metrics}


# ============================================================
# 2. 数据集扫描
# ============================================================
def collect_samples(dataset_root, weather_types, splits, weather_num_samples):
    """
    返回 list of dict:
        {weather, split, stem, lq_path, gt_path}
    """
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
                samples.append({
                    "weather": weather,
                    "split": split,
                    "stem": stem,
                    "lq_path": lq_map[stem],
                    "gt_path": gt_map[stem],
                })
                matched += 1
            logger.info(f"[数据集] {weather}/{split}: 匹配 {matched} 对")

    # 按 weather 限制样本数
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


# ============================================================
# 2b. 从 sobel 筛选 JSON 加载样本 (与 collect_samples 输出同构)
# ============================================================
def _derive_lq_path(gt_path: Path) -> Path:
    """
    从 GT 路径派生同名 LQ 路径: 把路径中最后一段 "GT" 替换为 "LQ".
    同时兼容正反斜杠, 也兼容 mixed separators.

    示例:
        D:/data/weafu/rain/train/GT/0001.png  -> D:/data/weafu/rain/train/LQ/0001.png
        D:\\data\\rain\\train\\GT\\0001.png   -> D:\\data\\rain\\train\\LQ\\0001.png
    """
    gt_str = str(gt_path)
    # 兼容两种 separator (含 mixed separators)
    for sep_g, sep_l in [
        ("/", "/"), ("\\", "\\"), ("/", "\\"), ("\\", "/"),
    ]:
        gt_str = gt_str.replace(f"{sep_g}GT{sep_g}", f"{sep_l}LQ{sep_l}")
    return Path(gt_str)


def collect_samples_from_json(
    json_path: str,
    weather_types: list[str],
    splits: list[str],
    weather_num_samples: dict,
):
    """
    从 scripts/filter_dataset/filter_by_sobel.py 输出的 JSON 中加载样本.

    JSON 结构 (来自 sobel):
        {
          "meta": {...},
          "weather_stats": {"rain/train": {...}, ...},
          "samples": [
            {"path": ".../GT/0001.png", "score": 88.10, "weather": "rain", "split": "train"},
            ...
          ]
        }

    输入:
        json_path:          筛选后的 JSON 路径
        weather_types:      要包含的天氣列表 (白名单)
        splits:             要包含的划分列表 (白名单)
        weather_num_samples:{weather: limit} 每种天气截断前 N 条 (按 sobel score 降序)

    返回: list of dict, 与 collect_samples() 输出同构
        [{weather, split, stem, lq_path, gt_path}, ...]
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

    # 1. 白名单过滤
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
        f"[JSON] 原始 {len(raw_samples)} -> 白名单后 {len(filtered)} (跳过 {skipped_weather} 条 weather 不匹配)"
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
        samples.append({
            "weather": s.get("weather", ""),
            "split":   s.get("split", "train"),
            "stem":    gt_path.stem,
            "lq_path": lq_path,
            "gt_path": gt_path,
        })
    if skipped_missing:
        logger.warning(
            f"[JSON] GT 或 LQ 不存在的样本 {skipped_missing} 条, 已跳过"
        )

    # 2. 按 weather 限制样本数
    # JSON 中 samples 已按 sobel score 降序排列, [:limit] 直接取最高分.
    if weather_num_samples:
        new_samples = []
        for w in (weather_types or []):
            ws = [s for s in samples if s["weather"] == w]
            limit = weather_num_samples.get(w, 0)
            if limit and limit > 0 and limit < len(ws):
                logger.info(f"[截断] {w}: {len(ws)} -> {limit}")
                new_samples.extend(ws[:limit])
            else:
                new_samples.extend(ws)
        samples = new_samples

    # 输出统计
    for w in (weather_types or []):
        cnt = sum(1 for s in samples if s["weather"] == w)
        logger.info(f"  - {w}: {cnt}")

    return samples


# ============================================================
# 3. Pipeline 加载
# ============================================================
def load_pipeline(args, weight_dtype, device):
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        if args.controlnet_model_name_or_path
        else None,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    if args.enable_xformers and is_xformers_available():
        pipeline.enable_xformers_memory_efficient_attention()
    return pipeline


# ============================================================
# 4. 单条样本处理: 采样 + 打分
# ============================================================
@torch.no_grad()
def process_one_sample(pipeline, scorer, sample, args, weight_dtype, device, enabled):
    """
    对单条 (LQ, GT) 采样 N 个候选并打分.
    保存到 {candidates_subdir}/{stem}/cand_*.png + score.json
    """
    stem = sample["stem"]
    lq_path = sample["lq_path"]
    gt_path = sample["gt_path"]
    weather = sample["weather"]
    split = sample["split"]

    # candidates_subdir 支持相对 (拼到 dataset_root/{weather}/{split}/) 和绝对两种模式.
    subdir = Path(args.candidates_subdir)
    if subdir.is_absolute():
        cand_root = subdir / weather / split / stem
    else:
        cand_root = (
            Path(args.dataset_root)
            / weather
            / split
            / args.candidates_subdir
            / stem
        )
    cand_root.mkdir(parents=True, exist_ok=True)

    # 读取图像并预处理到固定分辨率
    preprocess = transforms.Compose([
        transforms.Resize(
            (args.height, args.width),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop((args.height, args.width)),
    ])
    to_tensor = transforms.ToTensor()

    lq_pil = Image.open(lq_path).convert("RGB")
    gt_pil = Image.open(gt_path).convert("RGB")
    lq_pil = preprocess(lq_pil)
    gt_pil = preprocess(gt_pil)
    gt_tensor = to_tensor(gt_pil)  # [3, H, W] in [0, 1]

    cand_tensors = []
    skip_sampling = args.rescore_only
    sample_bs = max(1, getattr(args, "sample_batch_size", 4))

    # 分批采样: 每次把 sample_bs 张候选打包成一个 pipeline 调用
    #   - generator 列表保证每张图噪声独立 → 多样性不损失
    #   - prompt/image 列表长度统一为 sample_bs
    i = 0
    while i < args.num_candidates:
        end = min(i + sample_bs, args.num_candidates)
        batch_indices = list(range(i, end))

        all_cached = skip_sampling and all(
            (cand_root / f"cand_{idx:03d}.png").is_file() for idx in batch_indices
        )

        if all_cached:
            for idx in batch_indices:
                cand_path = cand_root / f"cand_{idx:03d}.png"
                cand_tensors.append(
                    to_tensor(Image.open(cand_path).convert("RGB"))
                )
        else:
            prompts = [""] * len(batch_indices)
            images  = [lq_pil] * len(batch_indices)
            generators = [
                torch.Generator(device=device).manual_seed(args.seed_base + idx)
                for idx in batch_indices
            ]
            with torch.autocast("cuda", enabled=(weight_dtype != torch.float32)):
                outs = pipeline(
                    prompt=prompts,
                    image=images,
                    num_inference_steps=args.inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generators,
                    height=args.height,
                    width=args.width,
                ).images  # list[PIL.Image]
            for idx, out in zip(batch_indices, outs):
                out = out.resize((args.width, args.height), Image.BILINEAR)
                cand_path = cand_root / f"cand_{idx:03d}.png"
                out.save(cand_path)
                cand_tensors.append(to_tensor(out))

        i = end

    # 按 args.reward_metrics 启用的项计算 reward
    rewards = scorer.score_all(cand_tensors, gt_tensor, metrics=enabled)

    # 仅保留全部指标 finite 的候选 (过滤 NaN / Inf)
    valid_idx = [
        k for k in range(len(cand_tensors))
        if all(np.isfinite(rewards[m][k]) for m in enabled)
    ]
    n_total = len(cand_tensors)
    n_valid = len(valid_idx)
    if n_valid < 2:
        logger.warning(
            f"[{weather}/{stem}] valid candidates = {n_valid}/{n_total} (< 2), 整条跳过"
        )
        return None

    score_payload = {
        "stem": stem,
        "weather": weather,
        "split": split,
        "num_candidates": n_valid,
        "num_candidates_total": n_total,
        "reward_metrics_enabled": enabled,
        "reward_weights_default": args.reward_weights,  # 默认权重, 实际训练可覆盖
        "candidates": [
            {
                "idx": int(k),
                **{m: float(rewards[m][k]) for m in enabled},
            }
            for k in valid_idx
        ],
    }

    score_path = cand_root / "score.json"
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(score_payload, f, indent=2, ensure_ascii=False)

    return score_payload


# ============================================================
# 5. Argparse + YAML 合并
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/build_preference.yaml")

    # CLI 覆盖 (可选)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--num_candidates", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--rescore_only", action="store_true")
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_batch_size", type=int, default=None,
        help="每次 pipeline 调用同时采样的候选数 (默认 4). "
             "显存够时可调到 8~16 以大幅加速; "
             "guidance_scale=1.0 时显存占用近似线性",
    )
    parser.add_argument(
        "--filtered_json", type=str, default=None,
        help="sobel 筛选后的 JSON 路径 (来自 scripts/filter_dataset/filter_by_sobel.py). "
             "指定后从中读取 GT 路径并按 weather 区分处理; "
             "不指定时按 dataset_root/{weather}/{split}/GT/ 扫描 (旧行为)",
    )
    return parser.parse_args()


def merge_config(args):
    if args.config and Path(args.config).is_file():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
    return args


# ============================================================
# 6. Main
# ============================================================
def main():
    args = parse_args()
    args = merge_config(args)
    args.train_method = "build_preference"  # 占位, 防止 merge_config 后属性缺失
    if not hasattr(args, "enable_xformers"):
        args.enable_xformers = False
    if not hasattr(args, "candidates_subdir"):
        args.candidates_subdir = "candidates"
    if not hasattr(args, "max_samples"):
        args.max_samples = None
    if not hasattr(args, "rescore_only"):
        args.rescore_only = False
    if not hasattr(args, "filtered_json"):
        args.filtered_json = None
    if not hasattr(args, "sample_batch_size") or args.sample_batch_size is None:
        args.sample_batch_size = 4

    # weather_num_samples 整合为 dict
    weather_num_samples = {}
    for w in (args.weather_types or []):
        attr = f"{w}_num"
        if hasattr(args, attr) and getattr(args, attr):
            weather_num_samples[w] = getattr(args, attr)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # device / dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 加载 pipeline
    logger.info(f"加载 pipeline from {args.pretrained_model_name_or_path}")
    pipeline = load_pipeline(args, weight_dtype, device)

    # 收集样本 (入口分流: JSON 模式 vs 旧扫描模式)
    if args.filtered_json:
        samples = collect_samples_from_json(
            args.filtered_json,
            args.weather_types,
            args.splits,
            weather_num_samples,
        )
    else:
        samples = collect_samples(
            args.dataset_root,
            args.weather_types,
            args.splits,
            weather_num_samples,
        )
    if args.max_samples:
        samples = samples[: args.max_samples]
    logger.info(f"待处理样本数: {len(samples)}")

    # 启用的奖励指标 (来自 yaml 的 reward_metrics 列表)
    enabled = getattr(args, "reward_metrics", None) or [
        "psnr", "ssim", "lpips", "clip_iqa"
    ]
    logger.info(f"启用的 reward 指标: {enabled}")

    # reward scorer
    scorer = RewardScorer(device, weight_dtype)
    # 按需懒加载 (避免没启用 clip_iqa 时仍要求 pyiqa)
    if "lpips" in enabled:
        scorer._ensure_lpips()
    if "clip_iqa" in enabled:
        try:
            scorer._ensure_clipiqa()
        except ImportError as e:
            logger.error(str(e))
            return
    logger.info(f"Reward 模型加载完成 ({', '.join(enabled)})")

    # 逐条处理
    summary = {
        "num_processed": 0,
        "num_skipped_error": 0,
        "num_skipped_weak": 0,    # NaN / valid < 2
        "sample_batch_size": args.sample_batch_size,
        "elapsed_sec": 0.0,
    }
    t0 = time.time()
    for idx, s in enumerate(samples):
        try:
            payload = process_one_sample(
                pipeline, scorer, s, args, weight_dtype, device, enabled,
            )
            if payload is None:
                summary["num_skipped_weak"] += 1
                continue
            summary["num_processed"] += 1
            if (idx + 1) % 20 == 0 or idx == 0:
                # 打印 Top-1 / Bottom-1 简单检查
                scores = [
                    sum(c[m] * args.reward_weights.get(m, 0) for m in enabled)
                    for c in payload["candidates"]
                ]
                best = max(scores)
                worst = min(scores)
                logger.info(
                    f"[{idx+1}/{len(samples)}] {s['weather']}/{s['stem']}  "
                    f"score_range=[{worst:.3f}, {best:.3f}]  gap={best-worst:.3f}  "
                    f"valid={payload['num_candidates']}/{payload['num_candidates_total']}"
                )
        except Exception as e:
            logger.exception(f"处理 {s['stem']} 失败: {e}")
            summary["num_skipped_error"] += 1

    summary["elapsed_sec"] = time.time() - t0
    logger.info(f"完成. summary = {summary}")

    # 写总览 manifest
    # manifest 也支持 candidates_subdir 为绝对路径的情况
    subdir = Path(args.candidates_subdir)
    manifest_parent = subdir if subdir.is_absolute() else Path(args.dataset_root)
    manifest_path = manifest_parent / "preference_manifest.json"
    # 按 weather 拆 stats, 便于排查
    by_weather = {}
    for s in samples:
        w = s["weather"]
        by_weather[w] = by_weather.get(w, 0) + 1
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": len(samples),
                "num_samples_by_weather": by_weather,
                "num_candidates_per_sample": args.num_candidates,
                "candidates_subdir": args.candidates_subdir,
                "reward_metrics": enabled,
                "reward_weights_default": args.reward_weights,
                "data_source": "filtered_json" if args.filtered_json else "dataset_root_scan",
                "filtered_json": args.filtered_json,
                "dataset_root": args.dataset_root,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"manifest 写入 {manifest_path}")


if __name__ == "__main__":
    main()
