#!/usr/bin/env python
"""
filter_by_sobel.py
==================

按 Sobel 梯度幅值对 GT 图像排序, 筛选出"细节最丰富"的图像,
把入选路径写入 JSON. 支持两种模式:

模式 A — 单目录 (向后兼容)
    递归扫描 --gt_root 下所有图像, 排序后取 top_k.

    python scripts/filter_dataset/filter_by_sobel.py \
        --gt_root /data/weafu/rain/train/GT \
        --top_k 7000 \
        --output_json ./experiment/filtered/rain.json

模式 B — 多天气 (推荐用于 SFT/DPO 数据准备)
    假设目录结构:
        {gt_root}/{weather}/{split}/GT/*.png
    对每个 (weather, split) 组合内的 GT 分别排序, 各取 top_k,
    输出一个统一 JSON, 样本带 weather 字段.

    python scripts/filter_dataset/filter_by_sobel.py \
        --gt_root /data/weafu \
        --weather_types rain snow haze \
        --splits train \
        --top_k 7000 \
        --output_json ./experiment/filtered/top7000_per_weather.json \
        --num_workers 8

支持的 metric:
    - mean    : Sobel 幅值的均值 (整体纹理/对比度, 默认)
    - std     : Sobel 幅值的标准差 (边缘分布的离散度)
    - mean_std: mean * std (复合指标)

实现:
    - cv2.Sobel (CV_32F, ksize=3) -> cv2.magnitude
    - ProcessPoolExecutor 并行计算, tqdm 进度
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("filter_by_sobel")


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# 1. 单图 Sobel 评分
# ============================================================
def sobel_score(
    img_path: str,
    metric: str = "mean",
    resize: Optional[int] = None,
) -> Tuple[str, float]:
    """返回 (path, score); 失败时 score = -1.0."""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return img_path, -1.0

        if resize and resize > 0:
            h, w = img.shape
            short = min(h, w)
            if short > resize:
                scale = resize / short
                img = cv2.resize(
                    img, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        if metric == "mean":
            s = float(mag.mean())
        elif metric == "std":
            s = float(mag.std())
        elif metric == "mean_std":
            s = float(mag.mean() * mag.std())
        else:
            raise ValueError(f"未知 metric: {metric}")

        return img_path, s
    except Exception as e:
        logger.debug(f"读取/计算失败 {img_path}: {e}")
        return img_path, -1.0


# ============================================================
# 2. 目录扫描
# ============================================================
def scan_images(root: str) -> List[str]:
    """递归收集 root 下所有图像文件路径."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
            paths.append(str(p.resolve()))
    paths.sort()
    return paths


# ============================================================
# 3. 并行评分
# ============================================================
def parallel_score(
    paths: List[str],
    metric: str,
    resize: Optional[int],
    num_workers: int,
) -> List[Tuple[str, float]]:
    """并行计算 Sobel 分数, 返回 (path, score) 列表 (与输入同序)."""
    if num_workers <= 1:
        results = []
        for p in tqdm(paths, desc="Sobel"):
            results.append(sobel_score(p, metric=metric, resize=resize))
        return results

    results: List[Optional[Tuple[str, float]]] = [None] * len(paths)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(sobel_score, p, metric, resize): i
            for i, p in enumerate(paths)
        }
        with tqdm(total=len(paths), desc="Sobel") as bar:
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
                bar.update(1)
    for i, r in enumerate(results):
        if r is None:
            results[i] = (paths[i], -1.0)
    return results  # type: ignore


# ============================================================
# 4. 单组排序 + 截取
# ============================================================
def rank_and_take(
    scored: List[Tuple[str, float]],
    top_k: int,
    min_score: Optional[float] = None,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], int]:
    """
    输入:  (path, score) 列表
    输出:  (selected_top_k, all_valid, invalid_count)
    """
    valid = [(p, s) for p, s in scored if s >= 0]
    invalid = len(scored) - len(valid)
    if min_score is not None:
        valid = [(p, s) for p, s in valid if s >= min_score]
    valid.sort(key=lambda x: x[1], reverse=True)
    take = min(top_k, len(valid))
    return valid[:take], valid, invalid


# ============================================================
# 5. 多天气模式
# ============================================================
def run_multi_weather(args) -> Dict:
    """
    假设结构: {gt_root}/{weather}/{split}/GT/
    对每个 (weather, split) 组合的 GT 排序, 各取 top_k, 汇总.
    """
    weather_stats: Dict[str, Dict] = {}
    all_selected: List[Dict] = []

    for weather in args.weather_types:
        for split in args.splits:
            gt_dir = Path(args.gt_root) / weather / split / "GT"
            if not gt_dir.is_dir():
                logger.warning(f"[跳过] {gt_dir} 不存在")
                weather_stats[f"{weather}/{split}"] = {
                    "total_scanned": 0,
                    "num_selected": 0,
                    "score_min": None,
                    "score_max": None,
                    "score_mean": None,
                }
                continue

            logger.info(f"[{weather}/{split}] 扫描 {gt_dir}")
            paths = scan_images(str(gt_dir))
            logger.info(f"[{weather}/{split}] 共 {len(paths)} 张")

            if not paths:
                weather_stats[f"{weather}/{split}"] = {
                    "total_scanned": 0,
                    "num_selected": 0,
                    "score_min": None,
                    "score_max": None,
                    "score_mean": None,
                }
                continue

            scored = parallel_score(
                paths,
                metric=args.metric,
                resize=args.resize,
                num_workers=args.num_workers,
            )

            selected, valid, invalid = rank_and_take(
                scored, args.top_k, min_score=args.min_score,
            )

            # 统计
            sel_scores = [s for _, s in selected]
            stats = {
                "total_scanned": len(paths),
                "num_valid": len(valid),
                "num_invalid": invalid,
                "num_selected": len(selected),
                "score_min": round(min(sel_scores), 4) if sel_scores else None,
                "score_max": round(max(sel_scores), 4) if sel_scores else None,
                "score_mean": round(sum(sel_scores) / len(sel_scores), 4) if sel_scores else None,
            }
            weather_stats[f"{weather}/{split}"] = stats
            logger.info(
                f"[{weather}/{split}] selected={len(selected)}  "
                f"score=[{stats['score_min']}, {stats['score_max']}]  "
                f"mean={stats['score_mean']}"
            )

            for p, s in selected:
                all_selected.append({
                    "path": p,
                    "score": round(float(s), 4),
                    "weather": weather,
                    "split": split,
                })

    return {
        "meta": {
            "gt_root": str(Path(args.gt_root).resolve()),
            "weather_types": args.weather_types,
            "splits": args.splits,
            "per_weather_top_k": args.top_k,
            "total_selected": len(all_selected),
            "metric": args.metric,
            "img_extensions": sorted(IMG_EXTENSIONS),
        },
        "weather_stats": weather_stats,
        "samples": all_selected,
    }


# ============================================================
# 6. 单目录模式
# ============================================================
def run_single_dir(args) -> Dict:
    """递归扫描 --gt_root, 排序后取 top_k (向后兼容)."""
    logger.info(f"扫描目录: {args.gt_root}")
    paths = scan_images(args.gt_root)
    logger.info(f"找到 {len(paths)} 张图像")

    if not paths:
        logger.error("未找到任何图像, 退出")
        sys.exit(1)

    scored = parallel_score(
        paths,
        metric=args.metric,
        resize=args.resize,
        num_workers=args.num_workers,
    )

    selected, valid, invalid = rank_and_take(
        scored, args.top_k, min_score=args.min_score,
    )

    if invalid:
        logger.warning(f"过滤掉 {invalid} 张无效图像")

    sel_scores = [s for _, s in selected]
    if sel_scores:
        logger.info(
            f"Top-{len(selected)} 分数: "
            f"min={min(sel_scores):.3f}  "
            f"max={max(sel_scores):.3f}  "
            f"mean={sum(sel_scores)/len(sel_scores):.3f}"
        )

    all_scores = [s for _, s in valid]
    if all_scores:
        logger.info(
            f"全部 {len(valid)} 张: "
            f"min={min(all_scores):.3f}  "
            f"max={max(all_scores):.3f}  "
            f"mean={sum(all_scores)/len(all_scores):.3f}"
        )

    samples = [
        {"path": p, "score": round(float(s), 4)}
        for p, s in selected
    ]

    return {
        "meta": {
            "gt_root": str(Path(args.gt_root).resolve()),
            "total_scanned": len(paths),
            "num_selected": len(samples),
            "metric": args.metric,
            "img_extensions": sorted(IMG_EXTENSIONS),
        },
        "samples": samples,
        "scores_all": [
            {"path": p, "score": round(float(s), 4)}
            for p, s in valid
        ],
    }


# ============================================================
# 7. CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="用 Sobel 算子筛选 GT 数据集中细节最丰富的图像 "
                    "(支持多天气分别 top_k 模式)",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="YAML 配置文件路径; 传入后, 未在 CLI 显式指定的参数从 yaml 填充",
    )
    p.add_argument(
        "--gt_root", type=str, default=None,
        help="GT 根目录. 单目录模式: 直接传 GT 目录; "
             "多天气模式: 传数据集根目录, 结构为 {gt_root}/{weather}/{split}/GT/",
    )
    p.add_argument(
        "--output_json", type=str, default=None,
        help="输出 JSON 路径",
    )
    p.add_argument(
        "--top_k", type=int, default=None,
        help="每种天气保留的图像数量 (默认 7000)",
    )
    p.add_argument(
        "--weather_types", type=str, nargs="+", default=None,
        help="天气类型列表, 如 --weather_types rain snow haze. "
             "不指定时为单目录模式; 指定后启用多天气模式, "
             "自动从 {gt_root}/{weather}/{split}/GT/ 读取",
    )
    p.add_argument(
        "--splits", type=str, nargs="+", default=None,
        help="参与筛选的数据划分 (默认 train)",
    )
    p.add_argument(
        "--metric", type=str, default=None,
        choices=["mean", "std", "mean_std"],
        help="细节丰富度指标 (默认 mean)",
    )
    p.add_argument(
        "--num_workers", type=int, default=None,
        help="并行 worker 数 (默认 8)",
    )
    p.add_argument(
        "--resize", type=int, default=None,
        help="计算前把短边 resize 到该值 (加速)",
    )
    p.add_argument(
        "--min_score", type=float, default=None,
        help="最低分数阈值, 低于该值的图像直接丢弃",
    )
    p.add_argument(
        "--save_scores", action="store_true",
        default=False,
        help="把全部 (path, score) 也写入 JSON 的 scores_all 字段 (仅单目录模式)",
    )
    return p.parse_args()


def merge_config(args):
    """将 yaml 配置填充到 args 中, 只覆盖 CLI 未显式指定的字段 (None)."""
    if not args.config:
        return args
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        logger.warning(f"--config 指定的文件不存在: {cfg_path}")
        return args
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in cfg.items():
        if not hasattr(args, k):
            # yaml 里有但 argparse 未声明的 key, 跳过
            continue
        current = getattr(args, k)
        # 只覆盖 CLI 未指定 (None) 的字段
        if current is None:
            setattr(args, k, v)
            logger.info(f"[yaml] {k} = {v}")
        # 特殊处理: bool action flag, yaml 设 true 时启用
        # argparse 里 --save_scores 是 action="store_true", 默认 False
        # yaml 写 save_scores: true 时强制启用
        elif isinstance(v, bool) and isinstance(current, bool) and v and not current:
            setattr(args, k, v)
    return args


def finalize_defaults(args):
    """对仍为 None 的字段填入程序默认值."""
    if args.top_k is None:
        args.top_k = 7000
    if args.splits is None:
        args.splits = ["train"]
    if args.metric is None:
        args.metric = "mean"
    if args.num_workers is None:
        args.num_workers = 8
    # gt_root 与 output_json 由 yaml/CLI 二选其一, 必须存在
    if not args.gt_root:
        raise ValueError("必须通过 --gt_root 或 yaml 配置提供 gt_root")
    if not args.output_json:
        raise ValueError("必须通过 --output_json 或 yaml 配置提供 output_json")
    return args


# 兼容保留: 旧版本别名 (单目录模式兼容)
_REQUIRED_BEFORE = ("gt_root", "output_json")


# ============================================================
# 8. Main
# ============================================================
def main():
    args = parse_args()
    args = merge_config(args)
    args = finalize_defaults(args)
    t0 = time.time()

    is_multi = bool(args.weather_types)

    if is_multi:
        payload = run_multi_weather(args)
        # 多天气模式默认写 weather_stats
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        total = payload["meta"]["total_selected"]
        logger.info(f"已写入 {total} 条 ({len(args.weather_types)} 种天气) 到 {out_path}")
    else:
        payload = run_single_dir(args)
        if not args.save_scores:
            payload.pop("scores_all", None)
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"已写入 {payload['meta']['num_selected']} 条到 {out_path}")

    logger.info(f"总耗时: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
