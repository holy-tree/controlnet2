"""
ControlNet 多天气图像恢复 - 独立评估脚本
============================================

用法:
    python evaluate.py --config ./config/eval.yaml

复用 train.yaml 的目录结构, 对 test 集按天气类型分别评估:
    PSNR  -- 越高越好
    SSIM  -- 越高越好 (0~1)
    LPIPS -- 越低越好 (感知距离)
    FID   -- 越低越好 (InceptionV3 特征分布距离)

输出:
    <output_dir>/<timestamp>_eval/
    ├── metrics.txt                # 汇总指标 (全样本 + 各天气)
    ├── <weather>/                 # 每种天气一个目录
    │   ├── 000_xxx_pred.png      # ControlNet 恢复图
    │   ├── 000_xxx_lq.png        # 输入退化图
    │   ├── 000_xxx_gt.png        # 真值图
    │   └── per_image_metrics.txt # 每张图的指标
    └── ...
"""

import argparse
import io
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.getcwd())

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils.import_utils import is_xformers_available

from dataloaders.paired_dataset import DEFAULT_WEATHER_PROMPTS
from ramseesr.utils.metrics import fid as calc_fid, lpips, psnr as calc_psnr, ssim as calc_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="ControlNet 多天气图像恢复评估")
    parser.add_argument("--config", type=str, default="./config/eval.yaml",
                        help="YAML 配置文件路径")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with io.open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset_for_eval(args_config: dict):
    """
    加载 test 集, 返回 (gt_path, lq_path, weather, subdataset) 元组列表.

    支持结构:
        {dataset_root}/{subdataset}/{gt,lq}/   (subdataset 模式, 自动遍历子目录)
        {dataset_root}/{weather}/{subdataset}/{gt,lq}/  (嵌套模式)

    subdataset 名 = "{weather}_{subdir}" (如 "rain_Rain100H"),
    保证在循环分组时不会冲突。
    """
    from pathlib import Path as P
    IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def _is_img(p: P) -> bool:
        return p.is_file() and p.suffix.lower() in IMG_EXT

    def _match_pairs(gt_dir: P, lq_dir: P):
        if not gt_dir.is_dir() or not lq_dir.is_dir():
            return []
        gt_map = {p.stem: p for p in gt_dir.iterdir() if _is_img(p)}
        lq_map = {p.stem: p for p in lq_dir.iterdir() if _is_img(p)}
        return [(gt_map[s], lq_map[s]) for s in sorted(set(gt_map) & set(lq_map))]

    all_samples = []
    for weather in args_config["weather_types"]:
        root_key = f"dataset_{weather}"
        dataset_root_str = args_config.get(root_key)
        if not dataset_root_str:
            print(f"[warn] {root_key} 未设置, 跳过 {weather}")
            continue

        dataset_root = P(dataset_root_str)
        if not dataset_root.is_dir():
            print(f"[warn] {dataset_root} 不存在, 跳过 {weather}")
            continue

        # 遍历子目录作为 subdataset, 但按 splits 过滤
        #   - 目录名是 split 名 (如 "test" / "train" / "test_a"): 只保留在 splits 列表里的
        #   - 目录名不在 splits 里 (如 "sub_a" / "testset_v1"): 不当 split, 由下面的"情况 1/2/3"尝试匹配
        splits = args_config.get("splits", ["test"])
        subdirs_all = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
        subdirs_split = [p for p in subdirs_all if p.name in splits]
        # 如果有按 split 命名的子目录, 用它; 否则保留所有(继续由情况 1/2/3 匹配)
        subdirs = subdirs_split if subdirs_split else subdirs_all
        if not subdirs:
            print(f"[warn] {dataset_root} 下没有子目录, 跳过 {weather}")
            continue

        for subdir in subdirs:
            sub_name = f"{weather}_{subdir.name}"

            # 情况 1: {subdir}/{gt,lq}/  (organize_testset.py 输出)
            gt_dir = subdir / "gt"
            lq_dir = subdir / "lq"
            pairs = _match_pairs(gt_dir, lq_dir)

            # 情况 2: {subdir}/{split}/{gt,lq}/  (train 风格)
            if not pairs:
                for split in args_config.get("splits", ["test"]):
                    pairs = _match_pairs(subdir / split / "GT", subdir / split / "LQ")
                    if pairs:
                        break

            # 情况 3: 兼容老结构 — 整目录作为 LQ/GT
            if not pairs:
                pairs = _match_pairs(subdir / "GT", subdir / "LQ")

            if not pairs:
                print(f"  [跳过] {sub_name}: 未找到图像对")
                continue

            print(f"  [加载] {sub_name}: {len(pairs)} 对")
            for gt_path, lq_path in pairs:
                all_samples.append((str(gt_path), str(lq_path), weather, sub_name))

    return all_samples


def resolve_controlnet_path(raw_path: str) -> str:
    """
    智能解析 ControlNet 权重路径, 支持以下目录结构:
      1. <path>/controlnet/config.json                       # save_pretrained 直接输出
      2. <path>/config.json                                  # 用户指定的就是权重目录
      3. <path>/checkpoint-<N>/controlnet/config.json        # accelerate epoch/step checkpoint
    自动检测并返回真正的 ControlNet 目录 (包含 config.json 的那个)。
    """
    p = Path(raw_path)
    if not p.is_absolute():
        p = Path.cwd() / p

    # 情况 1: 路径本身直接有 config.json
    if (p / "config.json").is_file():
        return str(p)

    # 情况 2: 路径下有 controlnet/ 子目录 (save_pretrained 输出)
    if (p / "controlnet" / "config.json").is_file():
        return str(p / "controlnet")

    # 情况 3: 路径下有 checkpoint-*/controlnet/ (accelerate save_state 输出)
    candidates = sorted(p.glob("checkpoint-*/controlnet/config.json"))
    if candidates:
        # 取最新的 checkpoint
        latest = candidates[-1]
        print(f"[resolve] 在 {p} 下发现多个 checkpoint, 使用最新的: {latest.parent.parent.name}")
        return str(latest.parent)

    # 找不到, 让 from_pretrained 自己报错
    return str(p)


def build_pipeline(args_config: dict, device, dtype):
    """构建推理 pipeline (复用 SD + 训练好的 ControlNet)"""
    cn_path = resolve_controlnet_path(args_config["controlnet_model_path"])
    print(f"[eval] ControlNet 路径: {cn_path}")
    controlnet = ControlNetModel.from_pretrained(cn_path)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args_config["pretrained_model_name_or_path"],
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args_config.get("enable_xformers_memory_efficient_attention", False):
        if is_xformers_available():
            pipeline.enable_xformers_memory_efficient_attention()
        else:
            print("[warn] xformers 不可用, 已跳过")

    return pipeline


def maybe_make_prompt(weather: str, args_config: dict) -> str:
    """根据 use_prompt / prompt_ratio 决定是否使用天气 prompt"""
    if not args_config.get("use_prompt", False):
        return ""
    if random.random() < args_config.get("prompt_ratio", 0.2):
        return DEFAULT_WEATHER_PROMPTS.get(weather, "")
    return ""


def evaluate(args_config: dict):
    # ===== 设备与精度 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32
    if args_config.get("mixed_precision") == "fp16":
        weight_dtype = torch.float16
    elif args_config.get("mixed_precision") == "bf16":
        weight_dtype = torch.bfloat16
    print(f"[eval] device={device}, dtype={weight_dtype}")

    # ===== 加载样本 =====
    samples = build_dataset_for_eval(args_config)
    print(f"[eval] 共加载 {len(samples)} 个样本")

    # 按 subdataset 分组 (key 如 "rain_Rain100H"); 同时记录所属 weather
    by_sub: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    sub_to_weather: Dict[str, str] = {}
    for gt_path, lq_path, weather, sub_name in samples:
        by_sub[sub_name].append((gt_path, lq_path))
        sub_to_weather[sub_name] = weather

    # 按 weather 分组 (聚合用)
    by_weather: Dict[str, List[str]] = defaultdict(list)
    for sub_name in by_sub:
        by_weather[sub_to_weather[sub_name]].append(sub_name)

    # ===== 评估采样数: 控制用于计算指标 (PSNR/SSIM/LPIPS) 的样本数 (per subdataset) =====
    # 默认使用全部 test 集 (保证指标统计准确)
    default_max = args_config.get("max_samples_per_weather", 0)
    for sub_name in by_sub:
        n = len(by_sub[sub_name])
        if default_max and default_max > 0 and n > default_max:
            random.shuffle(by_sub[sub_name])
            by_sub[sub_name] = by_sub[sub_name][:default_max]
            print(f"[eval] {sub_name}: 评估采样截断为 {default_max} (用于指标计算)")
        else:
            print(f"[eval] {sub_name}: 评估使用全部 {n} 样本")

    # ===== 可视化保存数: 控制保存多少组 pred/lq/gt 图 (per subdataset) =====
    save_counts: Dict[str, int] = {}
    save_predictions_global = args_config.get("save_predictions", True)
    if save_predictions_global:
        for sub_name in by_sub:
            # 支持 per-subdataset 配置: {subdataset_name}_num, 也兼容 {weather}_num
            sub_key = sub_name.split("_", 1)[-1] + "_num"  # 取 subdataset 短名
            weather = sub_to_weather[sub_name]
            v = args_config.get(f"{weather}_{sub_key}", None)
            if v is None:
                v = args_config.get(f"{weather}_num", -1)
            if v is None:
                v = -1
            if v == 0:
                save_counts[sub_name] = 0
            elif v < 0:
                save_counts[sub_name] = len(by_sub[sub_name])
            else:
                save_counts[sub_name] = min(v, len(by_sub[sub_name]))
            print(f"[eval] {sub_name}: 可视化保存 {save_counts[sub_name]} 张")
    else:
        for sub_name in by_sub:
            save_counts[sub_name] = 0
        print(f"[eval] save_predictions=false, 不保存任何预测图")

    # ===== 输出目录 =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_root = Path(args_config["output_dir"]) / f"{timestamp}_eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    # ===== 构建 pipeline =====
    pipeline = build_pipeline(args_config, device, weight_dtype)

    # ===== 图像预处理 =====
    preprocess = transforms.Compose([
        transforms.Resize(args_config["resolution"], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args_config["resolution"]),
        transforms.ToTensor(),
    ])

    # ===== 评估循环 =====
    # 存储每张图的指标: per_image_results[sub_name] = [(name, psnr, ssim, lpips), ...]
    per_image_results: Dict[str, List] = defaultdict(list)
    # 收集预测与 GT 张量, 用于计算 FID (每个 subdataset 和 weather 各一组)
    fid_preds_sub: Dict[str, List[torch.Tensor]] = defaultdict(list)
    fid_gts_sub: Dict[str, List[torch.Tensor]] = defaultdict(list)
    fid_preds_weather: Dict[str, List[torch.Tensor]] = defaultdict(list)
    fid_gts_weather: Dict[str, List[torch.Tensor]] = defaultdict(list)
    # 是否启用 FID
    enable_fid = args_config.get("enable_fid", True)

    # 固定种子以便复现
    if args_config.get("seed") is not None:
        random.seed(args_config["seed"])
        torch.manual_seed(args_config["seed"])

    total_samples = sum(len(v) for v in by_sub.values())
    pbar = tqdm(total=total_samples, desc="Eval")

    lpips_net = args_config.get("lpips_net", "alex")

    # 按 weather 顺序遍历 subdataset
    for weather in args_config["weather_types"]:
        if weather not in by_weather:
            print(f"[eval] 跳过 {weather}: 没有样本")
            continue

        weather_dir = eval_root / weather
        weather_dir.mkdir(parents=True, exist_ok=True)

        for sub_name in by_weather[weather]:
            sub_dir = eval_root / sub_name
            sub_dir.mkdir(parents=True, exist_ok=True)
            n_to_save = save_counts.get(sub_name, 0)

            for sample_idx, (gt_path, lq_path) in enumerate(by_sub[sub_name]):
                # 加载图像
                gt_img = preprocess(Image.open(gt_path).convert("RGB"))
                lq_img = preprocess(Image.open(lq_path).convert("RGB"))

                # 准备 prompt
                prompt = maybe_make_prompt(weather, args_config)

                # LQ -> diffusion -> pred
                lq_pil = transforms.ToPILImage()(lq_img)
                t0 = time.time()
                with torch.autocast("cuda", enabled=(device.type == "cuda")):
                    pred_pil = pipeline(
                        prompt,
                        lq_pil,
                        num_inference_steps=args_config["num_inference_steps"],
                        guidance_scale=args_config["guidance_scale"],
                        negative_prompt=args_config["negative_prompt"],
                        height=args_config["resolution"],
                        width=args_config["resolution"],
                    ).images[0]
                infer_time = time.time() - t0

                # pred -> tensor
                pred_tensor = transforms.ToTensor()(pred_pil).to(device).clamp(0, 1)
                gt_tensor = gt_img.to(device)

                # 计算指标 (每张都算)
                p = calc_psnr(pred_tensor, gt_tensor)
                s = calc_ssim(pred_tensor, gt_tensor)
                try:
                    l = lpips(pred_tensor, gt_tensor, net=lpips_net)
                except Exception as e:
                    print(f"[warn] LPIPS 计算失败 ({Path(gt_path).name}): {e}")
                    l = float("nan")

                stem = Path(gt_path).stem
                per_image_results[sub_name].append((stem, p, s, l, infer_time))

                # 收集用于 FID 计算的张量 (CPU 张量, 避免长时间占 GPU 显存)
                if enable_fid:
                    pred_cpu = pred_tensor.detach().cpu()
                    gt_cpu = gt_tensor.detach().cpu()
                    fid_preds_sub[sub_name].append(pred_cpu)
                    fid_gts_sub[sub_name].append(gt_cpu)
                    fid_preds_weather[weather].append(pred_cpu)
                    fid_gts_weather[weather].append(gt_cpu)

                # 保存图片 (受 save_counts 控制, 与评估数解耦)
                if sample_idx < n_to_save:
                    pred_pil.save(sub_dir / f"{sample_idx:03d}_{stem}_pred.png")
                    lq_pil.save(sub_dir / f"{sample_idx:03d}_{stem}_lq.png")
                    transforms.ToPILImage()(gt_img).save(sub_dir / f"{sample_idx:03d}_{stem}_gt.png")

                pbar.set_postfix(sub=sub_name, psnr=f"{p:.2f}", ssim=f"{s:.4f}", lpips=f"{l:.4f}")
                pbar.update(1)

    pbar.close()

    # ===== 汇总每个 subdataset 的指标 =====
    sub_metrics: Dict[str, Dict] = {}
    for sub_name, items in per_image_results.items():
        psnrs = [x[1] for x in items]
        ssims = [x[2] for x in items]
        lpipss = [x[3] for x in items if not (isinstance(x[3], float) and x[3] != x[3])]
        times = [x[4] for x in items]
        sub_metrics[sub_name] = {
            "weather": sub_to_weather[sub_name],
            "n": len(items),
            "psnr": sum(psnrs) / len(psnrs) if psnrs else 0.0,
            "ssim": sum(ssims) / len(ssims) if ssims else 0.0,
            "lpips": sum(lpipss) / len(lpipss) if lpipss else float("nan"),
            "avg_time": sum(times) / len(times) if times else 0.0,
        }

    # ===== 计算每个 subdataset 的 FID =====
    if enable_fid:
        print("\n[FID] 开始计算 FID...")
        for sub_name in sub_metrics:
            try:
                fid_val = calc_fid(fid_preds_sub[sub_name], fid_gts_sub[sub_name])
                sub_metrics[sub_name]["fid"] = fid_val
                print(f"  [FID] {sub_name}: {fid_val:.4f} (N={len(fid_preds_sub[sub_name])})")
            except Exception as e:
                print(f"  [FID] {sub_name} 计算失败: {e}")
                sub_metrics[sub_name]["fid"] = float("nan")
    else:
        for sub_name in sub_metrics:
            sub_metrics[sub_name]["fid"] = float("nan")

    # ===== 汇总每个 weather 的聚合指标 (基于其所有 subdataset) =====
    weather_metrics: Dict[str, Dict] = {}
    for weather in args_config["weather_types"]:
        sub_list = by_weather.get(weather, [])
        if not sub_list:
            continue
        # 聚合所有 subdataset 的样本
        all_p = []
        all_s = []
        all_l = []
        all_t = []
        for sub_name in sub_list:
            for _, p, s, l, t in per_image_results[sub_name]:
                all_p.append(p)
                all_s.append(s)
                if not (isinstance(l, float) and l != l):
                    all_l.append(l)
                all_t.append(t)
        weather_metrics[weather] = {
            "n": len(all_p),
            "psnr": sum(all_p) / len(all_p) if all_p else 0.0,
            "ssim": sum(all_s) / len(all_s) if all_s else 0.0,
            "lpips": sum(all_l) / len(all_l) if all_l else float("nan"),
            "avg_time": sum(all_t) / len(all_t) if all_t else 0.0,
        }

    # ===== 计算每个 weather 的 FID =====
    if enable_fid:
        for weather in args_config["weather_types"]:
            if weather in fid_preds_weather and len(fid_preds_weather[weather]) > 0:
                try:
                    fid_val = calc_fid(fid_preds_weather[weather], fid_gts_weather[weather])
                    weather_metrics[weather]["fid"] = fid_val
                    print(f"  [FID] {weather}: {fid_val:.4f} (N={len(fid_preds_weather[weather])})")
                except Exception as e:
                    print(f"  [FID] {weather} 计算失败: {e}")
                    weather_metrics[weather]["fid"] = float("nan")
            else:
                weather_metrics[weather]["fid"] = float("nan")
        # 总体 FID (所有 sample 合并)
        all_preds = []
        all_gts = []
        for w in fid_preds_weather:
            all_preds.extend(fid_preds_weather[w])
            all_gts.extend(fid_gts_weather[w])
        if all_preds:
            try:
                overall_fid = calc_fid(all_preds, all_gts)
                print(f"  [FID] Overall: {overall_fid:.4f} (N={len(all_preds)})")
            except Exception as e:
                print(f"  [FID] Overall 计算失败: {e}")
                overall_fid = float("nan")
        else:
            overall_fid = float("nan")
    else:
        overall_fid = float("nan")
        for w in weather_metrics:
            weather_metrics[w]["fid"] = float("nan")

    # 写每个 subdataset 的 per_image 指标文件
    for sub_name, items in per_image_results.items():
        per_img_path = eval_root / sub_name / "per_image_metrics.txt"
        with open(per_img_path, "w", encoding="utf-8") as f:
            f.write(f"# Per-image metrics for subdataset={sub_name}\n")
            f.write(f"# name, PSNR, SSIM, LPIPS, infer_time(s)\n")
            for stem, p, s, l, t in items:
                f.write(f"{stem}, {p:.4f}, {s:.4f}, {l:.4f}, {t:.2f}\n")

    # ===== 写总 metrics.txt =====
    summary_path = eval_root / "metrics.txt"
    total_n = sum(m["n"] for m in weather_metrics.values())
    all_psnrs = []
    all_ssims = []
    all_lpipss = []
    for items in per_image_results.values():
        for _, p, s, l, _ in items:
            all_psnrs.append(p)
            all_ssims.append(s)
            if not (isinstance(l, float) and l != l):
                all_lpipss.append(l)

    def _fmt(v):
        return f"{v:.4f}" if v == v else "  N/A  "

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("ControlNet Multi-Weather Image Restoration - Evaluation Report\n")
        f.write("=" * 90 + "\n")
        f.write(f"Timestamp:        {timestamp}\n")
        f.write(f"Model:           {args_config['controlnet_model_path']}\n")
        f.write(f"SD base:         {args_config['pretrained_model_name_or_path']}\n")
        ds_roots = {w: args_config.get(f"dataset_{w}", "N/A") for w in args_config["weather_types"]}
        f.write(f"Dataset roots:   rain={ds_roots.get('rain', 'N/A')}, snow={ds_roots.get('snow', 'N/A')}, haze={ds_roots.get('haze', 'N/A')}\n")
        f.write(f"Splits:          {args_config['splits']}\n")
        f.write(f"Weather types:   {args_config['weather_types']}\n")
        f.write(f"Resolution:      {args_config['resolution']}\n")
        f.write(f"Inference steps: {args_config['num_inference_steps']}\n")
        f.write(f"Guidance scale:  {args_config['guidance_scale']}\n")
        f.write(f"Use prompt:      {args_config.get('use_prompt', False)}\n")
        f.write(f"LPIPS backbone:  {lpips_net}\n")
        f.write(f"FID enabled:     {enable_fid}\n")

        # ===== 各子数据集指标 =====
        f.write("\n" + "-" * 90 + "\n")
        f.write("Per-Subdataset Metrics:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Subdataset':<28} {'Weather':<8} {'N':>5} {'PSNR (dB)':>10} {'SSIM':>10} {'LPIPS':>10} {'FID':>10} {'AvgTime(s)':>12}\n")
        f.write("-" * 90 + "\n")
        # 按 weather 顺序, 同一 weather 内按 sub_name 排序
        for weather in args_config["weather_types"]:
            for sub_name in by_weather.get(weather, []):
                m = sub_metrics[sub_name]
                f.write(f"{sub_name:<28} {weather:<8} {m['n']:>5} {m['psnr']:>10.4f} {m['ssim']:>10.4f} "
                        f"{_fmt(m['lpips']):>10} {_fmt(m.get('fid', float('nan'))):>10} {m['avg_time']:>12.2f}\n")

        # ===== 各 weather 聚合指标 =====
        f.write("\n" + "-" * 90 + "\n")
        f.write("Per-Weather Aggregated Metrics:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Weather':<12} {'N':>5} {'PSNR (dB)':>10} {'SSIM':>10} {'LPIPS':>10} {'FID':>10} {'AvgTime(s)':>12}\n")
        f.write("-" * 90 + "\n")
        for weather in args_config["weather_types"]:
            if weather not in weather_metrics:
                continue
            m = weather_metrics[weather]
            f.write(f"{weather:<12} {m['n']:>5} {m['psnr']:>10.4f} {m['ssim']:>10.4f} "
                    f"{_fmt(m['lpips']):>10} {_fmt(m.get('fid', float('nan'))):>10} {m['avg_time']:>12.2f}\n")

        # ===== 总体指标 =====
        f.write("-" * 90 + "\n")
        avg_psnr = sum(all_psnrs) / len(all_psnrs) if all_psnrs else 0.0
        avg_ssim = sum(all_ssims) / len(all_ssims) if all_ssims else 0.0
        avg_lpips = sum(all_lpipss) / len(all_lpipss) if all_lpipss else float("nan")
        f.write(f"{'ALL':<12} {total_n:>5} {avg_psnr:>10.4f} {avg_ssim:>10.4f} "
                f"{_fmt(avg_lpips):>10} {_fmt(overall_fid):>10} {'-':>12}\n")
        f.write("=" * 90 + "\n")

    # ===== 终端打印汇总 =====
    print("\n" + "=" * 90)
    print("Per-Subdataset Metrics:")
    print("-" * 90)
    print(f"{'Subdataset':<28} {'Weather':<8} {'N':>5} {'PSNR (dB)':>10} {'SSIM':>10} {'LPIPS':>10} {'FID':>10}")
    print("-" * 90)
    for weather in args_config["weather_types"]:
        for sub_name in by_weather.get(weather, []):
            m = sub_metrics[sub_name]
            print(f"{sub_name:<28} {weather:<8} {m['n']:>5} {m['psnr']:>10.4f} {m['ssim']:>10.4f} "
                  f"{_fmt(m['lpips']):>10} {_fmt(m.get('fid', float('nan'))):>10}")

    print("\n" + "=" * 90)
    print("Per-Weather Aggregated Metrics:")
    print("-" * 90)
    print(f"{'Weather':<12} {'N':>5} {'PSNR (dB)':>10} {'SSIM':>10} {'LPIPS':>10} {'FID':>10}")
    print("-" * 90)
    for weather in args_config["weather_types"]:
        if weather not in weather_metrics:
            continue
        m = weather_metrics[weather]
        print(f"{weather:<12} {m['n']:>5} {m['psnr']:>10.4f} {m['ssim']:>10.4f} "
              f"{_fmt(m['lpips']):>10} {_fmt(m.get('fid', float('nan'))):>10}")
    print("-" * 90)
    print(f"{'ALL':<12} {total_n:>5} {avg_psnr:>10.4f} {avg_ssim:>10.4f} "
          f"{_fmt(avg_lpips):>10} {_fmt(overall_fid):>10}")
    print("=" * 90)
    print(f"\n[eval] 评估完成, 结果保存到: {eval_root}")
    print(f"[eval] 汇总指标: {summary_path}")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)