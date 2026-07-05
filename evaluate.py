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

from dataloaders.paired_dataset import DEFAULT_WEATHER_PROMPTS, PairedCaptionDataset
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
    复用 PairedCaptionDataset 加载 test 集, 但不实际使用 __getitem__.
    我们只需要它扫到的样本路径列表 (samples).
    支持三种独立数据集路径: dataset_rain, dataset_snow, dataset_haze.
    """
    class _EmptyTokenizer:
        model_max_length = 77
        def __call__(self, text, **kwargs):
            class R:
                input_ids = [[0] * 77]
            return R()

    all_samples = []
    for weather in args_config["weather_types"]:
        root_key = f"dataset_{weather}"
        dataset_root = args_config.get(root_key)
        if not dataset_root:
            print(f"[warn] {root_key} 未设置, 跳过 {weather}")
            continue

        ds = PairedCaptionDataset(
            dataset_root=dataset_root,
            weather_types=[weather],
            splits=args_config["splits"],
            tokenizer=_EmptyTokenizer(),
            null_text_ratio=0.0,
            use_prompt=False,
            resolution=args_config["resolution"],
        )
        all_samples.extend(ds.samples)

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

    # 按 weather 分组
    by_weather: Dict[str, List] = defaultdict(list)
    for gt_path, lq_path, weather in samples:
        by_weather[weather].append((gt_path, lq_path))

    # ===== 评估采样数: 控制用于计算指标 (PSNR/SSIM/LPIPS) 的样本数 =====
    # 默认使用全部 test 集 (保证指标统计准确)
    default_max = args_config.get("max_samples_per_weather", 0)
    for w in by_weather:
        if default_max and default_max > 0 and len(by_weather[w]) > default_max:
            random.shuffle(by_weather[w])
            by_weather[w] = by_weather[w][:default_max]
            print(f"[eval] {w}: 评估采样截断为 {default_max} (用于指标计算)")
        else:
            print(f"[eval] {w}: 评估使用全部 {len(by_weather[w])} 样本")

    # ===== 可视化保存数: 控制保存多少组 pred/lq/gt 图 =====
    # 与评估数解耦: 无论评估多少张, 这里控制保存多少张
    save_counts: Dict[str, int] = {}
    save_predictions_global = args_config.get("save_predictions", True)
    if save_predictions_global:
        for w in by_weather:
            v = args_config.get(f"{w}_num", -1)
            if v is None:
                v = -1
            if v == 0:
                save_counts[w] = 0
            elif v < 0:
                # -1 表示全部保存
                save_counts[w] = len(by_weather[w])
            else:
                # 正数表示保存前 N 张
                save_counts[w] = min(v, len(by_weather[w]))
            print(f"[eval] {w}: 可视化保存 {save_counts[w]} 张")
    else:
        for w in by_weather:
            save_counts[w] = 0
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
    # 存储每张图的指标: per_image_results[weather] = [(name, psnr, ssim, lpips), ...]
    per_image_results: Dict[str, List] = defaultdict(list)
    # 收集预测与 GT 张量, 用于计算 FID (每个 weather 一组)
    fid_preds: Dict[str, List[torch.Tensor]] = defaultdict(list)
    fid_gts: Dict[str, List[torch.Tensor]] = defaultdict(list)
    # 是否启用 FID
    enable_fid = args_config.get("enable_fid", True)

    # 固定种子以便复现
    if args_config.get("seed") is not None:
        random.seed(args_config["seed"])
        torch.manual_seed(args_config["seed"])

    total_samples = sum(len(v) for v in by_weather.values())
    pbar = tqdm(total=total_samples, desc="Eval")

    lpips_net = args_config.get("lpips_net", "alex")

    for weather in args_config["weather_types"]:
        if weather not in by_weather:
            print(f"[eval] 跳过 {weather}: 没有样本")
            continue

        weather_dir = eval_root / weather
        weather_dir.mkdir(parents=True, exist_ok=True)
        n_to_save = save_counts.get(weather, 0)

        for sample_idx, (gt_path, lq_path) in enumerate(by_weather[weather]):
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
            per_image_results[weather].append((stem, p, s, l, infer_time))

            # 收集用于 FID 计算的张量 (CPU 张量, 避免长时间占 GPU 显存)
            if enable_fid:
                fid_preds[weather].append(pred_tensor.detach().cpu())
                fid_gts[weather].append(gt_tensor.detach().cpu())

            # 保存图片 (受 save_counts 控制, 与评估数解耦)
            if sample_idx < n_to_save:
                pred_pil.save(weather_dir / f"{sample_idx:03d}_{stem}_pred.png")
                lq_pil.save(weather_dir / f"{sample_idx:03d}_{stem}_lq.png")
                transforms.ToPILImage()(gt_img).save(weather_dir / f"{sample_idx:03d}_{stem}_gt.png")

            pbar.set_postfix(weather=weather, psnr=f"{p:.2f}", ssim=f"{s:.4f}", lpips=f"{l:.4f}")
            pbar.update(1)

    pbar.close()

    # ===== 汇总每个 weather 的指标 =====
    weather_metrics = {}
    for weather, items in per_image_results.items():
        psnrs = [x[1] for x in items]
        ssims = [x[2] for x in items]
        lpipss = [x[3] for x in items if not (isinstance(x[3], float) and x[3] != x[3])]  # 过滤 NaN
        times = [x[4] for x in items]
        weather_metrics[weather] = {
            "n": len(items),
            "psnr": sum(psnrs) / len(psnrs) if psnrs else 0.0,
            "ssim": sum(ssims) / len(ssims) if ssims else 0.0,
            "lpips": sum(lpipss) / len(lpipss) if lpipss else float("nan"),
            "avg_time": sum(times) / len(times) if times else 0.0,
        }

    # ===== 计算 FID (按 weather 汇总) =====
    if enable_fid:
        print("\n[FID] 开始计算 FID...")
        for weather in args_config["weather_types"]:
            if weather not in fid_preds or len(fid_preds[weather]) == 0:
                continue
            try:
                fid_val = calc_fid(fid_preds[weather], fid_gts[weather])
                weather_metrics[weather]["fid"] = fid_val
                print(f"  [FID] {weather}: {fid_val:.4f} (N={len(fid_preds[weather])})")
            except Exception as e:
                print(f"  [FID] {weather} 计算失败: {e}")
                weather_metrics[weather]["fid"] = float("nan")
        # 总体 FID (所有 weather 合并)
        all_preds = []
        all_gts = []
        for w in fid_preds:
            all_preds.extend(fid_preds[w])
            all_gts.extend(fid_gts[w])
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

    # 写每个 weather 的 per_image 指标文件
    for weather, items in per_image_results.items():
        per_img_path = eval_root / weather / "per_image_metrics.txt"
        with open(per_img_path, "w", encoding="utf-8") as f:
            f.write(f"# Per-image metrics for weather={weather}\n")
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

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ControlNet Multi-Weather Image Restoration - Evaluation Report\n")
        f.write("=" * 70 + "\n")
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
        f.write("-" * 70 + "\n")
        f.write(f"{'Weather':<12} {'N':>5} {'PSNR (dB)':>10} {'SSIM':>10} {'LPIPS':>10} {'FID':>10} {'AvgTime(s)':>12}\n")
        f.write("-" * 70 + "\n")

        for weather in args_config["weather_types"]:
            if weather not in weather_metrics:
                continue
            m = weather_metrics[weather]
            lpips_str = f"{m['lpips']:.4f}" if m['lpips'] == m['lpips'] else "  N/A  "
            fid_str = f"{m.get('fid', float('nan')):.4f}" if m.get('fid', float('nan')) == m.get('fid', float('nan')) else "  N/A  "
            f.write(f"{weather:<12} {m['n']:>5} {m['psnr']:>10.4f} {m['ssim']:>10.4f} "
                    f"{lpips_str:>10} {fid_str:>10} {m['avg_time']:>12.2f}\n")

        f.write("-" * 70 + "\n")
        avg_psnr = sum(all_psnrs) / len(all_psnrs) if all_psnrs else 0.0
        avg_ssim = sum(all_ssims) / len(all_ssims) if all_ssims else 0.0
        avg_lpips = sum(all_lpipss) / len(all_lpipss) if all_lpipss else float("nan")
        lpips_str = f"{avg_lpips:.4f}" if avg_lpips == avg_lpips else "  N/A  "
        fid_overall_str = f"{overall_fid:.4f}" if overall_fid == overall_fid else "  N/A  "
        f.write(f"{'ALL':<12} {total_n:>5} {avg_psnr:>10.4f} {avg_ssim:>10.4f} "
                f"{lpips_str:>10} {fid_overall_str:>10} {'-':>12}\n")
        f.write("=" * 70 + "\n")

    # 终端打印汇总
    print("\n" + "=" * 78)
    print(f"{'Weather':<12} {'N':>5} {'PSNR (dB)':>10} {'SSIM':>10} {'LPIPS':>10} {'FID':>10}")
    print("-" * 78)
    for weather in args_config["weather_types"]:
        if weather not in weather_metrics:
            continue
        m = weather_metrics[weather]
        lpips_str = f"{m['lpips']:.4f}" if m['lpips'] == m['lpips'] else "  N/A  "
        fid_str = f"{m.get('fid', float('nan')):.4f}" if m.get('fid', float('nan')) == m.get('fid', float('nan')) else "  N/A  "
        print(f"{weather:<12} {m['n']:>5} {m['psnr']:>10.4f} {m['ssim']:>10.4f} {lpips_str:>10} {fid_str:>10}")
    print("-" * 78)
    avg_lpips_str = f"{avg_lpips:.4f}" if avg_lpips == avg_lpips else "  N/A  "
    fid_overall_str = f"{overall_fid:.4f}" if overall_fid == overall_fid else "  N/A  "
    print(f"{'ALL':<12} {total_n:>5} {avg_psnr:>10.4f} {avg_ssim:>10.4f} {avg_lpips_str:>10} {fid_overall_str:>10}")
    print("=" * 78)
    print(f"\n[eval] 评估完成, 结果保存到: {eval_root}")
    print(f"[eval] 汇总指标: {summary_path}")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)