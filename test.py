'''
 * ControlNet-SR 推理脚本 (支持单张图 / 目录)
 * Modified from diffusers / SeeSR
 *
 * 主要功能:
 *   1. 单张图或目录批量推理 (通过 --input 传入路径)
 *   2. 显式 .eval() 所有模块, 关闭 gradient_checkpointing (推理加速 + 避免 train 模式残留)
 *   3. 打印每个关键模块的 dtype / 训练模式 (排查色彩偏差 / 精度问题)
 *   4. 可选 fp16 vs fp32 对比推理 (锁定精度对色彩的影响)
 *   5. 可选色彩统计: 输出 pred / lq 的 mean / std / per-channel mean (排查偏暗问题)
 *
 * 用法:
 *   python test.py --config config/test.yaml
 *   python test.py --input path/to/img.png --controlnet_model_path ... --output_dir ...
'''

import os
os.environ['http_proxy'] = 'http://nbproxy.mlp.oppo.local:8888'
os.environ['https_proxy'] = 'http://nbproxy.mlp.oppo.local:8888'

import sys
sys.path.append(os.getcwd())

import argparse
import glob
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from wavelet_color_fix import wavelet_color_fix, adain_color_fix


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def collect_inputs(input_path: str) -> List[str]:
    """
    支持单张图或目录批量输入.

    - 单张: --input path/to/img.png      → [path/to/img.png]
    - 目录: --input path/to/dir/         → [dir/img1.png, dir/img2.png, ...] (按文件名排序)
    - glob: --input 'path/to/dir/*.png'  → 按 glob 展开
    """
    p = Path(input_path)
    if p.is_file() and is_image(p):
        return [str(p)]
    if p.is_dir():
        files = sorted(
            f for f in p.iterdir() if f.is_file() and is_image(f)
        )
        return [str(f) for f in files]
    # 当作 glob 处理
    matches = sorted(glob.glob(input_path))
    matches = [m for m in matches if is_image(Path(m))]
    return matches


def load_pipeline(args, device, dtype):
    """
    加载 SD + ControlNet 推理 pipeline.

    关键改动 (vs 旧版):
      - 显式 .eval() 所有模块
      - 显式 disable_gradient_checkpointing (避免 forward 慢 2x)
      - 打印每个模块的 dtype / training flag (色彩偏差排查)
    """
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_path, subfolder="controlnet")

    # ===== 冻结 + 显式 eval (避免 train 模式残留) =====
    for m in (vae, text_encoder, unet, controlnet):
        m.requires_grad_(False)
        m.eval()

    # ===== 显式关闭 gradient_checkpointing (推理加速) =====
    if hasattr(controlnet, "disable_gradient_checkpointing"):
        controlnet.disable_gradient_checkpointing()
    if hasattr(unet, "disable_gradient_checkpointing"):
        unet.disable_gradient_checkpointing()

    # ===== 打印模型 dtype / training 标志 (排查精度 / 模式问题) =====
    if args.print_model_info:
        print("=" * 60)
        print("[Model Info]")
        for name, m in (("vae", vae), ("text_encoder", text_encoder),
                        ("unet", unet), ("controlnet", controlnet)):
            print(f"  {name:14s}  dtype={m.dtype}  training={m.training}  "
                  f"gradient_checkpointing={getattr(m, 'gradient_checkpointing', 'N/A')}")
        print("=" * 60)

    # ===== xformers 加速 (可选) =====
    if args.enable_xformers:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            print("[warn] xformers 不可用, 跳过")

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    return pipeline


def color_stats(img: Image.Image, label: str) -> dict:
    """
    统计 PIL 图的色彩信息 (排查偏暗问题).

    返回: mean (0~255), std, per-channel mean (R,G,B), 亮度 (HSV-V 均值)
    """
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    ch_mean = arr.reshape(-1, 3).mean(axis=0).tolist()
    hsv = np.asarray(img.convert("HSV"))
    v_mean = float(hsv[..., 2].mean())
    stats = {
        "label": label,
        "mean": mean,
        "std": std,
        "ch_mean_R": ch_mean[0],
        "ch_mean_G": ch_mean[1],
        "ch_mean_B": ch_mean[2],
        "hsv_V": v_mean,
    }
    if label:
        print(f"  [{label}]  mean={mean:6.2f}  std={std:6.2f}  "
              f"R/G/B={ch_mean[0]:5.1f}/{ch_mean[1]:5.1f}/{ch_mean[2]:5.1f}  "
              f"V(HSV)={v_mean:5.1f}")
    return stats


def maybe_color_fix(image: Image.Image, lq: Image.Image, method: str) -> Image.Image:
    if method == "wavelet":
        return wavelet_color_fix(image, lq)
    if method == "adain":
        return adain_color_fix(image, lq)
    return image


def run_single_inference(
    pipeline,
    lq_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    *,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    generator: torch.Generator,
    autocast_dtype: Optional[torch.dtype],
):
    """
    一次推理, 支持指定 autocast dtype (fp32 时即关闭混合精度).

    autocast_dtype:
        torch.float32 → 用 no-op context (完全 fp32)
        torch.float16 → autocast to fp16 (测 fp16 色彩影响)
        None         → 不包 autocast (用 pipeline 的 dtype)
    """
    if autocast_dtype == torch.float32:
        # 强制 fp32: 用 no_grad + 不开 autocast
        ctx = torch.no_grad()
    elif autocast_dtype is not None:
        ctx = torch.autocast("cuda", dtype=autocast_dtype)
    else:
        ctx = torch.autocast("cuda")

    with ctx:
        out = pipeline(
            prompt=prompt,
            image=lq_image,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
        ).images[0]
    return out


def run_full_matrix(args):
    """
    自动跑完整矩阵: 2 models × 2 dtypes = 4 次推理.

    用法:
        python test.py --run_full_matrix \
            --controlnet_model_path /path/to/SFT \
            --controlnet_model_path_b /path/to/DPO \
            --model_a_label SFT --model_b_label DPO \
            --output_dir exp/matrix

    输出:
        exp/matrix/SFT_fp32/*.png
        exp/matrix/SFT_fp16/*.png
        exp/matrix/DPO_fp32/*.png
        exp/matrix/DPO_fp16/*.png

    然后打印 4 列对比表, 区分"训练影响" vs "精度影响":
        - Δ_mean(fp16-fp32) < 3: fp16 精度对色彩影响小
        - Δ_mean(SFT-DPO) > 10: DPO 训练本身导致偏暗
    """
    import argparse as _ap

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    label_a = getattr(args, "model_a_label", None) or "model_A"
    label_b = getattr(args, "model_b_label", None) or "model_B"

    if not args.controlnet_model_path_b:
        raise ValueError("--run_full_matrix 需要同时传 --controlnet_model_path_b")

    runs = [
        (args.controlnet_model_path,    label_a, "fp32"),
        (args.controlnet_model_path,    label_a, "fp16"),
        (args.controlnet_model_path_b,  label_b, "fp32"),
        (args.controlnet_model_path_b,  label_b, "fp16"),
    ]

    print("=" * 70)
    print("[run_full_matrix] 完整矩阵: 2 模型 × 2 精度 = 4 次推理")
    print(f"  A = {label_a}  ({args.controlnet_model_path})")
    print(f"  B = {label_b}  ({args.controlnet_model_path_b})")
    print("=" * 70)

    for ckpt, label, dtype_str in runs:
        args_x = _ap.Namespace(**vars(args))
        args_x.run_full_matrix = False
        args_x.compare_two_models = False
        args_x.compare_dtypes = False
        args_x.controlnet_model_path = ckpt
        args_x.mixed_precision = dtype_str
        args_x.output_dir = str(out_root / f"{label}_{dtype_str}")
        print(f"\n>>> [{label} / {dtype_str}]  ckpt={ckpt}")
        main(args_x)

    # ===== 收集 4 个目录的色彩统计 =====
    print("\n" + "=" * 70)
    print("[色彩对比 - 完整矩阵]")
    print("=" * 70)

    img_exts = {".png", ".jpg", ".jpeg"}
    matrix_stats: dict = {}
    for _, label, dtype_str in runs:
        d = out_root / f"{label}_{dtype_str}"
        files = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in img_exts and p.is_file())
        stats_list = []
        for p in files:
            stats_list.append(color_stats(Image.open(p), ""))
        if stats_list:
            avg = {
                "mean": np.mean([s["mean"] for s in stats_list]),
                "std": np.mean([s["std"] for s in stats_list]),
                "R": np.mean([s["ch_mean_R"] for s in stats_list]),
                "G": np.mean([s["ch_mean_G"] for s in stats_list]),
                "B": np.mean([s["ch_mean_B"] for s in stats_list]),
                "V": np.mean([s["hsv_V"] for s in stats_list]),
            }
        else:
            avg = {"mean": float("nan"), "std": 0,
                   "R": 0, "G": 0, "B": 0, "V": 0}
        matrix_stats[(label, dtype_str)] = avg

    # 打印 4 行对比表
    print(f"  {'配置':<25s}  {'mean':>7s}  {'std':>6s}  "
          f"{'R':>6s}  {'G':>6s}  {'B':>6s}  {'V(HSV)':>7s}")
    print("  " + "-" * 70)
    for _, label, dtype_str in runs:
        s = matrix_stats[(label, dtype_str)]
        print(f"  {label + ' ' + dtype_str:<25s}  "
              f"{s['mean']:7.2f}  {s['std']:6.2f}  "
              f"{s['R']:6.1f}  {s['G']:6.1f}  {s['B']:6.1f}  {s['V']:7.2f}")

    # ===== 关键诊断 =====
    print("\n  [诊断] 拆解三组影响:")
    print("  " + "-" * 70)

    s_a32 = matrix_stats[(label_a, "fp32")]
    s_a16 = matrix_stats[(label_a, "fp16")]
    s_b32 = matrix_stats[(label_b, "fp32")]
    s_b16 = matrix_stats[(label_b, "fp16")]

    # 1) 精度影响 (单模型 fp16 vs fp32): A 模型 fp32 - fp16
    d_precision_a = s_a32["mean"] - s_a16["mean"]
    d_precision_b = s_b32["mean"] - s_b16["mean"]
    print(f"  [{label_a}] fp32 - fp16 = {d_precision_a:+.2f}   "
          f"(精度对 {label_a} 的色彩影响)")
    print(f"  [{label_b}] fp32 - fp16 = {d_precision_b:+.2f}   "
          f"(精度对 {label_b} 的色彩影响)")

    # 2) 训练影响 (A vs B, 同精度): fp32 下 A - B
    d_train_fp32 = s_a32["mean"] - s_b32["mean"]
    d_train_fp16 = s_a16["mean"] - s_b16["mean"]
    print(f"  [fp32] {label_a} - {label_b} = {d_train_fp32:+.2f}   "
          f"({label_b} 相对 {label_a} 的整体偏移)")
    print(f"  [fp16] {label_a} - {label_b} = {d_train_fp16:+.2f}")

    # 3) 结论
    print("\n  [结论]")
    if abs(d_precision_a) < 3 and abs(d_precision_b) < 3:
        print(f"    ✓ fp16/fp32 精度对色彩影响 < 3 → 排除精度问题")
    else:
        print(f"    ✗ fp16/fp32 精度对色彩影响 {max(abs(d_precision_a), abs(d_precision_b)):.1f} "
              f"→ 需用 fp32 推理 (--mixed_precision fp32)")

    if d_train_fp32 > 10:
        print(f"    ✗ {label_b} 相对 {label_a} 整体偏暗 {d_train_fp32:.1f} 像素值")
        print(f"      可能原因:")
        print(f"      - {label_b} 训练样本过少 / 偏向低亮度 GT")
        print(f"      - reward 选择偏向保守 / 拉向均值区域 (PSNR 偏均值)")
        print(f"      - 学习率过高 / 训练步数过多")
    elif d_train_fp32 < -10:
        print(f"    ! {label_b} 相对 {label_a} 偏亮 {-d_train_fp32:.1f}")
    else:
        print(f"    ✓ {label_a}/{label_b} 整体亮度差异 < 10 → 排除训练问题")
    print("=" * 70)


def run_two_model_comparison(args):
    """
    自动对比两个 controlnet (e.g. SFT vs DPO) 在同一组 LQ 上的差异.

    内部跑两次 main() (checkpoint A / B), 收集每张图的色彩统计,
    然后打印 side-by-side 对比表 (mean / V_HSV / per-channel).
    """
    import argparse as _ap

    print("=" * 70)
    print("[compare_two_models] 自动跑两个 checkpoint 并对比")
    print(f"  A (--controlnet_model_path):    {args.controlnet_model_path}")
    print(f"  B (--controlnet_model_path_b):  {args.controlnet_model_path_b}")
    print("=" * 70)

    # ===== 准备 A 模型 =====
    out_root = Path(args.output_dir)
    out_a = out_root / "A"
    out_b = out_root / "B"

    # ===== 跑 A =====
    args_a = _ap.Namespace(**vars(args))
    args_a.compare_two_models = False
    args_a.controlnet_model_path = args.controlnet_model_path
    args_a.output_dir = str(out_a)
    print("\n>>> 跑 A 模型")
    main(args_a)

    # ===== 跑 B =====
    args_b = _ap.Namespace(**vars(args))
    args_b.compare_two_models = False
    args_b.controlnet_model_path = args.controlnet_model_path_b
    args_b.output_dir = str(out_b)
    print("\n>>> 跑 B 模型")
    main(args_b)

    # ===== 收集并对比 =====
    print("\n" + "=" * 70)
    print("[色彩对比结果] (A - B = 差值, 正值 = A 比 B 更亮)")
    print("=" * 70)
    img_exts = {".png", ".jpg", ".jpeg"}
    files_a = sorted(p for p in out_a.iterdir() if p.suffix.lower() in img_exts and p.is_file())
    files_b = sorted(p for p in out_b.iterdir() if p.suffix.lower() in img_exts and p.is_file())
    names_b = {p.name: p for p in files_b}

    diff_means = []
    diff_vs = []
    for pa in files_a:
        if pa.name not in names_b:
            continue
        sa = color_stats(Image.open(pa), "")
        sb = color_stats(Image.open(names_b[pa.name]), "")
        d_mean = sa["mean"] - sb["mean"]
        d_V = sa["hsv_V"] - sb["hsv_V"]
        diff_means.append(d_mean)
        diff_vs.append(d_V)
        print(f"  {pa.name:40s}  "
              f"A.mean={sa['mean']:6.2f}  B.mean={sb['mean']:6.2f}  Δ={d_mean:+6.2f}  "
              f"| A.V={sa['hsv_V']:6.2f}  B.V={sb['hsv_V']:6.2f}  Δ={d_V:+6.2f}")

    if diff_means:
        print("-" * 70)
        print(f"  [avg]  Δmean = {np.mean(diff_means):+.2f}   ΔV(HSV) = {np.mean(diff_vs):+.2f}")
        if np.mean(diff_means) < -3:
            print(f"  [诊断] B 模型整体比 A 暗 {abs(np.mean(diff_means)):.1f} (像素均值)")
            print("         可能原因: 训练数据偏暗 / DPO reward 偏向保守 / 数值漂移")
        elif np.mean(diff_means) > 3:
            print(f"  [诊断] B 模型整体比 A 亮 {np.mean(diff_means):.1f} (像素均值)")
        else:
            print("  [诊断] A / B 整体亮度差异不显著 (< 3)")
    print("=" * 70)


def main(args):
    # ===== 加载 yaml 配置 =====
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # yaml 始终覆盖 argparse 默认值.
        # 注意: argparse 中 action="store_true" 的默认是 False (不是 None),
        #       原 is None 判断会让 boolean flag 永远不能被 yaml 覆盖.
        #       (例如 run_full_matrix: true 在 yaml 里会被忽略)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                setattr(args, key, value)
        # 打印关键诊断开关的最终值, 便于排查
        if args.print_model_info:
            print("=" * 60)
            print("[YAML Config 加载结果]")
            for k in ("run_full_matrix", "compare_two_models", "compare_dtypes",
                     "print_color_stats", "print_model_info", "mixed_precision",
                     "controlnet_model_path", "controlnet_model_path_b",
                     "model_a_label", "model_b_label"):
                if hasattr(args, k):
                    print(f"  {k} = {getattr(args, k)!r}")
            print("=" * 60)

    # ===== 双模型对比模式 (放在最前, 内部递归调用 main) =====
    if args.run_full_matrix:
        run_full_matrix(args)
        return
    if args.compare_two_models:
        if not args.controlnet_model_path_b:
            raise ValueError("--compare_two_models 需要同时传 --controlnet_model_path_b")
        run_two_model_comparison(args)
        return

    # ===== device / dtype =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    weight_dtype = dtype_map[args.mixed_precision]
    print(f"[setup] device={device}, dtype={weight_dtype}")

    # ===== 收集输入 (单张 / 目录 / glob) =====
    image_names = collect_inputs(args.image_path)
    if not image_names:
        raise FileNotFoundError(f"未在 {args.image_path} 找到任何图片")
    print(f"[setup] 共找到 {len(image_names)} 张图片")

    # ===== 输出目录 =====
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.save_prompts:
        (out_root / "txt").mkdir(parents=True, exist_ok=True)

    # ===== fp16 vs fp32 对比模式 =====
    # 若 compare_dtypes=True, 对每张图同时跑 fp32 和 fp16, 输出到 sample00 (fp32) / sample01 (fp16)
    # 用 args.compare_dtype_fp32 / compare_dtype_fp16 字段
    if args.compare_dtypes:
        args.sample_times = 2  # 自动覆盖
        # 重命名 sample00 -> fp32, sample01 -> fp16
        sample_dir_names = ["fp32", "fp16"]
    else:
        sample_dir_names = [f"sample{str(i).zfill(2)}" for i in range(args.sample_times)]

    for sd in sample_dir_names:
        (out_root / sd).mkdir(parents=True, exist_ok=True)

    # ===== 加载 pipeline (按主 dtype 加载) =====
    pipeline = load_pipeline(args, device, weight_dtype)

    # ===== prompt 配置 (默认空) =====
    added_prompt = args.added_prompt
    negative_prompt = args.negative_prompt

    # ===== 推理循环 =====
    time_list = []
    for image_idx, image_name in enumerate(image_names):
        print(f"=" * 60)
        print(f"[{image_idx + 1}/{len(image_names)}] {image_name}")
        lq_pil = Image.open(image_name).convert("RGB")

        # 几何预处理: 升采样到目标分辨率, 长宽对齐到 8 的倍数
        ori_w, ori_h = lq_pil.size
        rscale = args.upscale
        if ori_w < args.process_size // rscale or ori_h < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_w, ori_h)
            lq_pil = lq_pil.resize(
                (int(scale * ori_w), int(scale * ori_h)), Image.BICUBIC
            )
        # 升采样 rscale 倍 (用 BICUBIC 显式指定, 避免默认 NEAREST)
        lq_pil = lq_pil.resize(
            (lq_pil.size[0] * rscale, lq_pil.size[1] * rscale), Image.BICUBIC
        )
        # 对齐到 8 的倍数
        lq_pil = lq_pil.resize(
            (lq_pil.size[0] // 8 * 8, lq_pil.size[1] // 8 * 8), Image.BICUBIC
        )
        width, height = lq_pil.size
        print(f"  LQ input size: {height}x{width}")

        # prompt: 占位, 见后面预留的 RAM 接入点
        prompt = added_prompt

        # 统计 LQ 色彩
        if args.print_color_stats:
            print("  [色彩统计]")
            color_stats(lq_pil, "LQ(in)")

        # ===== 多 sample 循环 =====
        for sample_idx in range(args.sample_times):
            generator = torch.Generator(device=device)
            if args.seed is not None:
                generator.manual_seed(args.seed + sample_idx)

            t0 = time.time()
            # 根据 compare_dtypes 决定每个 sample 用的 autocast dtype
            if args.compare_dtypes:
                # sample 0 -> fp32, sample 1 -> fp16
                ac_dtype = torch.float32 if sample_idx == 0 else torch.float16
            else:
                ac_dtype = None  # 让 pipeline 自己决定 (默认 fp16 if pipeline is fp16)

            pred = run_single_inference(
                pipeline,
                lq_pil,
                prompt,
                negative_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=height,
                width=width,
                generator=generator,
                autocast_dtype=ac_dtype,
            )
            dt = time.time() - t0
            time_list.append(dt)

            # 色彩统计 (pred)
            if args.print_color_stats:
                color_stats(pred, f"PRED(sample{sample_idx})")

            # 色彩校正
            pred = maybe_color_fix(pred, lq_pil, args.align_method)

            # 还原到原始尺寸 * upscale
            pred = pred.resize((ori_w * rscale, ori_h * rscale), Image.BICUBIC)

            # 保存
            name = Path(image_name).stem
            out_path = out_root / sample_dir_names[sample_idx] / f"{name}.png"
            pred.save(out_path)
            print(f"  saved -> {out_path}  ({dt:.2f}s)")

            if args.save_prompts:
                (out_root / "txt" / f"{name}.txt").write_text(prompt, encoding="utf-8")

    # ===== 总结 =====
    print("=" * 60)
    if time_list:
        print(f"[done] avg time: {np.mean(time_list):.3f}s | "
              f"min: {np.min(time_list):.3f}s | max: {np.max(time_list):.3f}s | "
              f"total: {np.sum(time_list):.2f}s")

    # ===== 若做了 fp32 vs fp16 对比, 输出色彩差异总结 =====
    if args.compare_dtypes and args.print_color_stats:
        print("=" * 60)
        print("[fp32 vs fp16 色彩差异总结] (mean / V_HSV)")
        # 此处只是提示, 详细差异请看 saved images 实际对比
        print("  请对比 output/fp32/*.png 与 output/fp16/*.png 像素均值")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ControlNet-SR 推理 (支持单图 / 目录 / 精度对比)")
    # --- 模型路径 ---
    parser.add_argument("--controlnet_model_path", type=str, default=None)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    # --- 输入输出 ---
    parser.add_argument("--image_path", type=str, default=None,
                        help="单张图片路径 / 目录路径 / glob pattern")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_prompts", action="store_true")
    # --- 推理参数 ---
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp32", "fp16", "bf16"],
                        help="no/fp32 = 完全 fp32 (慢但精确); fp16 = 混合精度 (快)")
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--align_method", type=str,
                        choices=["wavelet", "adain", "nofix"], default="nofix")
    parser.add_argument("--added_prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    # --- RAM (可选, 暂未启用) ---
    parser.add_argument("--ram_ft_path", type=str, default=None)
    # --- 速度 / 精度 ---
    parser.add_argument("--enable_xformers", action="store_true")
    # --- 诊断开关 (本次重点) ---
    parser.add_argument("--print_model_info", action="store_true",
                        help="打印每个模块 dtype / training / gradient_checkpointing")
    parser.add_argument("--print_color_stats", action="store_true",
                        help="打印 pred / lq 的色彩统计 (mean / per-channel)")
    parser.add_argument("--compare_dtypes", action="store_true",
                        help="对每张图同时跑 fp32 和 fp16, 输出到 fp32/ 和 fp16/ 目录")
    parser.add_argument("--compare_two_models", action="store_true",
                        help="对比两个 controlnet (用 --controlnet_model_path 与 --controlnet_model_path_b), "
                             "自动跑两次并打印色彩差异总结")
    parser.add_argument("--controlnet_model_path_b", type=str, default=None,
                        help="(仅 --compare_two_models 生效) 第二个 controlnet 路径")
    parser.add_argument("--run_full_matrix", action="store_true",
                        help="自动跑完整矩阵: 2 模型 × 2 精度 (fp16/fp32) = 4 次推理, "
                             "最后打印色彩对比表 + 拆解训练影响 vs 精度影响")
    parser.add_argument("--model_a_label", type=str, default="A",
                        help="--run_full_matrix 时第一个模型的标签 (默认 A)")
    parser.add_argument("--model_b_label", type=str, default="B",
                        help="--run_full_matrix 时第二个模型的标签 (默认 B)")
    # --- yaml ---
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    args = parser.parse_args()
    main(args)