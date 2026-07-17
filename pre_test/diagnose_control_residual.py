"""
diagnose_control_residual.py
============================================================
目的
----
对训练好的 ControlNetSR 权重做三项快速排查, 验证 "C2F/zero_conv 控制残差
是否真的学到有效退化特征", 全部不重新训练, 全部跑在已有 checkpoint 上:

  [1] zero_conv 权重范数检查
      加载模型后, 遍历所有 zero_conv 层, 打印 L2 范数. 训练数万步后
      范数仍接近 0, 证明控制分支未参与优化 / SD 完全不受 LQ 约束.

  [2] 控制残差前向数值检查
      注册 forward hook 抓取所有 zero_conv 在一次 ControlNet 前向中的
      输出张量, 打印其 abs mean / abs max. 数值爆炸 / 几乎为 0 都属于
      异常, 正常区间 0.01~0.1.

  [3] 有无 C2F 控制消融对比
      同一张 LQ, 同一 pipeline, 同一 seed, 分别以
        controlnet_conditioning_scale = 1.0   (正常控制)
        controlnet_conditioning_scale = 0.0   (强制残差=0)
      跑两次推理, 对比 PSNR / SSIM / LPIPS 与可视化拼图. 差距 < 0.1 dB
      或画面无改善, 直接证明 C2F 完全失效.

用法
----
    # 使用默认配置
    python pre_test/diagnose_control_residual.py --config config/diagnose_control_residual.yaml

    # 命令行覆盖关键参数
    python pre_test/diagnose_control_residual.py \
        --controlnet_model_path ./experiment/ControlNetSR/checkpoint-50000 \
        --lq_image_path path/to/lq.png \
        --gt_image_path  path/to/gt.png \
        --mode all

    # 仅做权重范数检查 (不加载 SD / VAE, 最快)
    python pre_test/diagnose_control_residual.py --mode weights_only

    # 仅做消融 PSNR 对比
    python pre_test/diagnose_control_residual.py --mode ablation

输出
----
<pre_test>/output/<timestamp>_diag/
    ├── config_snapshot.yaml
    ├── summary.json                  # 机器可读汇总
    ├── summary.txt                   # 人类可读汇总 + 判读建议
    ├── weight_norms.txt              # [1] 各项权重范数
    ├── residual_stats.txt            # [2] 各项前向残差 abs mean / max
    ├── 000_<stem>_cond1.0_pred.png   # [3] 正常控制预测
    ├── 000_<stem>_cond0.0_pred.png   # [3] 强制残差=0 预测 (基线)
    ├── 000_<stem>_lq.png / gt.png    # 原图复制
    └── ablation_grid.png             # [3] LQ | cond=1.0 | cond=0.0 | GT 横排对比
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

# 让脚本独立运行时也能找到仓库内模块 (与 pre_test/test_randomness.py 同样套路)
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils.import_utils import is_xformers_available

from models.weather_restoration_controlnet import WeatherRestorationControlNet
from ramseesr.utils.metrics import (
    _get_lpips_model,
    lpips as calc_lpips,
    psnr as calc_psnr,
    ssim as calc_ssim,
)
from utils.evaluate import resolve_controlnet_path

# 与仓库其它脚本保持一致的图像后缀白名单
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _safe_print(text: str) -> None:
    """Windows GBK 控制台安全的 print, 失败时降级为 ASCII-only."""
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("ascii", errors="replace"))
        sys.stdout.buffer.write(b"\n")
        sys.stdout.flush()


# =============================================================================
# 默认配置 (与 config/diagnose_control_residual.yaml 同步)
# =============================================================================
DEFAULT_CONFIG: Dict = {
    "pretrained_model_name_or_path": "sd-research/stable-diffusion-2-base",
    "controlnet_model_path": "./experiment/ControlNetSR/checkpoint-50000",
    "lq_image_path": "experiment/eval/20260704_173616_eval/rain/000_rain-1701_lq.png",
    "gt_image_path": "experiment/eval/20260704_173616_eval/rain/000_rain-1701_gt.png",
    "output_dir": "./pre_test/output",
    "mode": "all",                         # {all, weights_only, residuals_only, ablation}
    "prompt": "",
    "negative_prompt": "dotted, noise, blur, lowres, smooth",
    "resolution": 512,
    "num_inference_steps": 20,
    "guidance_scale": 5.5,
    "mixed_precision": "fp16",
    "enable_xformers_memory_efficient_attention": False,
    "seed": 42,
    "lpips_net": "alex",
    # 判读阈值 (仅供 summary 给出提示, 不会终止流程)
    "weight_norm_min_healthy": 1e-4,
    "residual_abs_mean_min_healthy": 1e-3,
    "residual_abs_mean_max_healthy": 1.0,
    "ablation_psnr_drop_db": 0.1,
}


# =============================================================================
# 参数解析 + 配置合并
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ControlNetSR 控制残差三项快速排查 (权重范数 / 前向残差 / 消融 PSNR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/diagnose_control_residual.yaml",
        help="YAML 配置文件路径",
    )
    parser.add_argument("--controlnet_model_path", type=str, default=None)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--lq_image_path", type=str, default=None)
    parser.add_argument("--gt_image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["all", "weights_only", "residuals_only", "ablation"],
        help="运行模式: all=三项都做; weights_only=只查权重范数; "
             "residuals_only=只查前向残差; ablation=只做消融对比",
    )
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None,
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--lpips_net", type=str, default=None, choices=["alex", "vgg"])
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> Dict:
    """合并顺序: 默认值 < YAML < 命令行覆盖."""
    cfg: Dict = dict(DEFAULT_CONFIG)

    if args.config is not None and Path(args.config).is_file():
        with io.open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in yaml_cfg.items() if v is not None})

    overrides = {
        "controlnet_model_path": args.controlnet_model_path,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "lq_image_path": args.lq_image_path,
        "gt_image_path": args.gt_image_path,
        "output_dir": args.output_dir,
        "mode": args.mode,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "resolution": args.resolution,
        "seed": args.seed,
        "mixed_precision": args.mixed_precision,
        "lpips_net": args.lpips_net,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    if args.cpu:
        cfg["_force_cpu"] = True
    else:
        cfg["_force_cpu"] = False
    return cfg


def _load_controlnet_smart(cn_path: str) -> torch.nn.Module:
    """
    根据 <dir>/config.json 的 _class_name 自动选 ControlNet 类加载.

      - "ControlNetModel"             -> diffusers vanilla (读 .bin / .safetensors)
      - "WeatherRestorationControlNet" -> 项目自定义 (只读 .bin)
      - 其它/缺省: 走 vanilla
    """
    cfg_path = Path(cn_path) / "config.json"
    class_name = "ControlNetModel"
    if cfg_path.is_file():
        try:
            with io.open(cfg_path, "r", encoding="utf-8") as f:
                class_name = json.load(f).get("_class_name", "ControlNetModel")
        except Exception as e:
            print(f"[load] 解析 config.json 失败: {e}, 默认走 ControlNetModel")

    if class_name == "WeatherRestorationControlNet":
        print(f"[load] 检测到自定义类 {class_name}, 用 WeatherRestorationControlNet.from_pretrained 加载")
        # 强制先把目录切到绝对路径
        cn_path_abs = str(Path(cn_path).resolve())
        model = WeatherRestorationControlNet.from_pretrained(cn_path_abs)
    else:
        print(f"[load] 使用 vanilla ControlNetModel (class_name={class_name})")
        model = ControlNetModel.from_pretrained(cn_path)

    # 防御: from_pretrained 偶尔会留下 meta device, 强制搬回 cpu
    try:
        if any(p.is_meta for p in model.parameters()):
            print("[load] 检测到 meta device, 触发 to_empty + state_dict 重载")
            model.to_empty(device="cpu")
    except Exception:
        pass

    return model


# =============================================================================
# zero_conv 收集器 (同时兼容 vanilla ControlNetModel 与 WeatherRestorationControlNet)
# =============================================================================
def collect_zero_convs(controlnet: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    """
    返回 [(name, Conv2d), ...] 列表, 覆盖:
      - WeatherRestorationControlNet: 7 个 zero_conv_stage_* / zero_conv_down_* / zero_conv_mid
      - vanilla ControlNetModel:    12 个 controlnet_down_blocks + 1 个 controlnet_mid_block
    顺序按命名稳定排序, 便于 diff 比较.
    """
    found: List[Tuple[str, torch.nn.Module]] = []

    # 显式命名 (WeatherRestorationControlNet)
    explicit = [
        "zero_conv_stage_0", "zero_conv_stage_1", "zero_conv_stage_2", "zero_conv_stage_3",
        "zero_conv_down_0", "zero_conv_down_1", "zero_conv_mid",
    ]
    for name in explicit:
        m = getattr(controlnet, name, None)
        if isinstance(m, torch.nn.Conv2d):
            found.append((name, m))

    # vanilla ControlNetModel 的 12 + 1
    if hasattr(controlnet, "controlnet_down_blocks") and isinstance(controlnet.controlnet_down_blocks, torch.nn.ModuleList):
        for i, m in enumerate(controlnet.controlnet_down_blocks):
            if isinstance(m, torch.nn.Conv2d):
                found.append((f"controlnet_down_blocks.{i}", m))
    if hasattr(controlnet, "controlnet_mid_block") and isinstance(controlnet.controlnet_mid_block, torch.nn.Conv2d):
        found.append(("controlnet_mid_block", controlnet.controlnet_mid_block))

    # 兜底: 扫描所有 Conv2d, 取 weight/bias 全零的, 避免漏掉未识别的自定义类
    if not found:
        for name, m in controlnet.named_modules():
            if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (1, 1):
                found.append((name, m))

    return found


# =============================================================================
# [1] 权重范数检查
# =============================================================================
def check_weight_norms(
    controlnet: torch.nn.Module,
    cfg: Dict,
    out_dir: Path,
) -> Dict:
    zero_convs = collect_zero_convs(controlnet)
    if not zero_convs:
        print("[1/3] 警告: 未发现任何 zero_conv, 请确认模型类.")
        return {"zero_convs": [], "summary": {}}

    rows: List[Dict] = []
    skipped_meta: List[str] = []
    for name, m in zero_convs:
        with torch.no_grad():
            w = m.weight.detach()
            if w.is_meta:
                skipped_meta.append(name)
                continue
            w_norm = w.float().norm().item()
            b = m.bias.detach() if m.bias is not None else None
            b_norm = (b.float().norm().item() if b is not None and not b.is_meta else 0.0)
        rows.append({"name": name, "weight_norm": w_norm, "bias_norm": b_norm,
                     "in_ch": m.in_channels, "out_ch": m.out_channels})
    if skipped_meta:
        print(f"[1/3] 警告: 以下 zero_conv 仍位于 meta device, 已跳过: {skipped_meta}")

    lines = ["# zero_conv 权重范数检查  (训练后若仍接近 0 -> 控制分支未训练开)",
             f"# 共 {len(rows)} 个 zero_conv   健康阈值 |W| > {cfg['weight_norm_min_healthy']:.1e}\n"]
    header = f"{'name':<32} {'|W|':>14} {'|b|':>14}  {'shape':<14}  health"
    lines.append(header)
    lines.append("-" * len(header))
    norms = [r["weight_norm"] for r in rows]
    for r in rows:
        ok = r["weight_norm"] > cfg["weight_norm_min_healthy"]
        lines.append(f"{r['name']:<32} {r['weight_norm']:>14.6e} {r['bias_norm']:>14.6e}  "
                     f"({r['out_ch']},{r['in_ch']},1,1)        {'OK' if ok else 'NEAR_ZERO'}")
    text = "\n".join(lines)
    _safe_print(text)
    (out_dir / "weight_norms.txt").write_text(text + "\n", encoding="utf-8")

    summary = {
        "num_zero_convs": len(rows),
        "weight_norm_min": float(min(norms)) if norms else 0.0,
        "weight_norm_max": float(max(norms)) if norms else 0.0,
        "weight_norm_mean": float(np.mean(norms)) if norms else 0.0,
        "weight_norm_median": float(np.median(norms)) if norms else 0.0,
        "near_zero_count": int(sum(1 for n in norms if n <= cfg["weight_norm_min_healthy"])),
    }
    return {"zero_convs": rows, "summary": summary}


# =============================================================================
# [2] 前向残差统计 (用 forward hook 抓 zero_conv 输出)
# =============================================================================
class _ResidualStatHook:
    """对每个 zero_conv 注册 forward hook, 记录 (mean, max) of |out|."""

    def __init__(self) -> None:
        self.records: List[Dict] = []   # [{name, mean, max, shape, numel}, ...]
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def attach(self, zero_convs: List[Tuple[str, torch.nn.Module]]) -> None:
        for name, mod in zero_convs:
            handle = mod.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(_mod, _inp, out):
            with torch.no_grad():
                t = out.detach().float()
                self.records.append({
                    "name": name,
                    "abs_mean": t.abs().mean().item(),
                    "abs_max": t.abs().max().item(),
                    "shape": list(t.shape),
                    "numel": int(t.numel()),
                })
        return hook


@torch.no_grad()
def _run_controlnet_forward_only(
    pipeline: StableDiffusionControlNetPipeline,
    lq_pil: Image.Image,
    cfg: Dict,
    hook: _ResidualStatHook,
) -> None:
    """
    不走完整 denoising loop, 直接调 pipeline.controlnet(...)
    跑一次 LQ 编码 -> 4 个特征 -> 7/13 个 zero_conv 输出, hook 抓残差.
    """
    device = pipeline._execution_device
    dtype = pipeline.controlnet.dtype
    height = width = cfg["resolution"]

    # 用 pipeline.image_processor 把 PIL -> tensor (与 denoising loop 内部一致)
    cond_tensor = pipeline.image_processor.preprocess(
        lq_pil, height=height, width=width
    ).to(device=device, dtype=dtype)
    if cond_tensor.dim() == 3:
        cond_tensor = cond_tensor.unsqueeze(0)

    # 用空 prompt / 随机 timestep, 目的是让所有 zero_conv 都被触发
    batch = cond_tensor.shape[0]
    prompt = cfg.get("prompt", "") or " "
    text_emb = pipeline.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt=None,
    )
    # encode_prompt 在 cfg=False 时只返回 prompt_embeds
    if isinstance(text_emb, tuple):
        prompt_embeds = text_emb[0]
    else:
        prompt_embeds = text_emb

    # 随机 latent + 一个 timestep, 模拟真实 forward 触发所有 zero_conv
    latent = torch.randn(
        (batch, pipeline.unet.config.in_channels, height // 8, width // 8),
        device=device, dtype=dtype,
    )
    t = torch.tensor([int(pipeline.scheduler.config.num_train_timesteps) - 1],
                     device=device, dtype=torch.long)
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    # 用 autocast 包裹, 让 ControlNet 内部 UNet/attn 也走 fp16, 与训练时一致
    if dtype == torch.float16:
        ctx = torch.autocast("cuda", dtype=torch.float16)
    elif dtype == torch.bfloat16:
        ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        ctx = torch.autocast("cuda", enabled=False)
    with ctx:
        pipeline.controlnet(
            sample=latent,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=cond_tensor,
            conditioning_scale=1.0,
            return_dict=False,
        )


def check_residual_stats(
    controlnet: torch.nn.Module,
    pipeline: StableDiffusionControlNetPipeline,
    lq_pil: Image.Image,
    cfg: Dict,
    out_dir: Path,
) -> Dict:
    zero_convs = collect_zero_convs(controlnet)
    if not zero_convs:
        print("[2/3] 警告: 未发现任何 zero_conv.")
        return {"records": [], "summary": {}}

    hook = _ResidualStatHook()
    hook.attach(zero_convs)
    try:
        _run_controlnet_forward_only(pipeline, lq_pil, cfg, hook)
    finally:
        hook.remove()

    if not hook.records:
        print("[2/3] 警告: forward hook 未抓到任何 zero_conv 输出.")
        return {"records": [], "summary": {}}

    lines = ["# 控制残差 (zero_conv 输出) abs 统计  正常区间 1e-2 ~ 1.0",
             f"# 共 {len(hook.records)} 次 zero_conv forward   阈值 "
             f"|.| mean in [{cfg['residual_abs_mean_min_healthy']:.1e}, "
             f"{cfg['residual_abs_mean_max_healthy']:.1e}]\n"]
    header = f"{'name':<32} {'abs_mean':>14} {'abs_max':>14}  {'shape':<22}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in hook.records:
        lines.append(f"{r['name']:<32} {r['abs_mean']:>14.6e} {r['abs_max']:>14.6e}  {str(r['shape']):<22}")
    text = "\n".join(lines)
    _safe_print(text)
    (out_dir / "residual_stats.txt").write_text(text + "\n", encoding="utf-8")

    means = [r["abs_mean"] for r in hook.records]
    maxs = [r["abs_max"] for r in hook.records]
    summary = {
        "num_records": len(hook.records),
        "abs_mean_min": float(min(means)),
        "abs_mean_max": float(max(means)),
        "abs_mean_mean": float(np.mean(means)),
        "abs_mean_median": float(np.median(means)),
        "abs_max_global": float(max(maxs)),
    }
    return {"records": hook.records, "summary": summary}


# =============================================================================
# [3] 消融对比: controlnet_conditioning_scale ∈ {1.0, 0.0}
# =============================================================================
def _pil_to_unit_tensor(pil: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)


@torch.no_grad()
def _infer_one(
    pipeline: StableDiffusionControlNetPipeline,
    lq_pil: Image.Image,
    cfg: Dict,
    conditioning_scale: float,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    context = (
        torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype is not None
        else torch.autocast("cpu", enabled=False)
    )
    with context:
        out = pipeline(
            cfg.get("prompt", ""),
            lq_pil,
            height=cfg["resolution"],
            width=cfg["resolution"],
            num_inference_steps=cfg["num_inference_steps"],
            guidance_scale=cfg["guidance_scale"],
            negative_prompt=cfg.get("negative_prompt"),
            controlnet_conditioning_scale=conditioning_scale,
            generator=generator,
        )
    return out.images[0]


def _make_grid_4(images: List[Image.Image], labels: List[str]) -> Image.Image:
    """横向拼接 4 张等大图 + 顶部标签."""
    w, h = images[0].size
    pad = 8
    bar_h = 28
    total_w = w * len(images) + pad * (len(images) - 1)
    canvas = Image.new("RGB", (total_w, h + bar_h), (255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, (img, lab) in enumerate(zip(images, labels)):
        x = i * (w + pad)
        canvas.paste(img, (x, bar_h))
        draw.text((x + 4, 4), lab, fill=(0, 0, 0), font=font)
    return canvas


def run_ablation(
    pipeline: StableDiffusionControlNetPipeline,
    lq_pil: Image.Image,
    gt_pil: Optional[Image.Image],
    cfg: Dict,
    out_dir: Path,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
) -> Dict:
    print(f"[3/3] 消融推理  seed={cfg['seed']}  resolution={cfg['resolution']}  "
          f"steps={cfg['num_inference_steps']}  cfg={cfg['guidance_scale']}")

    lq_pil = lq_pil.convert("RGB").resize((cfg["resolution"], cfg["resolution"]), Image.BICUBIC)
    if gt_pil is not None:
        gt_pil = gt_pil.convert("RGB").resize((cfg["resolution"], cfg["resolution"]), Image.BICUBIC)

    t0 = time.time()
    pred_with = _infer_one(pipeline, lq_pil, cfg, 1.0, device, autocast_dtype)
    t_with = time.time() - t0
    t0 = time.time()
    pred_without = _infer_one(pipeline, lq_pil, cfg, 0.0, device, autocast_dtype)
    t_without = time.time() - t0
    print(f"      cond=1.0 用时 {t_with:.2f}s,  cond=0.0 用时 {t_without:.2f}s")

    stem = Path(cfg["lq_image_path"]).stem
    pred_with.save(out_dir / f"{stem}_cond1.0_pred.png")
    pred_without.save(out_dir / f"{stem}_cond0.0_pred.png")
    lq_pil.save(out_dir / f"{stem}_lq.png")
    if gt_pil is not None:
        gt_pil.save(out_dir / f"{stem}_gt.png")

    # 指标 (有 GT 时才有意义)
    metrics: Dict = {}
    if gt_pil is not None:
        gt_t = _pil_to_unit_tensor(gt_pil, device)
        with_t = _pil_to_unit_tensor(pred_with, device)
        without_t = _pil_to_unit_tensor(pred_without, device)
        m_with = {
            "psnr": calc_psnr(with_t, gt_t),
            "ssim": calc_ssim(with_t, gt_t),
        }
        m_without = {
            "psnr": calc_psnr(without_t, gt_t),
            "ssim": calc_ssim(without_t, gt_t),
        }
        # LPIPS 懒加载 (避免非消融模式强行下载)
        try:
            _ = _get_lpips_model(cfg.get("lpips_net", "alex"), device=device)
            m_with["lpips"] = calc_lpips(with_t, gt_t, net=cfg.get("lpips_net", "alex"))
            m_without["lpips"] = calc_lpips(without_t, gt_t, net=cfg.get("lpips_net", "alex"))
        except Exception as e:
            print(f"      LPIPS 跳过: {e}")
            m_with["lpips"] = float("nan")
            m_without["lpips"] = float("nan")

        metrics = {
            "cond1.0": m_with,
            "cond0.0": m_without,
            "delta_psnr": m_with["psnr"] - m_without["psnr"],
            "delta_ssim": m_with["ssim"] - m_without["ssim"],
            "delta_lpips": (
                (m_with["lpips"] - m_without["lpips"])
                if not (m_with["lpips"] != m_with["lpips"]) else float("nan")
            ),
        }
        # 与 LQ 的 PSNR (看 LQ→GT 距离, 供解读)
        lq_t = _pil_to_unit_tensor(lq_pil, device)
        metrics["lq_to_gt_psnr"] = calc_psnr(lq_t, gt_t)
        metrics["lq_to_gt_ssim"] = calc_ssim(lq_t, gt_t)

    # 横排对比图
    grid_imgs = [lq_pil, pred_with, pred_without] + ([gt_pil] if gt_pil is not None else [])
    grid_labels = ["LQ", "pred (cond=1.0)", "pred (cond=0.0)"] + (["GT"] if gt_pil is not None else [])
    _make_grid_4(grid_imgs, grid_labels).save(out_dir / "ablation_grid.png")

    # 控制残差视觉差图
    diff = np.abs(np.asarray(pred_with, dtype=np.int16) - np.asarray(pred_without, dtype=np.int16))
    diff = np.clip(diff * 4, 0, 255).astype(np.uint8)   # ×4 增强显示
    Image.fromarray(diff).save(out_dir / f"{stem}_abs_diff_x4.png")

    return {
        "metrics": metrics,
        "inference_time_sec": {"cond1.0": t_with, "cond0.0": t_without},
    }


# =============================================================================
# Pipeline 加载
# =============================================================================
def build_pipeline(cfg: Dict, device: torch.device, dtype: torch.dtype
                   ) -> StableDiffusionControlNetPipeline:
    cn_path = resolve_controlnet_path(cfg["controlnet_model_path"])
    print(f"[load] ControlNet 权重目录: {cn_path}")
    controlnet = _load_controlnet_smart(cn_path)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        cfg["pretrained_model_name_or_path"],
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if cfg.get("enable_xformers_memory_efficient_attention", False):
        if is_xformers_available():
            pipeline.enable_xformers_memory_efficient_attention()
        else:
            print("[warn] xformers 不可用, 已跳过")
    return pipeline


# =============================================================================
# 图像加载辅助
# =============================================================================
def _load_image(path: str) -> Image.Image:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"找不到图像: {p}")
    return Image.open(p).convert("RGB")


# =============================================================================
# 主流程
# =============================================================================
def main() -> None:
    args = parse_args()
    cfg = merge_config(args)

    device = torch.device("cpu" if cfg["_force_cpu"] or not torch.cuda.is_available() else "cuda")
    if device.type == "cpu" and cfg["mixed_precision"] != "no":
        print("[warn] CPU 模式, 强制关闭 mixed_precision")
        cfg["mixed_precision"] = "no"
    autocast_dtype = (
        torch.float16 if cfg["mixed_precision"] == "fp16" and device.type == "cuda"
        else torch.bfloat16 if cfg["mixed_precision"] == "bf16" and device.type == "cuda"
        else None
    )
    dtype = torch.float32 if device.type == "cpu" or autocast_dtype is None else autocast_dtype

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg["output_dir"]) / f"{timestamp}_diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["_output_dir"] = str(out_dir)

    # 写入 config snapshot
    snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}
    snapshot["device"] = str(device)
    snapshot["dtype"] = str(dtype)
    snapshot["timestamp"] = timestamp
    with io.open(out_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False, allow_unicode=True)

    print(f"[init] device={device}  dtype={dtype}  mode={cfg['mode']}")
    print(f"[init] 输出目录: {out_dir}")

    results: Dict = {"config": snapshot, "weight_norms": None,
                     "residual_stats": None, "ablation": None}

    # ------ [1] 权重范数 (不依赖 LQ 图, 直接加载 checkpoint) ------
    if cfg["mode"] in ("all", "weights_only"):
        print("\n" + "=" * 70)
        print("[1/3] zero_conv 权重范数检查  (无需推理, 只加载权重)")
        print("=" * 70)
        cn_path = resolve_controlnet_path(cfg["controlnet_model_path"])
        controlnet_only = _load_controlnet_smart(cn_path)
        results["weight_norms"] = check_weight_norms(controlnet_only, cfg, out_dir)
        del controlnet_only

    # ------ 下面三项都需要加载完整 SD pipeline ------
    if cfg["mode"] in ("all", "residuals_only", "ablation"):
        print("\n[load] 构建 SD + ControlNet pipeline ...")
        pipeline = build_pipeline(cfg, device, dtype)

        lq_pil = _load_image(cfg["lq_image_path"])
        gt_pil: Optional[Image.Image] = None
        if cfg.get("gt_image_path") and Path(cfg["gt_image_path"]).is_file():
            gt_pil = _load_image(cfg["gt_image_path"])
        else:
            print(f"[warn] 未提供有效 GT 图, 消融 PSNR 仍会跑, 但跳过 SSIM/LPIPS/对比")

        # ------ [2] 前向残差统计 ------
        if cfg["mode"] in ("all", "residuals_only"):
            print("\n" + "=" * 70)
            print("[2/3] 控制残差 (zero_conv 输出) 前向数值  (一次前向, 无需 denoising)")
            print("=" * 70)
            results["residual_stats"] = check_residual_stats(
                pipeline.controlnet, pipeline, lq_pil, cfg, out_dir,
            )

        # ------ [3] 消融对比 ------
        if cfg["mode"] in ("all", "ablation"):
            print("\n" + "=" * 70)
            print("[3/3] 消融对比: cond=1.0 vs cond=0.0")
            print("=" * 70)
            results["ablation"] = run_ablation(
                pipeline, lq_pil, gt_pil, cfg, out_dir, device, autocast_dtype,
            )

        del pipeline
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------ 汇总 + 判读建议 ------
    _write_summary(results, cfg, out_dir)
    print(f"\n[done] 全部结果已写入: {out_dir}")


def _write_summary(results: Dict, cfg: Dict, out_dir: Path) -> None:
    wn = (results.get("weight_norms") or {}).get("summary", {})
    rs = (results.get("residual_stats") or {}).get("summary", {})
    ab = (results.get("ablation") or {}).get("metrics", {}) if results.get("ablation") else {}

    diagnostics: List[str] = []

    if wn:
        if wn.get("weight_norm_max", 0.0) <= cfg["weight_norm_min_healthy"]:
            diagnostics.append(
                f"[1/3] [BAD] 所有 zero_conv |W| 均 <= {cfg['weight_norm_min_healthy']:.0e}, "
                f"控制分支基本未训练, 建议重新检查训练配置 (lr / 冻结 / zero_conv init).")
        elif wn.get("near_zero_count", 0) == 0:
            diagnostics.append(
                f"[1/3] [OK] zero_conv 权重范数健康 (max={wn.get('weight_norm_max'):.2e}, "
                f"mean={wn.get('weight_norm_mean'):.2e}).")
        else:
            diagnostics.append(
                f"[1/3] [WARN] {wn.get('near_zero_count')}/{wn.get('num_zero_convs')} 个 "
                f"zero_conv 权重范数仍 <= {cfg['weight_norm_min_healthy']:.0e}.")

    if rs:
        m = rs.get("abs_mean_mean", 0.0)
        if m < cfg["residual_abs_mean_min_healthy"]:
            diagnostics.append(
                f"[2/3] [BAD] 控制残差 abs mean 平均仅 {m:.2e} < {cfg['residual_abs_mean_min_healthy']:.0e}, "
                f"接近 zero_conv 初始化值, 控制分支对 SD 无约束.")
        elif m > cfg["residual_abs_mean_max_healthy"]:
            diagnostics.append(
                f"[2/3] [BAD] 控制残差 abs mean 平均 {m:.2e} > {cfg['residual_abs_mean_max_healthy']:.1f}, "
                f"残差数值爆炸, 容易污染 SD 预训练特征 (大面积色块/伪影).")
        else:
            diagnostics.append(
                f"[2/3] [OK] 控制残差数值处于正常区间 (abs mean ~= {m:.2e}).")

    if ab:
        d_psnr = ab.get("delta_psnr")
        if d_psnr is None:
            pass
        elif d_psnr < cfg["ablation_psnr_drop_db"]:
            diagnostics.append(
                f"[3/3] [BAD] 消融 PSNR 差距仅 {d_psnr:+.3f} dB (< {cfg['ablation_psnr_drop_db']} dB 阈值), "
                f"控制分支对画质几乎无贡献, C2F 未学到有效退化特征.")
        else:
            diagnostics.append(
                f"[3/3] [OK] 消融 PSNR 差距 {d_psnr:+.3f} dB (cond=1.0 优于 cond=0.0), "
                f"控制分支对画质有可观测的正向贡献.")

    summary = {
        "weight_norms": wn,
        "residual_stats": rs,
        "ablation": results.get("ablation"),
        "diagnostics": diagnostics,
    }
    with io.open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = ["# 诊断汇总", ""]
    if wn:
        lines += ["[1/3] 权重范数:",
                  f"  zero_conv 总数: {wn.get('num_zero_convs', 0)}",
                  f"  |W| min/max/mean/median: "
                  f"{wn.get('weight_norm_min', 0):.2e} / "
                  f"{wn.get('weight_norm_max', 0):.2e} / "
                  f"{wn.get('weight_norm_mean', 0):.2e} / "
                  f"{wn.get('weight_norm_median', 0):.2e}",
                  f"  仍接近 0 的层数: {wn.get('near_zero_count', 0)}",
                  ""]
    if rs:
        lines += ["[2/3] 前向残差 (zero_conv 输出):",
                  f"  记录数: {rs.get('num_records', 0)}",
                  f"  abs mean min/max/mean/median: "
                  f"{rs.get('abs_mean_min', 0):.2e} / "
                  f"{rs.get('abs_mean_max', 0):.2e} / "
                  f"{rs.get('abs_mean_mean', 0):.2e} / "
                  f"{rs.get('abs_mean_median', 0):.2e}",
                  f"  abs max 全局最大: {rs.get('abs_max_global', 0):.2e}",
                  ""]
    if ab:
        lines += ["[3/3] 消融 PSNR 对比:",
                  f"  cond=1.0  PSNR/SSIM/LPIPS: "
                  f"{ab.get('cond1.0', {}).get('psnr', float('nan')):.3f} / "
                  f"{ab.get('cond1.0', {}).get('ssim', float('nan')):.4f} / "
                  f"{ab.get('cond1.0', {}).get('lpips', float('nan')):.4f}",
                  f"  cond=0.0  PSNR/SSIM/LPIPS: "
                  f"{ab.get('cond0.0', {}).get('psnr', float('nan')):.3f} / "
                  f"{ab.get('cond0.0', {}).get('ssim', float('nan')):.4f} / "
                  f"{ab.get('cond0.0', {}).get('lpips', float('nan')):.4f}",
                  f"  delta PSNR (1.0 - 0.0): {ab.get('delta_psnr', float('nan')):+.4f} dB",
                  f"  delta SSIM (1.0 - 0.0): {ab.get('delta_ssim', float('nan')):+.4f}",
                  f"  delta LPIPS(1.0 - 0.0): {ab.get('delta_lpips', float('nan')):+.4f}  (负值=cond=1.0 更优)",
                  f"  LQ->GT  PSNR/SSIM: "
                  f"{ab.get('lq_to_gt_psnr', float('nan')):.3f} / "
                  f"{ab.get('lq_to_gt_ssim', float('nan')):.4f}",
                  ""]
    lines += ["判读建议:"]
    lines += [f"  {d}" for d in diagnostics] if diagnostics else ["  (无可用诊断)"]
    text = "\n".join(lines)
    _safe_print("\n" + text)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
