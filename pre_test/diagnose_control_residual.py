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
    # 实验 A: 把控制残差乘 0.1, 验证残差幅值爆炸是色块根源
    "exp_a_scale": 0.1,
    # 实验 C: 固定 TimedC2F 的 gamma, 关闭动态时序缩放
    "exp_c_gamma": 0.5,
    # 选用哪些实验: "none" / "a" / "b" / "c" / "ab" / "abc" / "all"
    "experiments": "abc",
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
    parser.add_argument("--experiments", type=str, default=None,
                        help="启用哪些实验: a=cond_scale=0.1 残差压制; "
                             "b=C2F/Adapter/zero_conv 三段幅值统计; "
                             "c=固定 TimedC2F gamma. 例: 'abc' / 'a,c' / 'none'")
    parser.add_argument("--exp_a_scale", type=float, default=None,
                        help="实验 A 的控制残差缩放系数 (默认 0.1)")
    parser.add_argument("--exp_c_gamma", type=float, default=None,
                        help="实验 C 固定的 gamma 值 (默认 0.5, 关闭动态时序)")
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
        "exp_a_scale": args.exp_a_scale,
        "exp_c_gamma": args.exp_c_gamma,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    # 解析 --experiments 字符串 -> set of {a,b,c}
    raw = (args.experiments if args.experiments is not None else cfg.get("experiments", "abc"))
    raw_norm = raw.replace(",", "").replace(" ", "").lower()
    if raw_norm in ("none", "", "off"):
        cfg["_experiments"] = set()
    elif raw_norm in ("all", "abc"):
        cfg["_experiments"] = {"a", "b", "c"}
    else:
        cfg["_experiments"] = {c for c in raw_norm if c in "abc"}

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


def is_weather_controlnet(controlnet: torch.nn.Module) -> bool:
    """检测是否为项目自定义的 WeatherRestorationControlNet (有 c2f / adapter)."""
    return hasattr(controlnet, "timed_c2f_blocks") and hasattr(controlnet, "adapters") \
        and hasattr(controlnet, "weather_encoder")


def collect_weather_intermediate(controlnet: torch.nn.Module
                                  ) -> Tuple[List[Tuple[str, torch.nn.Module]],
                                             List[Tuple[str, torch.nn.Module]]]:
    """
    仅在 WeatherRestorationControlNet 下有定义.
    返回 (c2f_outputs, adapter_outputs):
      c2f_outputs: 4 个 TimedC2FBlock.c2f 子模块 (C2F 主干出口)
      adapter_outputs: 4 个 LightweightAdapter
    """
    c2f_list: List[Tuple[str, torch.nn.Module]] = []
    adapter_list: List[Tuple[str, torch.nn.Module]] = []
    if not is_weather_controlnet(controlnet):
        return c2f_list, adapter_list
    for i, blk in enumerate(controlnet.timed_c2f_blocks):
        c2f_list.append((f"timed_c2f_blocks.{i}.c2f", blk.c2f))
    for i, adp in enumerate(controlnet.adapters):
        adapter_list.append((f"adapters.{i}", adp))
    return c2f_list, adapter_list


# =============================================================================
# 实验 C: monkey-patch TimedC2FBlock.forward 强制 gamma 固定
# =============================================================================
def _patch_timed_c2f_fixed_gamma(controlnet: torch.nn.Module, fixed_gamma: float) -> List:
    """
    对每个 TimedC2FBlock, 把 forward 替换为强制 gamma = fixed_gamma 的版本.
    返回 handles 列表, 用于事后恢复.
    """
    if not is_weather_controlnet(controlnet):
        return []

    def _make_new_forward(orig_time_emb, orig_time_mlp, orig_c2f, fg: float):
        def new_forward(x, timestep):
            if not torch.is_tensor(timestep):
                timestep = torch.tensor([timestep] * x.shape[0], device=x.device)
            if timestep.ndim == 0:
                timestep = timestep.unsqueeze(0).expand(x.shape[0])
            elif timestep.shape[0] == 1 and x.shape[0] > 1:
                timestep = timestep.expand(x.shape[0])
            target_dtype = x.dtype
            te = orig_time_emb(timestep).to(target_dtype)
            # 关键: gamma 强制为常数, 忽略 time_mlp 实际输出
            gamma = torch.full((x.shape[0], 1), fg, device=x.device, dtype=target_dtype)
            refined = orig_c2f(x)
            scale = 1.0 + 0.2 * gamma.view(-1, 1, 1, 1)
            return refined * scale
        return new_forward

    import types
    handles = []
    for blk in controlnet.timed_c2f_blocks:
        blk._orig_forward = blk.forward
        blk.forward = types.MethodType(
            _make_new_forward(blk.time_emb, blk.time_mlp, blk.c2f, fixed_gamma), blk,
        )
        handles.append(blk)
    return handles


def _unpatch_timed_c2f(handles: List) -> None:
    for blk in handles:
        if hasattr(blk, "_orig_forward"):
            blk.forward = blk._orig_forward
            del blk._orig_forward


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
    """对每个 zero_conv 注册 forward hook, 记录 (mean, std, abs_mean, abs_max) of out."""

    def __init__(self, record_signed: bool = False) -> None:
        self.records: List[Dict] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._record_signed = record_signed  # 实验 B 还要 mean/std/min/max, 而非只看 |·|

    def attach(self, mods: List[Tuple[str, torch.nn.Module]]) -> None:
        for name, mod in mods:
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
                rec = {
                    "name": name,
                    "abs_mean": t.abs().mean().item(),
                    "abs_max": t.abs().max().item(),
                    "shape": list(t.shape),
                    "numel": int(t.numel()),
                }
                if self._record_signed:
                    rec.update({
                        "mean": t.mean().item(),
                        "std": t.std().item(),
                        "min": t.min().item(),
                        "max": t.max().item(),
                    })
                self.records.append(rec)
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
    include_intermediate: bool = False,
) -> Dict:
    """
    默认只统计 zero_conv 输出 (abs mean / max). 实验 B 启用时, 同时 hook
    C2F 主干出口 + Adapter 出口, 统计 mean / std / min / max 全套幅值信息.
    """
    zero_convs = collect_zero_convs(controlnet)
    if not zero_convs:
        print("[2/3] 警告: 未发现任何 zero_conv.")
        return {"records": [], "summary": {}, "intermediate": []}

    targets: List[Tuple[str, torch.nn.Module]] = list(zero_convs)
    c2f_targets, adapter_targets = [], []
    if include_intermediate and is_weather_controlnet(controlnet):
        c2f_targets, adapter_targets = collect_weather_intermediate(controlnet)
        targets = c2f_targets + adapter_targets + zero_convs

    hook = _ResidualStatHook(record_signed=include_intermediate)
    hook.attach(targets)
    try:
        _run_controlnet_forward_only(pipeline, lq_pil, cfg, hook)
    finally:
        hook.remove()

    if not hook.records:
        print("[2/3] 警告: forward hook 未抓到任何 zero_conv 输出.")
        return {"records": [], "summary": {}, "intermediate": []}

    # ---- 写 zero_conv 部分的旧格式 (兼容原 summary) ----
    zc_records = [r for r in hook.records if r["name"] in {n for n, _ in zero_convs}]
    lines = ["# 控制残差 (zero_conv 输出) abs 统计  正常区间 1e-2 ~ 1.0",
             f"# 共 {len(zc_records)} 次 zero_conv forward   阈值 "
             f"|.| mean in [{cfg['residual_abs_mean_min_healthy']:.1e}, "
             f"{cfg['residual_abs_mean_max_healthy']:.1e}]\n"]
    header = f"{'name':<32} {'abs_mean':>14} {'abs_max':>14}  {'shape':<22}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in zc_records:
        lines.append(f"{r['name']:<32} {r['abs_mean']:>14.6e} {r['abs_max']:>14.6e}  {str(r['shape']):<22}")
    text = "\n".join(lines)
    _safe_print(text)
    (out_dir / "residual_stats.txt").write_text(text + "\n", encoding="utf-8")

    means = [r["abs_mean"] for r in zc_records]
    maxs = [r["abs_max"] for r in zc_records]
    summary = {
        "num_records": len(zc_records),
        "abs_mean_min": float(min(means)) if means else 0.0,
        "abs_mean_max": float(max(means)) if means else 0.0,
        "abs_mean_mean": float(np.mean(means)) if means else 0.0,
        "abs_mean_median": float(np.median(means)) if means else 0.0,
        "abs_max_global": float(max(maxs)) if maxs else 0.0,
    }

    intermediate: List[Dict] = []
    if include_intermediate and c2f_targets:
        intermediate = [r for r in hook.records if r["name"] not in {n for n, _ in zero_convs}]
        _write_intermediate_stats(intermediate, c2f_targets, adapter_targets, out_dir)

    return {"records": zc_records, "summary": summary, "intermediate": intermediate}


def _write_intermediate_stats(
    intermediate: List[Dict],
    c2f_targets: List[Tuple[str, torch.nn.Module]],
    adapter_targets: List[Tuple[str, torch.nn.Module]],
    out_dir: Path,
) -> None:
    """实验 B: 写 C2F/Adapter 中间段幅值 (mean/std/min/max)."""
    by_name = {r["name"]: r for r in intermediate}
    c2f_names = {n for n, _ in c2f_targets}
    adp_names = {n for n, _ in adapter_targets}

    lines = ["# 实验 B: C2F -> Adapter -> zero_conv 三段幅值统计",
             "# 重点观察 mean / std / |max| 的逐级放大倍数.\n"]

    def _block(title: str, names: List[str]):
        lines.append(f"## {title}")
        lines.append(f"{'name':<32} {'mean':>13} {'std':>13} {'|min|':>13} {'|max|':>13}  shape")
        lines.append("-" * 100)
        for n in names:
            r = by_name.get(n)
            if r is None:
                lines.append(f"{n:<32} (not found)")
                continue
            lines.append(f"{n:<32} {r['mean']:>13.4e} {r['std']:>13.4e} "
                         f"{abs(r['min']):>13.4e} {r['max']:>13.4e}  {str(r['shape'])}")
        lines.append("")

    if c2f_names:
        _block("C2F 主干出口 (timed_c2f_blocks[i].c2f)", [n for n, _ in c2f_targets])
    if adp_names:
        _block("Adapter 出口 (adapters[i])", [n for n, _ in adapter_targets])

    # 放大倍数分析: Adapter / C2F
    lines.append("## 放大倍数分析 (每 stage: adapter_abs_max / c2f_abs_max)")
    lines.append(f"{'stage':<10} {'c2f_|max|':>14} {'adapter_|max|':>14}  ratio")
    lines.append("-" * 60)
    for i in range(len(c2f_targets)):
        c2f_r = by_name.get(c2f_targets[i][0])
        adp_r = by_name.get(adapter_targets[i][0])
        if c2f_r and adp_r:
            ratio = adp_r["abs_max"] / max(c2f_r["abs_max"], 1e-12)
            lines.append(f"stage_{i:<3} {c2f_r['abs_max']:>14.4e} {adp_r['abs_max']:>14.4e}  x{ratio:>8.2f}")
    text = "\n".join(lines)
    _safe_print("\n" + text + "\n")
    (out_dir / "intermediate_stats.txt").write_text(text + "\n", encoding="utf-8")


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
    experiments: set = cfg.get("_experiments", set()) or set()
    exp_a = experiments and "a" in experiments
    exp_c = experiments and "c" in experiments
    is_weather = is_weather_controlnet(pipeline.controlnet)
    if exp_c and not is_weather:
        print("[3/3] 实验 C 需要 WeatherRestorationControlNet (无 c2f / adapter), 跳过.")
        exp_c = False

    print(f"[3/3] 消融推理  seed={cfg['seed']}  resolution={cfg['resolution']}  "
          f"steps={cfg['num_inference_steps']}  cfg={cfg['guidance_scale']}  "
          f"experiments=A:{exp_a} C:{exp_c}")

    lq_pil = lq_pil.convert("RGB").resize((cfg["resolution"], cfg["resolution"]), Image.BICUBIC)
    if gt_pil is not None:
        gt_pil = gt_pil.convert("RGB").resize((cfg["resolution"], cfg["resolution"]), Image.BICUBIC)

    # ---- 准备实验列表 ----
    # (key, label, cond_scale, exp_c_gamma 或 None)
    runs: List[Tuple[str, str, float, Optional[float]]] = [
        ("cond0.0", "pred (cond=0.0, baseline)", 0.0, None),
        ("cond1.0", "pred (cond=1.0, original)", 1.0, None),
    ]
    if exp_a:
        runs.append((
            f"expA_cond{cfg['exp_a_scale']:.2f}".rstrip("0").rstrip("."),
            f"pred (cond={cfg['exp_a_scale']}, A=scale down)",
            float(cfg["exp_a_scale"]), None,
        ))
    if exp_c:
        runs.append((
            f"expC_gamma{cfg['exp_c_gamma']:.2f}".rstrip("0").rstrip("."),
            f"pred (gamma={cfg['exp_c_gamma']}, C=fixed gamma)",
            1.0, float(cfg["exp_c_gamma"]),
        ))

    # ---- 逐个推理 ----
    preds: Dict[str, Image.Image] = {}
    times: Dict[str, float] = {}
    for key, _label, cond, fixed_gamma in runs:
        patched_handles = []
        if fixed_gamma is not None:
            patched_handles = _patch_timed_c2f_fixed_gamma(pipeline.controlnet, fixed_gamma)
        try:
            t0 = time.time()
            preds[key] = _infer_one(pipeline, lq_pil, cfg, cond, device, autocast_dtype)
            times[key] = time.time() - t0
            print(f"      {key} (cond={cond}, gamma={fixed_gamma}) 用时 {times[key]:.2f}s")
        finally:
            _unpatch_timed_c2f(patched_handles)

    stem = Path(cfg["lq_image_path"]).stem
    for key, pil in preds.items():
        pil.save(out_dir / f"{stem}_{key}_pred.png")
    lq_pil.save(out_dir / f"{stem}_lq.png")
    if gt_pil is not None:
        gt_pil.save(out_dir / f"{stem}_gt.png")

    # ---- 指标 (有 GT 时才有意义) ----
    metrics: Dict = {}
    if gt_pil is not None:
        gt_t = _pil_to_unit_tensor(gt_pil, device)
        lq_t = _pil_to_unit_tensor(lq_pil, device)
        for key, pil in preds.items():
            t = _pil_to_unit_tensor(pil, device)
            metrics[key] = {
                "psnr": calc_psnr(t, gt_t),
                "ssim": calc_ssim(t, gt_t),
            }
        # LPIPS 懒加载
        try:
            _ = _get_lpips_model(cfg.get("lpips_net", "alex"), device=device)
            for key, pil in preds.items():
                t = _pil_to_unit_tensor(pil, device)
                metrics[key]["lpips"] = calc_lpips(t, gt_t, net=cfg.get("lpips_net", "alex"))
        except Exception as e:
            print(f"      LPIPS 跳过: {e}")
            for key in preds:
                metrics[key]["lpips"] = float("nan")

        # Δ 全部以 cond=0.0 为基线 (也支持与 cond=1.0 对比)
        baseline = metrics.get("cond0.0", {})
        for key, m in metrics.items():
            d = {}
            for k in ("psnr", "ssim", "lpips"):
                if k in m and k in baseline and not (m[k] != m[k]) and not (baseline[k] != baseline[k]):
                    d[f"delta_{k}_vs_baseline"] = m[k] - baseline[k]
            m.update(d)
        # 与 LQ 的 PSNR
        metrics["lq_to_gt_psnr"] = calc_psnr(lq_t, gt_t)
        metrics["lq_to_gt_ssim"] = calc_ssim(lq_t, gt_t)

    # ---- 横排对比图 (LQ | 每个实验 | GT) ----
    grid_imgs = [lq_pil]
    grid_labels = ["LQ"]
    for key, label, _c, _g in runs:
        grid_imgs.append(preds[key])
        grid_labels.append(label)
    if gt_pil is not None:
        grid_imgs.append(gt_pil)
        grid_labels.append("GT")
    _make_grid_4(grid_imgs, grid_labels).save(out_dir / "ablation_grid.png")

    # 控制残差视觉差图 (cond=1.0 vs cond=0.0)
    if "cond1.0" in preds and "cond0.0" in preds:
        diff = np.abs(np.asarray(preds["cond1.0"], dtype=np.int16)
                      - np.asarray(preds["cond0.0"], dtype=np.int16))
        diff = np.clip(diff * 4, 0, 255).astype(np.uint8)
        Image.fromarray(diff).save(out_dir / f"{stem}_abs_diff_cond1_vs_0_x4.png")
    # A 实验 vs cond=0.0 差图 (看压制后的差异)
    for key in preds:
        if key.startswith("expA") and "cond0.0" in preds:
            diff = np.abs(np.asarray(preds[key], dtype=np.int16)
                          - np.asarray(preds["cond0.0"], dtype=np.int16))
            diff = np.clip(diff * 4, 0, 255).astype(np.uint8)
            Image.fromarray(diff).save(out_dir / f"{stem}_abs_diff_{key}_vs_0_x4.png")
        if key.startswith("expC") and "cond1.0" in preds:
            diff = np.abs(np.asarray(preds[key], dtype=np.int16)
                          - np.asarray(preds["cond1.0"], dtype=np.int16))
            diff = np.clip(diff * 4, 0, 255).astype(np.uint8)
            Image.fromarray(diff).save(out_dir / f"{stem}_abs_diff_{key}_vs_1_x4.png")

    return {
        "metrics": metrics,
        "inference_time_sec": times,
        "runs": [{"key": k, "label": lab, "cond_scale": c, "fixed_gamma": g}
                 for k, lab, c, g in runs],
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
                     "residual_stats": None, "ablation": None,
                     "experiments": list(cfg["_experiments"])}

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

        # ------ [2] 前向残差统计 (+ 实验 B 中间段幅值) ------
        if cfg["mode"] in ("all", "residuals_only"):
            print("\n" + "=" * 70)
            label_b = "+ 实验 B 中间段 (C2F/Adapter/zero_conv 三段)" if "b" in cfg["_experiments"] else ""
            print(f"[2/3] 控制残差 (zero_conv 输出) 前向数值  {label_b}")
            print("=" * 70)
            results["residual_stats"] = check_residual_stats(
                pipeline.controlnet, pipeline, lq_pil, cfg, out_dir,
                include_intermediate="b" in cfg["_experiments"],
            )

        # ------ [3] 消融对比 (+ 实验 A cond=0.1 / 实验 C gamma=0.5) ------
        if cfg["mode"] in ("all", "ablation"):
            print("\n" + "=" * 70)
            extra = []
            if "a" in cfg["_experiments"]:
                extra.append(f"实验 A (cond_scale={cfg['exp_a_scale']})")
            if "c" in cfg["_experiments"]:
                extra.append(f"实验 C (gamma={cfg['exp_c_gamma']})")
            extra_str = " + ".join(extra) if extra else "(仅 cond=0.0 vs 1.0)"
            print(f"[3/3] 消融推理  {extra_str}")
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
        # 旧指标: cond=1.0 vs cond=0.0 (兼容老 summary)
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

        # 实验 A: 控制残差乘 0.1, 若色块根源是幅值爆炸, 此处 PSNR 应明显回升
        runs_meta = (results.get("ablation") or {}).get("runs", [])
        run_keys = [r["key"] for r in runs_meta]
        exp_a_key = next((k for k in run_keys if k.startswith("expA")), None)
        if exp_a_key and exp_a_key in ab and "cond0.0" in ab and "cond1.0" in ab:
            psnr_a = ab[exp_a_key].get("psnr", float("nan"))
            psnr_1 = ab["cond1.0"].get("psnr", float("nan"))
            psnr_0 = ab["cond0.0"].get("psnr", float("nan"))
            if psnr_a == psnr_a and psnr_1 == psnr_1 and psnr_0 == psnr_0:
                if psnr_a > psnr_1:
                    diagnostics.append(
                        f"[实验 A] [BAD] cond={cfg['exp_a_scale']} 的 PSNR ({psnr_a:.3f}) "
                        f"高于 cond=1.0 ({psnr_1:.3f}) 差 {psnr_a - psnr_1:+.3f} dB. "
                        f"100% 确认控制残差幅值爆炸是色块根源; "
                        f"建议永久加入可学习残差缩放 (alpha 初始 0.1).")
                elif psnr_a < psnr_0:
                    diagnostics.append(
                        f"[实验 A] [BAD] cond={cfg['exp_a_scale']} 的 PSNR ({psnr_a:.3f}) "
                        f"甚至低于 cond=0.0 基线 ({psnr_0:.3f}), 控制分支方向可能是反的 "
                        f"或残差方向需检查.")
                else:
                    diagnostics.append(
                        f"[实验 A] [WARN] cond={cfg['exp_a_scale']} ({psnr_a:.3f}) 与 "
                        f"cond=1.0 ({psnr_1:.3f}) 差异不显著, 残差缩放可能不是主要矛盾.")

        # 实验 C: 固定 gamma, 若动态时序是次要干扰, 画质变化应较小
        exp_c_key = next((k for k in run_keys if k.startswith("expC")), None)
        if exp_c_key and exp_c_key in ab and "cond1.0" in ab:
            psnr_c = ab[exp_c_key].get("psnr", float("nan"))
            psnr_1 = ab["cond1.0"].get("psnr", float("nan"))
            if psnr_c == psnr_c and psnr_1 == psnr_1:
                d = psnr_c - psnr_1
                if abs(d) > cfg["ablation_psnr_drop_db"] * 2:  # > 0.2 dB 才算"明显"
                    diagnostics.append(
                        f"[实验 C] [WARN] gamma={cfg['exp_c_gamma']} 的 PSNR ({psnr_c:.3f}) "
                        f"与 cond=1.0 ({psnr_1:.3f}) 差 {d:+.3f} dB, "
                        f"动态时序对画质有明显影响, 建议同步修改 TimedC2F.")
                else:
                    diagnostics.append(
                        f"[实验 C] [OK] gamma={cfg['exp_c_gamma']} 与原始动态 gamma 差 {d:+.3f} dB, "
                        f"时序模块影响不大, 可暂时搁置, 优先修 Adapter/残差缩放.")

    summary = {
        "weight_norms": wn,
        "residual_stats": rs,
        "ablation": results.get("ablation"),
        "diagnostics": diagnostics,
        "experiments_enabled": results.get("experiments", []),
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
        # ---- 完整消融表 ----
        lines += ["[3/3] 消融对比 (基线 cond=0.0):"]
        runs_meta = (results.get("ablation") or {}).get("runs", [])
        header = f"  {'key':<22} {'cond':>6} {'gamma':>6}  {'PSNR':>8} {'SSIM':>7} {'LPIPS':>7}  {'d_PSNR_vs_0':>13}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for r in runs_meta:
            k = r["key"]
            m = ab.get(k, {})
            d = m.get("delta_psnr_vs_baseline", float("nan"))
            d_str = f"{d:+.3f} dB" if d == d else "  n/a"
            lines.append(
                f"  {k:<22} {r['cond_scale']:>6.2f} "
                f"{(r['fixed_gamma'] if r['fixed_gamma'] is not None else float('nan')):>6.2f}  "
                f"{m.get('psnr', float('nan')):>8.3f} {m.get('ssim', float('nan')):>7.4f} "
                f"{m.get('lpips', float('nan')):>7.4f}  {d_str:>13}"
            )
        if "lq_to_gt_psnr" in ab:
            lines += ["",
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
