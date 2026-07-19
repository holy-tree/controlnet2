#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import yaml

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

from models.weather_restoration_controlnet import WeatherRestorationControlNet
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import math
import re


# ============================================================
# LoRA (Low-Rank Adaptation) 手动实现 - 不依赖 peft 库
# ============================================================
class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear.

    数学: y = W @ x + (B @ A) @ x * scaling
    其中 W 是冻结的原权重, A (rank x in) 和 B (out x rank) 是 trainable.
    init: A = Kaiming uniform, B = zeros → 初始 LoRA 输出 = 0, 不破坏原模型.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 16, alpha: int = 32,
                 lora_dropout: float = 0.0):
        super().__init__()
        self.original = original_linear
        # 冻结原权重 (Phase 4: UNet 主体保持冻结, 仅 LoRA 训练)
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA 矩阵 (A 随机, B 零)
        # 强制 fp32, 防止 autocast 把 grad 变成 fp16 触发 GradScaler 错误
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Init: A = kaiming, B = 0 → lora_delta 初始为 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        original_out = self.original(x)
        # LoRA delta: x @ A^T @ B^T * scaling
        lora_delta = (self.lora_dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return original_out + lora_delta


def apply_lora_to_unet(unet, rank: int = 16, alpha: int = 32,
                       target_module_names: list = None,
                       lora_dropout: float = 0.0):
    """把 UNet 中指定名称的 nn.Linear 替换为 LoRALinear.

    target_module_names: 要包装的模块短名, 默认 ['to_q', 'to_v', 'to_k', 'to_out.0']
                        覆盖 SD2 UNet 的 self-attn 和 cross-attn 的 Q/K/V/output 投影.
    返回: (包装的模块数, 新增的 trainable 参数数)
    """
    if target_module_names is None:
        target_module_names = ['to_q', 'to_v', 'to_k']

    wrapped_count = 0
    new_trainable_params = 0
    # 按名字找到目标模块并替换
    for name, module in unet.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # name 是类似 "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
        # 取最后一段判断是否要包装
        short_name = name.split('.')[-1]
        if short_name not in target_module_names:
            continue
        # 找 parent module
        parent = unet
        parts = name.split('.')
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1]
        # 替换为 LoRA wrapper
        lora_mod = LoRALinear(module, rank=rank, alpha=alpha, lora_dropout=lora_dropout)
        setattr(parent, attr_name, lora_mod)
        wrapped_count += 1
        # 统计新增 trainable
        new_trainable_params += lora_mod.lora_A.numel() + lora_mod.lora_B.numel()

    return wrapped_count, new_trainable_params

from dataloaders.paired_dataset import PairedCaptionDataset
from dataloaders.dpo_preference_dataset import (
    DPOPreferenceDataset,
    collate_fn_dpo,
)

from ramseesr.utils.metrics import psnr as calc_psnr, ssim as calc_ssim

from utils.dpo import (
    build_ref_controlnet,
    compute_dpo_loss,
)

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0")

logger = get_logger(__name__)

from torchvision import transforms
tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def _log_arca_monitor(controlnet, accelerator, global_step, unet_stds=None):
    """
    ARCA 分层 alpha + 残差 std 监控 (论文 method 章节消融图表数据源).

    表格列:
        name | alpha | tanh(alpha) | scale=last_residual_scale | res_std | unet_std | R
    备注:
        - res_std 来自 self.main_arca[i]._last_residual_stats, 由 ARCA.forward 在 training 模式缓存
        - unet_stds: 7 个 UNet 主干 hidden 标准差 (按 main_arca[0..3] + down_arca[0..1] + mid_arca 顺序).
                     None 时 R 列显示 N/A.
        - R = res_std / unet_std: 控制残差相对 UNet 主干特征的大小.
          R < 0.05 残差太小; 0.05~0.3 健康; > 0.5 残差过大开始污染 SD.
        - 用 accelerator.unwrap_model 取回原模型, 兼容 DDP/Accelerate 包装
    """
    try:
        from models.arca import collect_arca_monitor, format_arca_monitor
        raw = accelerator.unwrap_model(controlnet)
        records = collect_arca_monitor(raw)
        if not records:
            return
        # unet_stds 必须长度匹配 (7), 否则置 None
        if unet_stds is not None and len(unet_stds) != len(records):
            unet_stds = None
        table = format_arca_monitor(records, unet_hidden_stds=unet_stds)
        # logger.info 不支持多行字符串, 用分隔的 print
        for line in table.split("\n"):
            logger.info(f"[ARCA step {global_step:>6d}] {line}")
        # 同时把 alpha 标量 + R 比值推到 tensorboard
        for i, r in enumerate(records):
            log_dict = {f"arca/alpha_{i}": r["alpha"],
                        f"arca/tanh_alpha_{i}": r["tanh_alpha"]}
            if unet_stds is not None and r.get("std") is not None:
                unet_s = unet_stds[i] if unet_stds[i] > 0 else 1e-12
                r_ratio = r["std"] / unet_s
                log_dict[f"arca/R_{i}"] = r_ratio
                log_dict[f"arca/unet_std_{i}"] = unet_stds[i]
            accelerator.log(log_dict, step=global_step)
    except Exception as e:
        logger.warning(f"[ARCA] 监控打印失败: {e}")


@torch.no_grad()
def _capture_unet_baseline_stds(accelerator, controlnet, unet, vae, text_encoder,
                                batch, weight_dtype, noise_scheduler,
                                global_step):
    """
    抓 5 个 UNet 关键 hidden 位置 (down_blocks.0/1/2/3 + mid_block) 的 std.
    跑 1 次额外的 unet forward (使用 ARCA 真实残差),
    抓 5 个点 → 映射到 7 个 ARCA stage.

    返回 7 个 std 列表 (按 main_arca[0..3] + down_arca[0..1] + mid_arca 顺序).
    """
    try:
        raw_unet = accelerator.unwrap_model(unet)
        raw_cn = accelerator.unwrap_model(controlnet)
        if not hasattr(raw_unet, "down_blocks"):
            return None
    except Exception:
        return None

    # 准备 inputs
    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
    encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

    # 随机 timestep + noise (与训练时类似)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, int(noise_scheduler.config.num_train_timesteps),
        (latents.shape[0],), device=latents.device,
    ).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # 用真实 controlnet 算 13 个 down_res + 1 mid_res
    raw_cn.train(False)
    with torch.no_grad():
        down_res, mid_res = raw_cn(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=batch["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype),
            return_dict=False,
        )

    # 注册 post-hook 抓 down_blocks[0..3] + mid_block 输出 hidden
    captured_stds = []  # 5 个, 顺序: [down_0, down_1, down_2, down_3, mid]

    def make_hook():
        def hook(module, input, output):
            # ResNet block 输出: (h, ) tuple 或 tensor
            h = output[0] if isinstance(output, tuple) else output
            captured_stds.append(h.detach().float().std().item())
        return hook

    raw_unet.train(False)
    handles = []
    for i, db in enumerate(raw_unet.down_blocks):
        handles.append(db.register_forward_hook(make_hook()))
    handles.append(raw_unet.mid_block.register_forward_hook(make_hook()))

    # 跑 unet forward with 真实残差 (实际使用时的 hidden)
    with torch.no_grad():
        try:
            _ = raw_unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    d.to(dtype=weight_dtype) for d in down_res
                ],
                mid_block_additional_residual=mid_res.to(dtype=weight_dtype),
            ).sample
        except Exception as e:
            logger.warning(f"[ARCA] UNet baseline forward 失败: {e}")
            for h in handles: h.remove()
            return None

    for h in handles: h.remove()

    if len(captured_stds) < 5:
        return None

    # 5 个 std 映射到 7 个 ARCA stage:
    #   main_arca[0] -> captured_stds[0]  (down_blocks.0: 320ch, 64x64)
    #   main_arca[1] -> captured_stds[1]  (down_blocks.1: 640ch, 32x32)
    #   main_arca[2] -> captured_stds[2]  (down_blocks.2: 1280ch, 16x16)
    #   main_arca[3] -> captured_stds[3]  (down_blocks.3: 1280ch, 8x8)
    #   down_arca[0] -> captured_stds[1]  (与 main_arca[1] 同一 down_block)
    #   down_arca[1] -> captured_stds[2]  (与 main_arca[2] 同一 down_block)
    #   mid_arca     -> captured_stds[4]  (mid_block: 1280ch, 8x8)
    unet_stds = [
        captured_stds[0],  # main_arca.0
        captured_stds[1],  # main_arca.1
        captured_stds[2],  # main_arca.2
        captured_stds[3],  # main_arca.3
        captured_stds[1],  # down_arca.0
        captured_stds[2],  # down_arca.1
        captured_stds[4],  # mid_arca
    ]
    return unet_stds


def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")

        images = []

        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    validation_prompt, validation_image, num_inference_steps=20, generator=generator
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/notebook/data/group/LowLevelLLM/models/diffusion_models/stable-diffusion-2-base",
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experience/controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # ARCA 自适应残差校准模块相关 (论文 method)
    parser.add_argument("--arca_lr", type=float, default=3e-6,
                        help="7 个 stage_alpha 的独立学习率 (建议 2e-6 ~ 5e-6, "
                             "远小于 --learning_rate, 防止 alpha 增长过快).")
    parser.add_argument("--arca_alpha_weight_decay", type=float, default=0.0,
                        help="stage_alpha 的 weight_decay (默认 0, 不应加衰减).")
    parser.add_argument("--gate_lr", type=float, default=5e-4,
                        help="ARCA gate 单独学习率 (直接乘子无 saturation, "
                             "需较高 LR 让 gate 从 1.0 降到目标值 0.3~0.5).")
    parser.add_argument("--gate_weight_decay", type=float, default=0.0,
                        help="ARCA gate 的 weight_decay (默认 0, 让 gate 可自由降到 < 1).")
    parser.add_argument("--use_unet_lora", action="store_true",
                        help="是否给 UNet attention 加 LoRA (Phase 5, +2~4 dB 预期). 默认关闭, 需要显式启用.")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank (默认 16, 典型 8/16/32).")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (通常 = 2*rank).")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout (默认 0, 不用 dropout).")
    parser.add_argument("--lora_target_modules", type=str, default="to_q,to_v",
                        help="要加 LoRA 的模块名 (逗号分隔), 默认 'to_q,to_v' (Q/V 投影).")
    parser.add_argument("--arca_log_interval", type=int, default=500,
                        help="每 N 步打印一次 7 层 alpha + 残差 std 监控表.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", type=lambda x: x.lower() == "true", default=None,
        help="Whether or not to use xformers (default: auto detect)"
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--run_validation_steps",
        type=int,
        default=0,
        help=(
            "Run run_epoch_validation (PSNR/SSIM on real LQ/GT) every X steps. "
            "0 = disable (use epoch-based only). >0 = trigger every X steps regardless of epoch boundary. "
            "Example: 2000 = evaluate PSNR every 2000 training steps."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet_SR",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument("--root_folders",  type=str , default='' )
    parser.add_argument("--null_text_ratio", type=float, default=0.5)
    parser.add_argument("--ram_ft_path", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # ==================== DPO 相关参数 ====================
    parser.add_argument(
        "--train_method", type=str, default="sft",
        choices=["sft", "dpo"],
        help="训练阶段: sft=标准 MSE 监督训练 (默认), dpo=偏好优化",
    )
    parser.add_argument(
        "--sft_controlnet_ckpt", type=str, default=None,
        help="DPO 阶段加载的 SFT 训练好的 controlnet 权重目录 (含 diffusion_pytorch_model.bin)",
    )
    parser.add_argument(
        "--beta_dpo", type=float, default=5000.0,
        help="DPO KL 强度 (Wallace et al. 2023 默认 5000)",
    )
    parser.add_argument(
        "--sft_loss_weight", type=float, default=0.0,
        help="可选: 在 DPO 损失上叠加 GT 监督 MSE 损失, 防止 reward hacking",
    )
    parser.add_argument(
        "--latent_l1_weight", type=float, default=0.1,
        help="SFT 阶段在 noise MSE 上叠加 latent L1 重建损失, 给模型像素/隐空间内容锚定 (建议 0.05~0.3, 0=关闭)",
    )
    parser.add_argument(
        "--lpips_weight", type=float, default=0.05,
        help="SFT 阶段叠加 LPIPS 感知损失权重 (修复高频纹理/细节, 建议 0.03~0.1, 0=关闭)",
    )
    parser.add_argument(
        "--lpips_interval", type=int, default=4,
        help="每隔 N 步算一次 LPIPS, 其余步置 0, 降低 VAE 解码开销",
    )
    parser.add_argument(
        "--freq_loss_weight", type=float, default=0.1,
        help="SFT 阶段叠加 FFT 频域 L1 损失, 保留高频细节 (雨丝/雪粒), "
             "建议 0.05~0.2, 0=关闭",
    )
    parser.add_argument(
        "--lpips_net", type=str, default="alex", choices=["alex", "vgg"],
        help="LPIPS backbone: alex (快) 或 vgg (准)",
    )
    parser.add_argument(
        "--candidates_subdir", type=str, default="candidates",
        help="build_preference.py 写出的候选子目录名",
    )
    parser.add_argument(
        "--top_k_ratio", type=float, default=0.30,
        help="winner 池: 聚合分数前 k%% 候选",
    )
    parser.add_argument(
        "--bottom_k_ratio", type=float, default=0.30,
        help="loser 池: 聚合分数后 k%% 候选",
    )
    parser.add_argument(
        "--normalize_reward", action="store_true",
        default=True,
        help="训练时聚合 reward 先做候选内 min-max 归一化 (避免 PSNR 主导)",
    )
    parser.add_argument(
        "--no_normalize_reward", dest="normalize_reward", action="store_false",
        help="禁用候选内 min-max 归一化 (使用简单加权和)",
    )
    parser.add_argument(
        "--min_gap", type=float, default=0.02,
        help="winner/loser 聚合分数之差小于该值则视为弱偏好, 跳过",
    )
    parser.add_argument(
        "--reward_weight_psnr", type=float, default=0.25,
        help="reward 聚合权重: PSNR (越大越好)",
    )
    parser.add_argument(
        "--reward_weight_ssim", type=float, default=0.25,
        help="reward 聚合权重: SSIM (越大越好)",
    )
    parser.add_argument(
        "--reward_weight_lpips", type=float, default=-0.30,
        help="reward 聚合权重: LPIPS (越小越好, 权重应为负)",
    )
    parser.add_argument(
        "--reward_weight_clip_iqa", type=float, default=0.20,
        help="reward 聚合权重: CLIP-IQA (越大越好)",
    )
    parser.add_argument(
        "--augment_geo", action="store_true",
        help="DPO 阶段启用几何增强 (Flip/Scale/CenterCrop), 严禁颜色增强",
    )
    parser.add_argument(
        "--geo_flip_prob", type=float, default=0.5,
        help="几何增强: 水平翻转概率",
    )
    parser.add_argument(
        "--geo_scale_low", type=float, default=0.9,
        help="几何增强: 短边随机缩放下限",
    )
    parser.add_argument(
        "--geo_scale_high", type=float, default=1.1,
        help="几何增强: 短边随机缩放上限",
    )

    # 新增:三层嵌套数据集配置
    parser.add_argument("--dataset_root", type=str, default="./datasets",
                        help="数据集根目录, 结构: {dataset_root}/{weather}/{split}/{GT,LQ}/")
    parser.add_argument("--weather_types", type=str, nargs="+", default=["rain", "snow", "haze"],
                        help="参与训练的天气类型列表")
    parser.add_argument("--splits", type=str, nargs="+", default=["train"],
                        help="参与训练的数据划分列表")
    parser.add_argument("--rain_num", type=int, default=3,
                        help="rain 数据集使用的样本数 (默认 3)")
    parser.add_argument("--snow_num", type=int, default=3,
                        help="snow 数据集使用的样本数 (默认 3)")
    parser.add_argument("--haze_num", type=int, default=3,
                        help="haze 数据集使用的样本数 (默认 3)")

    # 新增:Prompt 开关
    parser.add_argument("--use_prompt", action="store_true",
                        help="是否使用 prompt 文本引导 (默认 False, 仅图像条件训练)")
    parser.add_argument("--prompt_ratio", type=float, default=0.2,
                        help="use_prompt=True 时使用天气 prompt 的概率 (推荐 0.15~0.25)")
    parser.add_argument("--weather_prompts", type=str, nargs="+", default=None,
                        help="自定义天气 prompt 描述, 格式: rain:desc snow:desc haze:desc")

    # 新增:每个 epoch 结束时的验证
    parser.add_argument("--run_validation", type=lambda x: x.lower() == "true", default=None,
                        help="是否在每个 epoch 结束后进行验证")
    parser.add_argument("--validation_num_samples", type=int, default=4,
                        help="每种天气生成的预测样本数 (验证时采样数)")
    parser.add_argument("--validation_inference_steps", type=int, default=20,
                        help="验证时扩散推理步数")
    parser.add_argument("--validation_guidance_scale", type=float, default=5.5,
                        help="验证时 CFG guidance_scale")
    parser.add_argument("--validation_negative_prompt", type=str,
                        default="dotted, noise, blur, lowres, smooth",
                        help="验证时的 negative prompt")

    # 新增: C2F-MAFC 增强条件编码器参数
    parser.add_argument("--c2d_dw_expand", type=int, default=1,
                        help="C2FBlock 内部 DW_Expand (Stage 1 固定 1)")
    parser.add_argument("--c2d_ffn_expand", type=int, default=2,
                        help="C2FBlock 内部 FFN_Expand")
    parser.add_argument("--c2d_dropout", type=float, default=0.0,
                        help="C2FBlock 内部 dropout 概率")
    parser.add_argument("--c2d_reduction", type=int, default=8,
                        help="C2FBlock 内部 ChannelAttention 通道压缩倍率")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    # if args.dataset_name is not None and args.train_data_dir is not None:
    #     raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def make_train_dataset(args, tokenizer, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }


def _coerce_yaml_value(current, value):
    """
    PyYAML 在解析类似 '5e-5' 时有时会返回字符串而不是 float,
    这里根据 argparse 端的原始类型做一次转换保护。
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(current, float) and isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    if isinstance(current, int) and isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    return value


def main(args):
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(args, key):
                current = getattr(args, key)
                # 只有当 YAML 中的值不为 None 时才覆盖 (None 表示"未设置")
                if value is None:
                    continue
                # list 类型: 优先用 YAML 的 (YAML 一般更明确)
                if isinstance(current, list) and current and not isinstance(value, list):
                    # 当前是默认 list 且 YAML 不是 list (如 bool), 跳过
                    continue
                setattr(args, key, _coerce_yaml_value(current, value))
            else:
                setattr(args, key, value)

    # 解析 weather_prompts (从 dict 或 CLI 的 key:value 列表)
    weather_prompts_dict = None
    if isinstance(args.weather_prompts, list):
        weather_prompts_dict = {}
        for item in args.weather_prompts:
            if ":" in item:
                k, v = item.split(":", 1)
                weather_prompts_dict[k.strip()] = v.strip()
    elif isinstance(args.weather_prompts, dict):
        weather_prompts_dict = args.weather_prompts

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing Weather Restoration ControlNet weights")
        controlnet = WeatherRestorationControlNet.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing Weather Restoration ControlNet from scratch")
        controlnet = WeatherRestorationControlNet(
            c2d_dw_expand=args.c2d_dw_expand,
            c2d_ffn_expand=args.c2d_ffn_expand,
            c2d_dropout=args.c2d_dropout,
            c2d_reduction=args.c2d_reduction,
        )
        controlnet._init_new_modules()

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = WeatherRestorationControlNet.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # === Phase 5: UNet LoRA (大幅突破 SD2 的 restoration 上限, 预期 +2~4 dB)
    #     LoRA 在 attention 的 Q/V (或 Q/K/V) 投影上加低秩适配, 训练量小 (~20M),
    #     SD2 UNet 主体保持冻结, 只有 LoRA 矩阵 (A, B) 训练.
    if args.use_unet_lora:
        target_module_names = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
        n_wrapped, n_params = apply_lora_to_unet(
            unet,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_module_names=target_module_names,
            lora_dropout=args.lora_dropout,
        )
        logger.info(
            f"[Phase 5: LoRA] 包装了 {n_wrapped} 个 Linear 层 "
            f"(target={target_module_names}, rank={args.lora_rank}, alpha={args.lora_alpha}), "
            f"新增 trainable 参数 {n_params:,}"
        )

    # ## init the RAM or DAPE model
    # from ram.models.ram_lora import ram
    # from ram import get_transform
    # if args.ram_ft_path is None:
    #     print("======== USE Original RAM ========")
    # else:
    #     print("==============")
    #     print(f"USE FT RAM FROM: {args.ram_ft_path}")
    #     print("==============")

    # RAM = ram(pretrained='preset/models/ram_swin_large_14m.pth',
    #             pretrained_condition=args.ram_ft_path, 
    #             image_size=384,
    #             vit='swin_l')
    # RAM.eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    # ARCA-aware 分组: 7 个 stage_alpha 用单独小 lr (--arca_lr),
    # 7 个 stage_gate 用单独高 lr (--gate_lr, 直接乘子需要快速学习),
    # 其余 controlnet 参数 (含 7 个 zero_conv + DWConv/PWConv/LN + 编码器) 用 --learning_rate.
    # Phase 5 LoRA: 从 UNet 中收集 LoRA 参数 (lora_A, lora_B), 用主学习率
    alpha_params, gate_params, other_params, lora_params = [], [], [], []
    for name, p in controlnet.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".alpha"):
            alpha_params.append(p)
        elif name.endswith(".gate"):
            gate_params.append(p)
        else:
            other_params.append(p)

    # Phase 5: 收集 UNet LoRA 参数
    if args.use_unet_lora:
        for name, p in unet.named_parameters():
            if not p.requires_grad:
                continue
            # LoRA 参数 (lora_A, lora_B) 在 LoRALinear wrapper 里, 名字含 lora_A / lora_B
            if "lora_A" in name or "lora_B" in name:
                lora_params.append(p)
        if lora_params:
            print(f"[Phase 5: LoRA] 检测到 {len(lora_params)} 个 UNet LoRA 参数, "
                  f"lr={args.learning_rate}")

    if alpha_params:
        print(f"[ARCA] 检测到 {len(alpha_params)} 个 stage_alpha 参数, "
              f"lr={args.arca_lr}, weight_decay={args.arca_alpha_weight_decay}")
    if gate_params:
        print(f"[ARCA] 检测到 {len(gate_params)} 个 stage_gate 参数, "
              f"lr={args.gate_lr}, weight_decay={args.gate_weight_decay}")
    param_groups = [
        {"params": other_params, "lr": args.learning_rate,
         "weight_decay": args.adam_weight_decay},
    ]
    if lora_params:
        param_groups.append({
            "params": lora_params, "lr": args.learning_rate,
            "weight_decay": args.adam_weight_decay,
            "name": "unet_lora",
        })
    if alpha_params:
        param_groups.append({
            "params": alpha_params, "lr": args.arca_lr,
            "weight_decay": args.arca_alpha_weight_decay,
            "name": "arca_alpha",
        })
    if gate_params:
        param_groups.append({
            "params": gate_params, "lr": args.gate_lr,
            "weight_decay": args.gate_weight_decay,
            "name": "arca_gate",
        })
    if not alpha_params and not gate_params:
        # Fallback: 不应该发生, 但保留以防 ARCA 结构变更
        pass
    optimizer = optimizer_class(
        param_groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ==================== DPO: ref_controlnet 构建 ====================
    # 仅在 train_method=dpo 时启用:
    #   1. 加载 SFT 阶段训练好的 controlnet 权重
    #   2. 深拷贝一份作为 ref_controlnet, 冻结, eval, 不参与 optimizer / accelerator.prepare
    ref_controlnet = None
    if args.train_method == "dpo":
        if args.sft_controlnet_ckpt:
            logger.info(f"[DPO] 加载 SFT controlnet 权重 from {args.sft_controlnet_ckpt}")
            # 从原版 unet 初始化结构 (与 main 顶部的 controlnet 一致), 然后 load_state_dict
            sft_cn = WeatherRestorationControlNet.from_pretrained(args.sft_controlnet_ckpt)
            controlnet.load_state_dict(sft_cn.state_dict())
            del sft_cn
        else:
            logger.warning(
                "[DPO] 未指定 --sft_controlnet_ckpt, "
                "将使用当前初始化的 controlnet 作为 ref_controlnet 起点"
            )
        ref_controlnet = build_ref_controlnet(controlnet)
        logger.info("[DPO] ref_controlnet 已构建 (frozen, eval mode)")

    # 按 weather 限制样本数: 优先用 args.<weather>_num, 否则不限制
    weather_num_samples = {}
    for w in args.weather_types:
        attr = f"{w}_num"
        if hasattr(args, attr):
            v = getattr(args, attr)
            if v is not None and v > 0:
                weather_num_samples[w] = v

    if args.train_method == "dpo":
        reward_weights = {
            "psnr":     args.reward_weight_psnr,
            "ssim":     args.reward_weight_ssim,
            "lpips":    args.reward_weight_lpips,
            "clip_iqa": args.reward_weight_clip_iqa,
        }
        train_dataset = DPOPreferenceDataset(
            dataset_root=args.dataset_root,
            weather_types=args.weather_types,
            splits=args.splits,
            tokenizer=tokenizer,
            resolution=args.resolution,
            candidates_subdir=args.candidates_subdir,
            reward_weights=reward_weights,
            top_k_ratio=args.top_k_ratio,
            bottom_k_ratio=args.bottom_k_ratio,
            min_gap=args.min_gap,
            normalize=args.normalize_reward,
            augment_geo=args.augment_geo,
            geo_flip_prob=args.geo_flip_prob,
            geo_scale_range=(args.geo_scale_low, args.geo_scale_high),
            null_text_ratio=args.null_text_ratio,
            use_prompt=args.use_prompt,
            prompt_ratio=args.prompt_ratio,
            weather_prompts=weather_prompts_dict,
            weather_num_samples=weather_num_samples,
        )
        train_collate_fn = collate_fn_dpo
    else:
        train_dataset = PairedCaptionDataset(
            dataset_root=args.dataset_root,
            weather_types=args.weather_types,
            splits=args.splits,
            tokenizer=tokenizer,
            null_text_ratio=args.null_text_ratio,
            use_prompt=args.use_prompt,
            prompt_ratio=args.prompt_ratio,
            weather_prompts=weather_prompts_dict,
            resolution=args.resolution,
            weather_num_samples=weather_num_samples,
        )
        train_collate_fn = None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.dataloader_num_workers,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    # RAM.to(accelerator.device, dtype=weight_dtype)

    # ==================== DPO: ref_controlnet 同步到 device / dtype ====================
    if ref_controlnet is not None:
        ref_controlnet.to(accelerator.device, dtype=weight_dtype)

    # ==================== LPIPS 感知损失模型加载 (SFT 阶段使用) ====================
    # LPIPS 保持 fp32, 避免 fp16 数值不稳; 一次加载复用
    lpips_model = None
    if args.lpips_weight > 0.0:
        try:
            import lpips as lpips_pkg
            lpips_model = lpips_pkg.LPIPS(net=args.lpips_net, verbose=False).to(accelerator.device)
            lpips_model.eval()
            for p in lpips_model.parameters():
                p.requires_grad_(False)
            logger.info(f"[LPIPS] 加载完成: net={args.lpips_net}, weight={args.lpips_weight}, interval={args.lpips_interval}")
        except Exception as e:
            logger.warning(f"[LPIPS] 加载失败 ({e}), 已关闭感知损失")
            lpips_model = None

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        # 过滤掉 tensorboard add_hparams 不支持的类型 (Path/None/复杂对象等)
        def _to_tb_value(v):
            if v is None:
                return "None"
            if isinstance(v, (str, bool, int, float)):
                return v
            if isinstance(v, list):
                return ",".join(str(x) for x in v)
            return str(v)
        tracker_config = {k: _to_tb_value(v) for k, v in tracker_config.items()}

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            full_ckpt_path = os.path.join(args.output_dir, path)
            full_ckpt_path_obj = Path(full_ckpt_path)

            # === Patch: 当模型新增参数 (如 ARCA gate) 时, 旧 optimizer state 大小不匹配
            #     会报 "parameter group that doesn't match the size of optimizer's group".
            #     处理方案:
            #       1) 备份 scheduler.bin (LR 状态很重要, 必须保留)
            #       2) 删除 optimizer.bin (让 load_state 不抛 "group size" 错误)
            #       3) load_state 会因缺失 optimizer 抛 FileNotFoundError, 接住
            #       4) 手动恢复 scheduler.bin (避免 LR 从 schedule 起点重置!)
            #     scaler.pt 和 random_states 没恢复 (前 1k step 内可自校正)
            scheduler_bin = full_ckpt_path_obj / "scheduler.bin"
            if scheduler_bin.exists():
                sched_backup = full_ckpt_path_obj / "scheduler.bin.bak"
                if not sched_backup.exists():
                    shutil.copy2(scheduler_bin, sched_backup)
                    accelerator.print(f"[patch] 备份 scheduler → {sched_backup.name}")
                # 删除 scheduler.bin: 避免老 cosine state 加载到新 constant scheduler
                # (lambda(last_epoch) 会给出错误的 LR). 新 scheduler 从 base_lr=5e-5 开始.
                scheduler_bin.unlink()
                accelerator.print(
                    "[patch] 已删除 scheduler.bin (新 scheduler 从 base_lr 开始, "
                    "避免老 state 类型不兼容)"
                )

            opt_bin = full_ckpt_path_obj / "optimizer.bin"
            if opt_bin.exists():
                opt_backup = full_ckpt_path_obj / "optimizer.bin.bak"
                if not opt_backup.exists():
                    shutil.copy2(opt_bin, opt_backup)
                    accelerator.print(f"[patch] 备份 optimizer → {opt_backup.name}")
                opt_bin.unlink()
                accelerator.print(
                    "[patch] 已删除 optimizer.bin (新参数 gate 不兼容旧 Adam state, "
                    "模型权重已加载, Adam 重新初始化)"
                )

            try:
                accelerator.load_state(full_ckpt_path)
            except FileNotFoundError as e:
                # optimizer.bin / scheduler.bin 已删, accelerate 跳过加载, 这里吃掉 FileNotFoundError
                err_msg = str(e).lower()
                if "optimizer" in err_msg or "scheduler" in err_msg:
                    accelerator.print(f"[patch] 跳过缺失文件: {e}")
                else:
                    raise

            # 不再手动恢复 scheduler.bin: 它已被删除, 新的 scheduler (constant) 从 base_lr 开始
            # 打印 LR 验证
            last_lrs = lr_scheduler.get_last_lr()
            accelerator.print(
                f"[patch] 新 scheduler 启动 LR = {last_lrs[0]:.2e} "
                f"(constant schedule, 不会衰减)"
            )

            # === Patch: 把所有 gate 重置为 1.0 (直接乘子等价于无 gate)
            #     因为 checkpoint 里的 gate 可能不是 1.0 (旧 sigmoid init 或上次错误值),
            #     新代码用直接乘子, gate=4.0 会让 residual 放 4x (破坏模型),
            #     所以必须重置到 1.0 才能保证初始行为与无 gate 一致.
            with torch.no_grad():
                reset_count = 0
                for n, p in controlnet.named_parameters():
                    if n.endswith(".gate"):
                        old_val = p.data.item()
                        if abs(old_val - 1.0) > 1e-6:
                            p.data.fill_(1.0)
                            reset_count += 1
                if reset_count > 0:
                    accelerator.print(
                        f"[patch] 重置 {reset_count} 个 gate 参数为 1.0 "
                        f"(直接乘子无 gate 等价行为)"
                    )

                # === Patch: 把 proj_128 权重置 0
                #     老 checkpoint 加载时 proj_128 是随机初始化 (旧代码里没设 0 init),
                #     随机权重注入 F64 会污染已学特征 (实测让 PSNR 从 18.11 跌到 14)
                #     重置为 0 后初始行为 = 无 F128 注入, 与 baseline 一致
                #     训练过程中 proj_128 慢慢学到非零权重, F128 贡献逐渐出现
                proj_reset_count = 0
                for n, p in controlnet.named_parameters():
                    # Phase 4.1: 同时重置 proj_128 和 f128_refine (上采样 conv)
                    if (n.endswith("proj_128.weight")
                            or n.endswith("proj_128.bias")
                            or n.endswith("f128_refine.1.weight")):
                        if p.data.abs().max() > 1e-6:
                            p.data.zero_()
                            proj_reset_count += 1
                if proj_reset_count > 0:
                    accelerator.print(
                        f"[patch] 重置 {proj_reset_count} 个 F128 相关权重为 0 "
                        f"(proj_128 + f128_refine, 避免随机权重污染 F64)"
                    )

            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                # Convert images to latent space
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # ==================== DPO 关键不变量 ====================
                # winner / loser 必须共享同一 timestep / noise, 否则 DPO 损失会爆炸.
                # collate_fn_dpo 已沿 batch 维把 winner/loser 拼成 [2*B, ...], 这里强制对齐.
                if args.train_method == "dpo":
                    timesteps = timesteps.chunk(2, dim=0)[0].repeat(2)
                    noise = noise.chunk(2, dim=0)[0].repeat(2, 1, 1, 1)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

                controlnet_image = batch["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.train_method == "dpo":
                    # ---------- DPO 损失 ----------
                    # 1) ref_controlnet 前向 (no_grad)
                    with torch.no_grad():
                        down_res_ref, mid_res_ref = ref_controlnet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            controlnet_cond=controlnet_image,
                            return_dict=False,
                        )
                        ref_pred = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            down_block_additional_residuals=[
                                s.to(dtype=weight_dtype) for s in down_res_ref
                            ],
                            mid_block_additional_residual=mid_res_ref.to(dtype=weight_dtype),
                        ).sample

                    loss_dpo, implicit_acc, _ = compute_dpo_loss(
                        model_pred, ref_pred, target, beta_dpo=args.beta_dpo,
                    )

                    # 2) 可选: SFT 正则 (用 GT 做 MSE, 防止 reward hacking)
                    if args.sft_loss_weight > 0.0:
                        # 选取 winner 部分 [B] 与 GT [B] 算 MSE
                        model_pred_winner = model_pred.chunk(2, dim=0)[0]
                        # GT 的 latent 在 chunk(2) 之后只有 B, 需要让 dataloader 提供 GT
                        if "pixel_values_gt" in batch:
                            gt_pixels = batch["pixel_values_gt"].to(
                                accelerator.device, dtype=weight_dtype
                            )
                            gt_latents = (
                                vae.encode(gt_pixels).latent_dist.sample()
                                * vae.config.scaling_factor
                            )
                            gt_noise = torch.randn_like(gt_latents)
                            gt_t = torch.randint(
                                0, noise_scheduler.config.num_train_timesteps,
                                (gt_latents.shape[0],), device=latents.device,
                            ).long()
                            gt_noisy = noise_scheduler.add_noise(
                                gt_latents, gt_noise, gt_t,
                            )
                            # 用 winner 的 encoder_hidden_states / lq 重新前向
                            enc_w = encoder_hidden_states.chunk(2, dim=0)[0]
                            lq_w = controlnet_image.chunk(2, dim=0)[0]
                            d_w, m_w = controlnet(
                                gt_noisy, gt_t,
                                encoder_hidden_states=enc_w,
                                controlnet_cond=lq_w,
                                return_dict=False,
                            )
                            pred_gt = unet(
                                gt_noisy, gt_t,
                                encoder_hidden_states=enc_w,
                                down_block_additional_residuals=[
                                    s.to(dtype=weight_dtype) for s in d_w
                                ],
                                mid_block_additional_residual=m_w.to(dtype=weight_dtype),
                            ).sample
                            loss_sft = F.mse_loss(
                                pred_gt.float(), gt_noise.float(), reduction="mean"
                            )
                            loss = loss_dpo + args.sft_loss_weight * loss_sft
                        else:
                            loss = loss_dpo
                    else:
                        loss = loss_dpo
                else:
                    # ---------- SFT 损失: noise MSE + latent L1 + LPIPS ----------
                    # noise MSE: 原有扩散损失, 约束轨迹
                    loss_mse = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # latent L1: 从 noise 预测还原 x0, 在隐空间与 GT 对齐,
                    # 弥补 noise MSE 不约束最终 RGB 的缺陷, 直接缓解色彩/纹理漂移
                    loss_l1 = torch.tensor(0.0, device=model_pred.device)
                    pred_x0 = None
                    if args.latent_l1_weight > 0.0:
                        alphas_cumprod = noise_scheduler.alphas_cumprod.to(model_pred.device)
                        alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1).float()
                        sqrt_alpha = alpha_t.sqrt()
                        sqrt_one_minus_alpha = (1.0 - alpha_t).sqrt()
                        # x_t = sqrt(a)*x0 + sqrt(1-a)*eps  =>  x0 = (x_t - sqrt(1-a)*eps) / sqrt(a)
                        pred_x0 = (noisy_latents.float() - sqrt_one_minus_alpha * model_pred.float()) / sqrt_alpha
                        pred_x0 = pred_x0.clamp(-3.0, 3.0)
                        loss_l1 = F.l1_loss(pred_x0, latents.float(), reduction="mean")

                    loss = loss_mse + args.latent_l1_weight * loss_l1

                    # Frequency (FFT) loss: 在隐空间频域对齐, 保留高频细节
                    # (雨丝/雪粒). 与 latent L1 互补, L1 保幅值, FFT 保频谱.
                    loss_freq = torch.tensor(0.0, device=model_pred.device)
                    if args.freq_loss_weight > 0.0 and pred_x0 is not None:
                        # rfft2: 实数输入 → 复数输出, 形状 [B, C, H, W//2+1]
                        # 取幅值 (相位信息对内容重建帮助小, 幅值更稳定)
                        pred_fft = torch.fft.rfft2(pred_x0, norm="ortho")
                        tgt_fft = torch.fft.rfft2(latents.float(), norm="ortho")
                        loss_freq = F.l1_loss(pred_fft.abs(), tgt_fft.abs(), reduction="mean")

                    loss = loss + args.freq_loss_weight * loss_freq

                    # LPIPS 感知损失: 每 N 步算一次, 解码 pred_x0/latents 到 RGB,
                    # 在感知特征空间对齐 GT, 修复高频纹理/细节.
                    # VAE.decode 不参与梯度回传到这里 (用 pred_x0 反传),
                    # 仍受 fp16 数值影响, 所以反传时转 fp32 再 clamp.
                    loss_lpips = torch.tensor(0.0, device=model_pred.device)
                    if (lpips_model is not None
                            and args.lpips_weight > 0.0
                            and pred_x0 is not None
                            and (global_step % args.lpips_interval == 0)):
                        try:
                            # 清碎片, 给 LPIPS+VAE 解码留显存 (LoRA 已占不少)
                            torch.cuda.empty_cache()
                            with torch.no_grad():
                                # decode 走 no_grad 避免 VAE 内部存大 activation map
                                scaling = vae.config.scaling_factor
                                pred_rgb = vae.decode(pred_x0.to(weight_dtype) / scaling).sample.float().clamp(-1, 1)
                                gt_rgb = vae.decode(latents.to(weight_dtype) / scaling).sample.float().clamp(-1, 1)
                            # LPIPS 内部已经把 [-1, 1] 映射到感知空间
                            d = lpips_model(pred_rgb, gt_rgb)
                            loss_lpips = d.mean()
                        except Exception as e:
                            logger.warning(f"[LPIPS] 计算失败 ({e}), 跳过本步")
                            loss_lpips = torch.tensor(0.0, device=model_pred.device)

                    loss = loss + args.lpips_weight * loss_lpips

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # 限制 checkpoint 数量: 按 step 升序保留最新 N 个, 删最旧的
                        # (与 HF Trainer.save_total_limit 等价; HF Trainer 在 --train_method dpo 时也是用同名参数,
                        # 但本项目是自定义循环, 名字叫 checkpoints_total_limit)
                        if args.checkpoints_total_limit is not None and args.checkpoints_total_limit > 0:
                            ckpts = sorted(
                                [
                                    d for d in os.listdir(args.output_dir)
                                    if d.startswith("checkpoint-")
                                    and os.path.isdir(os.path.join(args.output_dir, d))
                                ],
                                key=lambda x: int(x.split("-")[1]),
                            )
                            if len(ckpts) > args.checkpoints_total_limit:
                                num_to_remove = len(ckpts) - args.checkpoints_total_limit
                                rm_list = ckpts[:num_to_remove]
                                logger.info(
                                    f"checkpoints_total_limit={args.checkpoints_total_limit} 已超, "
                                    f"删除最旧的 {num_to_remove} 个: {rm_list}"
                                )
                                for d in rm_list:
                                    p = os.path.join(args.output_dir, d)
                                    try:
                                        shutil.rmtree(p)
                                    except Exception as e:
                                        logger.warning(f"删除 {p} 失败: {e}")
                    # if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    if False:
                        image_logs = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

                    # ===== 按 step 评估 PSNR/SSIM (放在内层循环内, 这样 global_step % N == 0 时才被检查) =====
                    if (args.run_validation
                            and args.run_validation_steps > 0
                            and global_step > 0
                            and global_step % args.run_validation_steps == 0):
                        controlnet.eval()
                        unet.eval()
                        vae.eval()
                        run_epoch_validation(
                            vae, unet, controlnet, text_encoder, tokenizer,
                            accelerator, weight_dtype, args, epoch, train_dataset
                        )
                        controlnet.train()
                        unet.train()
                        vae.train()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.train_method == "dpo":
                logs["implicit_acc"] = float(implicit_acc.detach().item())
            else:
                if args.latent_l1_weight > 0.0:
                    logs["loss_mse"] = loss_mse.detach().item()
                    logs["loss_l1"] = loss_l1.detach().item()
                if args.freq_loss_weight > 0.0:
                    logs["loss_freq"] = loss_freq.detach().item()
                if args.lpips_weight > 0.0:
                    logs["loss_lpips"] = loss_lpips.detach().item()

            # ARCA 监控: 每 N 步打印 7 层 alpha + 残差 std + UNet 残差比 R (论文消融图表数据源)
            if (args.arca_log_interval > 0
                    and global_step > 0
                    and global_step % args.arca_log_interval == 0
                    and accelerator.is_main_process):
                # 抓 5 个 UNet 关键 hidden 位置的 std (用于 R = res_std/unet_std 监控)
                unet_stds = _capture_unet_baseline_stds(
                    accelerator, controlnet, unet, vae, text_encoder,
                    batch, weight_dtype, noise_scheduler, global_step,
                )
                _log_arca_monitor(controlnet, accelerator, global_step, unet_stds=unet_stds)

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        # ===== Epoch 结束: 验证 + PSNR/SSIM (按 step 评估开启时跳过, 避免重复) =====
        if (args.run_validation
                and args.run_validation_steps == 0):
            # 确保 controlnet / unet / vae 处于 eval 模式
            controlnet.eval()
            unet.eval()
            vae.eval()
            run_epoch_validation(
                vae, unet, controlnet, text_encoder, tokenizer,
                accelerator, weight_dtype, args, epoch, train_dataset
            )
            # 恢复 train 模式
            controlnet.train()
            unet.train()
            vae.train()

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


@torch.no_grad()
def run_epoch_validation(vae, unet, controlnet, text_encoder, tokenizer, accelerator, weight_dtype, args, epoch, train_dataset):
    """
    每个 epoch 结束后调用:
      1. 对 rain/snow/haze 三种天气,各生成 N 张 pred 图,保存到 validation/<timestamp>/<weather>/ 下
      2. 计算验证集上的 PSNR / SSIM (基于生成的 pred 与 GT 比较)
    """
    if not accelerator.is_main_process:
        return

    logger.info(f"[Epoch {epoch}] 开始验证 ...")

    # 创建本次验证的输出目录: output_dir/validation/<timestamp>_epoch<epoch>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    val_root = Path(args.output_dir) / "validation" / f"{timestamp}_epoch{epoch}"
    val_root.mkdir(parents=True, exist_ok=True)

    # 创建 inference pipeline (与 test.py 一致)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=accelerator.unwrap_model(controlnet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    num_samples = args.validation_num_samples
    weather_metrics = {}  # weather -> {"psnr": [...], "ssim": [...]}

    for weather in args.weather_types:
        # 从 train_dataset.samples 中按 weather 抽取 num_samples 个样本
        candidates = [s for s in train_dataset.samples if s[2] == weather]
        if not candidates:
            logger.warn(f"[Epoch {epoch}] 没有 {weather} 类别的样本, 跳过")
            continue

        # 固定种子以便跨 epoch 复现 val 集:
        #   - 种子只依赖 weather (与 epoch 无关), 保证同一 weather 在所有 epoch 用同一组 val 图
        #   - 用 hashlib 取代 hash() 以避免 Python 3.3+ 默认 hash 随机化 (PYTHONHASHSEED=random)
        #   - base 0 + weather-derived offset, 保证三种 weather 的 val 集互不重叠
        import hashlib
        weather_seed = int(hashlib.md5(weather.encode('utf-8')).hexdigest()[:8], 16) % (2**31)
        random.seed(weather_seed)
        selected = random.sample(candidates, min(num_samples, len(candidates)))

        weather_dir = val_root / weather
        weather_dir.mkdir(parents=True, exist_ok=True)

        psnr_list, ssim_list = [], []

        for sample_idx, (gt_path, lq_path, _) in enumerate(selected):
            # 读取 LQ 和 GT
            from torchvision import transforms as tvt
            preprocess = tvt.Compose([
                tvt.Resize(args.resolution, interpolation=tvt.InterpolationMode.BILINEAR),
                tvt.CenterCrop(args.resolution),
                tvt.ToTensor(),
            ])
            lq_img = preprocess(Image.open(lq_path).convert("RGB"))
            gt_img = preprocess(Image.open(gt_path).convert("RGB"))

            # GT 归一化到 [-1, 1] 后通过 VAE 重建 -> 比较"重建的 GT"与"模型生成的 pred"
            # 实际上: pred 是从 LQ 生成的恢复图, 我们将其与 GT 在像素空间比较
            prompt = ""
            if args.use_prompt and random.random() < args.prompt_ratio:
                prompt = train_dataset.weather_prompts.get(weather, "")

            # 把 LQ 喂给 pipeline, 生成 pred
            lq_pil = tvt.ToPILImage()(lq_img)
            with torch.autocast("cuda"):
                pred_pil = pipeline(
                    prompt,
                    lq_pil,
                    num_inference_steps=args.validation_inference_steps,
                    guidance_scale=args.validation_guidance_scale,
                    negative_prompt=args.validation_negative_prompt,
                    height=args.resolution,
                    width=args.resolution,
                ).images[0]

            # pred -> tensor [3,H,W] in [0,1]
            pred_tensor = tvt.ToTensor()(pred_pil).to(accelerator.device).clamp(0, 1)

            # 计算 PSNR / SSIM (pred vs gt, 都在 [0,1])
            p = calc_psnr(pred_tensor, gt_img.to(accelerator.device))
            s = calc_ssim(pred_tensor, gt_img.to(accelerator.device))
            psnr_list.append(p)
            ssim_list.append(s)

            # 保存 pred 和 GT LQ 图像
            stem = Path(gt_path).stem
            pred_pil.save(weather_dir / f"{sample_idx:03d}_{stem}_pred.png")
            lq_pil.save(weather_dir / f"{sample_idx:03d}_{stem}_lq.png")
            # 也保存 GT 作为参考 (便于人工查看)
            gt_pil = tvt.ToPILImage()(gt_img)
            gt_pil.save(weather_dir / f"{sample_idx:03d}_{stem}_gt.png")

        if psnr_list:
            avg_p = sum(psnr_list) / len(psnr_list)
            avg_s = sum(ssim_list) / len(ssim_list)
            weather_metrics[weather] = {"psnr": avg_p, "ssim": avg_s}
            logger.info(f"[Epoch {epoch}] [{weather}] PSNR={avg_p:.3f} dB, SSIM={avg_s:.4f} (n={len(psnr_list)})")

    # 写一个汇总 JSON / txt
    summary_path = val_root / "metrics.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Num samples per weather: {num_samples}\n")
        f.write(f"Inference steps: {args.validation_inference_steps}\n")
        f.write(f"Guidance scale: {args.validation_guidance_scale}\n\n")
        f.write("Per-weather metrics:\n")
        for weather, m in weather_metrics.items():
            f.write(f"  {weather:8s}  PSNR={m['psnr']:.3f} dB  SSIM={m['ssim']:.4f}\n")
        if weather_metrics:
            avg_psnr = sum(m["psnr"] for m in weather_metrics.values()) / len(weather_metrics)
            avg_ssim = sum(m["ssim"] for m in weather_metrics.values()) / len(weather_metrics)
            f.write(f"\nAverage:        PSNR={avg_psnr:.3f} dB  SSIM={avg_ssim:.4f}\n")

    # 也通过 accelerator.log 记录到 tensorboard / wandb
    log_dict = {f"val/{w}/psnr": m["psnr"] for w, m in weather_metrics.items()}
    log_dict.update({f"val/{w}/ssim": m["ssim"] for w, m in weather_metrics.items()})
    if weather_metrics:
        log_dict["val/avg_psnr"] = sum(m["psnr"] for m in weather_metrics.values()) / len(weather_metrics)
        log_dict["val/avg_ssim"] = sum(m["ssim"] for m in weather_metrics.values()) / len(weather_metrics)
    accelerator.log(log_dict, step=epoch)

    logger.info(f"[Epoch {epoch}] 验证完成, 结果保存到: {val_root}")
    del pipeline
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    main(args)
