"""C2DControlNet: 在 ControlNet 条件嵌入层后注入 C2F-MAFC 增强模块.

设计要点:
    - 继承 ControlNetModel, 仅修改一个 sum 点:
        原: sample = sample + controlnet_cond
        新: sample = sample + (cond_emb_feat + alpha * proj(C2FBlock(cond_emb_feat)))
    - C2FBlock / proj / alpha 在 __init__ 末尾通过 _init_c2d_modules 初始化
    - 显式逐层权重加载 (废弃 __class__ 转换)
    - DPO deepcopy 兼容性: 所有新增模块均为标准 nn.Module 子模块
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from diffusers import ControlNetModel

from .c2f_block import C2FBlock


class C2DControlNet(ControlNetModel):
    """在 controlnet_cond_embedding 后单点注入 C2F-MAFC 增强模块的 ControlNet.

    新增模块:
        c2f_block:   C2FBlock (粗细分支 + MAFC 多注意力融合)
        proj:        nn.Conv2d 1x1 投影层 (Kaiming 初始化)
        alpha:       可学习标量缩放系数 (init=0, 全程可训练)

    前向改动 (唯一):
        cond_emb_feat = self.controlnet_cond_embedding(controlnet_cond)
        c2f_out = self.c2f_block(cond_emb_feat)
        sample = sample + cond_emb_feat + self.alpha * self.proj(c2f_out)

    其它 (time_proj, time_embedding, conv_in, down_blocks, mid_block,
          controlnet_down_blocks, controlnet_mid_block) 完全沿用父类.
    """

    def __init__(
        self,
        *args,
        c2d_dw_expand: int = 1,
        c2d_ffn_expand: int = 2,
        c2d_dropout: float = 0.0,
        c2d_reduction: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # c 必须 == block_out_channels[0] (与 controlnet_cond_embedding 输出对齐)
        c = self.config.block_out_channels[0]

        # 新增模块
        self.c2f_block = C2FBlock(
            c=c,
            DW_Expand=c2d_dw_expand,
            FFN_Expand=c2d_ffn_expand,
            drop_out_rate=c2d_dropout,
            reduction=c2d_reduction,
        )
        self.proj = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        # 阶段 3: 新增模块初始化
        self._init_c2d_modules()

    # ========================================================================
    # 新增模块初始化 (Stage 3)
    # ========================================================================

    def _init_c2d_modules(self) -> None:
        """C2F 内部 Kaiming normal + β/γ=0; proj Kaiming normal; alpha=0."""
        # C2F 内部
        for m in self.c2f_block.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        # β/γ = 0
        nn.init.zeros_(self.c2f_block.beta)
        nn.init.zeros_(self.c2f_block.gamma)

        # proj: Kaiming normal (与 C2F 内部保持一致)
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        # alpha: 标量 0
        with torch.no_grad():
            self.alpha.zero_()

    # ========================================================================
    # 分层初始化工厂 (Stage 1 + 2 + 3 一站式)
    # ========================================================================

    @classmethod
    def from_unet_c2d(
        cls,
        unet,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: tuple[int, ...] | None = (16, 32, 96, 256),
        conditioning_channels: int = 3,
        c2d_dw_expand: int = 1,
        c2d_ffn_expand: int = 2,
        c2d_dropout: float = 0.0,
        c2d_reduction: int = 8,
    ) -> "C2DControlNet":
        """分层初始化: baseline 权重加载 + C2F/proj/alpha 构造 + 阶段 3 init.

        1) ControlNetModel.from_unet 加载 baseline (UNet 主干权重)
        2) 实例化 C2DControlNet 子类 -> 触发 C2F/proj/alpha 构造
        3) 显式逐层拷贝原生主干权重到子类实例
        4) _init_c2d_modules() (在子类 __init__ 末尾已自动调用)
        """
        # ---- Stage 1: 原生 baseline 加载 ----
        base = ControlNetModel.from_unet(
            unet,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        # ---- Stage 2: 实例化子类, 触发 C2F/proj/alpha 构造 ----
        c2d = cls(
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=(
                unet.config.projection_class_embeddings_input_dim
                if "projection_class_embeddings_input_dim" in unet.config else None
            ),
            mid_block_type=unet.config.mid_block_type,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
            c2d_dw_expand=c2d_dw_expand,
            c2d_ffn_expand=c2d_ffn_expand,
            c2d_dropout=c2d_dropout,
            c2d_reduction=c2d_reduction,
        )

        # ---- 显式逐层拷贝原生主干权重 ----
        c2d.conv_in.load_state_dict(base.conv_in.state_dict())
        c2d.time_proj.load_state_dict(base.time_proj.state_dict())
        c2d.time_embedding.load_state_dict(base.time_embedding.state_dict())
        c2d.down_blocks.load_state_dict(base.down_blocks.state_dict())
        c2d.mid_block.load_state_dict(base.mid_block.state_dict())
        if c2d.class_embedding is not None and base.class_embedding is not None:
            c2d.class_embedding.load_state_dict(base.class_embedding.state_dict())

        # controlnet_cond_embedding / controlnet_down_blocks / controlnet_mid_block
        # 均在父类 __init__ 中以零初始化创建, 子类与之结构一致, 无需拷贝

        del base
        return c2d

    # ========================================================================
    # 前向: 仅修改 sample = sample + controlnet_cond 这一行
    # ========================================================================

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: torch.Tensor | None = None,
        timestep_cond: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ):
        # ---- 1. channel order ----
        channel_order = self.config.controlnet_conditioning_channel_order
        if channel_order == "rgb":
            pass
        elif channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        else:
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # ---- 2. attention_mask ----
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # ---- 3. time embedding ----
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)
            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        # ---- 4. pre-process (此处注入 C2F) ----
        sample = self.conv_in(sample)

        cond_emb_feat = self.controlnet_cond_embedding(controlnet_cond)
        c2f_out = self.c2f_block(cond_emb_feat)
        sample = sample + cond_emb_feat + self.alpha * self.proj(c2f_out)

        # ---- 5. down ----
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # ---- 6. mid ----
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        # ---- 7. control net blocks ----
        controlnet_down_block_res_samples = ()
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)
        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # ---- 8. scaling ----
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)
            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]
        else:
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        from diffusers.models.controlnets.controlnet import ControlNetOutput
        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )