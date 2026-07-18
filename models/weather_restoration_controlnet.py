"""Weather Restoration ControlNet: 退化感知 + C2F + 时序调制 + 分层残差注入.

完整前向链路:
    LQ RGB (B, 3, 512, 512)
        ↓
    WeatherDegradationEncoder → [F64, F32, F16, F8]
        ↓
    Per stage: TimedC2FBlock + LightweightAdapter
        ↓
    4 主零卷积 (per scale) + 2 下采样位置零卷积 (channel projection) + 1 mid 零卷积
        ↓
    12 个 down_block_res_samples + 1 个 mid_block_res_sample
        ↓
    喂入 SD2 UNet (down_block_additional_residuals, mid_block_additional_residual)

类设计:
    - 独立 nn.Module (不继承 ControlNetModel), forward 签名与 ControlNetModel 完全兼容
    - train_controlnet.py / StableDiffusionControlNetPipeline 均可 duck-type 使用
    - 7 个零卷积全部 zero-init, 训练初期整体输出恒为 0, 等价"无 ControlNet"

SD2 UNet 期望的 12 down_res 位置 (block_out_channels=(320,640,1280,1280), layers=2):
    pos 0, 1, 2  (320ch, 64x64)  ← F64  (init, block0_res1, block0_res2)
    pos 3        (320ch, 32x32)  ← F32 → 320ch 投影 (block0_downsample 输出)
    pos 4, 5     (640ch, 32x32)  ← F32  (block1_res1, block1_res2)
    pos 6        (640ch, 16x16)  ← F16 → 640ch 投影 (block1_downsample 输出)
    pos 7, 8     (1280ch, 16x16) ← F16  (block2_res1, block2_res2)
    pos 9, 10, 11(1280ch, 8x8)   ← F8   (block2_downsample, block3_res1, block3_res2)
    mid          (1280ch, 8x8)   ← F8   (mid_block)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from diffusers import ControlNetModel

from .arca import ARCAResidualCalibrator, LightweightAdapter
from .c2f_block import (
    LightweightAdapter as _LegacyLightweightAdapter,
    TimedC2FBlock,
    zero_conv,
)
from .weather_encoder import WeatherDegradationEncoder


class WeatherRestorationControlNet(ControlNetModel):
    """退化感知 ControlNet: Weather Encoder + 多尺度 Timed-C2F + 分层残差注入.

    继承 ControlNetModel 仅用于通过 diffusers pipeline 的 isinstance 检查
    (pipeline_controlnet.py:655 要求 self.controlnet 是 ControlNetModel 实例).

    实现技巧:
        - 跳过 ControlNetModel.__init__ 的重型初始化 (不创建 down_blocks/mid_block 等)
        - 直接调用 nn.Module.__init__ + 手动构建 self.config
        - self.forward 完全重写, 不依赖任何父类构造的子模块
        - 父类的 down_blocks / mid_block / controlnet_down_blocks 等属性不会被创建,
          pipeline.__call__ 也不会访问它们 (pipeline 只调用 forward 并使用返回值)

    与原生 ControlNet 接口完全兼容:
        forward(sample, timestep, encoder_hidden_states, controlnet_cond, ...)
        返回 (down_block_res_samples, mid_block_res_sample)
    其中:
        - sample, encoder_hidden_states: 兼容参数, 本类不使用 (无 cross-attn / 无 time FiLM)
        - controlnet_cond: LQ RGB 图 (B, 3, 512, 512), 实际驱动整个网络
    """

    SD2_STAGE_CHANNELS = (320, 640, 1280, 1280)

    def __init__(
        self,
        in_channels: int = 3,
        # TimedC2F 内部参数
        c2d_dw_expand: int = 1,
        c2d_ffn_expand: int = 2,
        c2d_dropout: float = 0.0,
        c2d_reduction: int = 8,
        c2d_time_dim: int = 128,
        # 兼容 ControlNet pipeline 的接口字段 (本类不使用)
        cross_attention_dim: int = 1280,
        block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280),
        conditioning_channels: int = 3,
    ):
        # 关键: 跳过 ControlNetModel.__init__ 的重型初始化, 直接走 nn.Module.__init__
        # 这样 isinstance(cn, ControlNetModel) 仍为 True (通过继承),
        # 但不会浪费显存/时间去构建 down_blocks / mid_block / controlnet_down_blocks 等
        nn.Module.__init__(self)
        self.use_arca = True

        # ---- 1. 退化感知编码器 ----
        self.weather_encoder = WeatherDegradationEncoder(in_channels=in_channels)

        # ---- 2. 四级 Timed-C2F ----
        self.timed_c2f_blocks = nn.ModuleList([
            TimedC2FBlock(
                c=ch, DW_Expand=c2d_dw_expand, FFN_Expand=c2d_ffn_expand,
                drop_out_rate=c2d_dropout, reduction=c2d_reduction,
                time_dim=c2d_time_dim,
            )
            for ch in self.SD2_STAGE_CHANNELS
        ])

        # ---- 3. ARCA: 4 主 + 2 投影 + 1 mid = 7 个独立 stage, 每个含 1 个可学习 alpha ----
        # 4 个主 stage (与 4 个 c2f 块一一对应)
        self.main_arca = nn.ModuleList([
            ARCAResidualCalibrator(320, 320),     # F64
            ARCAResidualCalibrator(640, 640),     # F32
            ARCAResidualCalibrator(1280, 1280),   # F16
            ARCAResidualCalibrator(1280, 1280),   # F8
        ])
        # 2 个下采样投影 (从下一 stage 投影通道)
        self.down_arca = nn.ModuleList([
            ARCAResidualCalibrator(640, 320),     # F32 -> 320
            ARCAResidualCalibrator(1280, 640),    # F16 -> 640
        ])
        # 1 个 mid
        self.mid_arca = ARCAResidualCalibrator(1280, 1280)

        # ---- 4. config (供 pipeline / save_pretrained 读取) ----
        # 通过 register_to_config 注入 config (ConfigMixin 标准做法),
        # 避免 self.config = ... 触发 ConfigMixin 的 read-only property 错误.
        from diffusers.configuration_utils import FrozenDict
        self._internal_dict = FrozenDict({
            "in_channels": in_channels,
            "cross_attention_dim": cross_attention_dim,
            "block_out_channels": tuple(block_out_channels),
            "conditioning_channels": conditioning_channels,
            # 下面这些字段是 pipeline / save_pretrained 可能读到的占位字段
            "down_block_types": ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                                  "CrossAttnDownBlock2D", "DownBlock2D"),
            "sample_size": None,
            "transformer_layers_per_block": 1,
            "attention_head_dim": 8,
            "num_attention_heads": None,
            "use_linear_projection": False,
            "class_embed_type": None,
            "num_class_embeds": None,
            "upcast_attention": False,
            "resnet_time_scale_shift": "default",
            "projection_class_embeddings_input_dim": None,
            "mid_block_type": "UNetMidBlock2DCrossAttn",
            "controlnet_conditioning_channel_order": "rgb",
            "conditioning_embedding_out_channels": (16, 32, 96, 256),
            "global_pool_conditions": False,
            "encoder_hid_dim": None,
            "encoder_hid_dim_type": None,
            "addition_embed_type": None,
            "addition_time_embed_dim": None,
            "act_fn": "silu",
            "norm_num_groups": 32,
            "norm_eps": 1e-5,
            "downsample_padding": 1,
            "mid_block_scale_factor": 1.0,
            "only_cross_attention": False,
            "loading_state_dict": False,
            "_class_name": "WeatherRestorationControlNet",
            "_diffusers_version": "0.25.0",
        })

        # 注意: 不要手动设置 self.dtype, ControlNetModel 继承自 ModelMixin,
        #       dtype 是 property, 动态从参数 dtype 取值.

    # ------------------------------------------------------------------------
    # 类级 property: .config 返回 _internal_dict, 让 diffusers pipeline
    # 的 controlnet.config.X 访问链路 (如 controlnet.config.global_pool_conditions)
    # 在 skip __init__ 后仍能工作.
    # ------------------------------------------------------------------------
    @property
    def config(self):  # type: ignore[override]
        return self._internal_dict

    # ------------------------------------------------------------------------
    # 类级 property: 把 7 个 zero_conv 通过统一命名暴露, 兼容旧代码 / 诊断脚本.
    #   - ARCA 模式: 指向 main_arca[i] / down_arca[i] / mid_arca 内部的 zero_conv
    #   - 旧模式: 指向 __init__ 里直接赋的 self.zero_conv_stage_0 (Conv2d 实例)
    #   - state_dict 不会收集 property, ARCA 模式不会有重复 key
    # ------------------------------------------------------------------------
    @property
    def zero_conv_stage_0(self):  # type: ignore[override]
        if hasattr(self, "main_arca"):
            return self.main_arca[0].zero_conv
        return self.__dict__["zero_conv_stage_0"]

    @property
    def zero_conv_stage_1(self):  # type: ignore[override]
        if hasattr(self, "main_arca"):
            return self.main_arca[1].zero_conv
        return self.__dict__["zero_conv_stage_1"]

    @property
    def zero_conv_stage_2(self):  # type: ignore[override]
        if hasattr(self, "main_arca"):
            return self.main_arca[2].zero_conv
        return self.__dict__["zero_conv_stage_2"]

    @property
    def zero_conv_stage_3(self):  # type: ignore[override]
        if hasattr(self, "main_arca"):
            return self.main_arca[3].zero_conv
        return self.__dict__["zero_conv_stage_3"]

    @property
    def zero_conv_down_0(self):  # type: ignore[override]
        if hasattr(self, "down_arca"):
            return self.down_arca[0].zero_conv
        return self.__dict__["zero_conv_down_0"]

    @property
    def zero_conv_down_1(self):  # type: ignore[override]
        if hasattr(self, "down_arca"):
            return self.down_arca[1].zero_conv
        return self.__dict__["zero_conv_down_1"]

    @property
    def zero_conv_mid(self):  # type: ignore[override]
        if hasattr(self, "mid_arca"):
            return self.mid_arca.zero_conv
        return self.__dict__["zero_conv_mid"]

    def __getattr__(self, name: str):
        """防御性兜底: pipeline 可能在不同版本中访问 self.X 上不存在的字段.
        对于未注册的 config 属性, 返回安全的默认值而非抛 AttributeError.

        重要: 必须先调用 super().__getattr__ 让 nn.Module / ModelMixin / ConfigMixin
        的标准查找逻辑生效 (parameters / buffers / modules / _internal_dict).
        """
        # 1) 先走标准链路 (nn.Module → ModelMixin → ConfigMixin 的 __getattr__)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        # 2) 再尝试从 _internal_dict 取 (config field 的快速访问)
        try_dict = self.__dict__.get("_internal_dict")
        if try_dict is not None and name in try_dict:
            return try_dict[name]

        # 3) 真正的"未知字段", 抛 AttributeError
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'."
        )

    # ========================================================================
    # 新增模块初始化
    # ========================================================================

    def _init_new_modules(self) -> None:
        """新增模块初始化:
            - Encoder 内部: Kaiming normal + Norm = 1/0
            - TimedC2F 内部: Kaiming normal + β/γ = 0 + Time MLP 小权重
            - ARCA 内部: DWConv/PWConv Kaiming + LN=1/0 + alpha=0 (训练初残差=0)
            - 全部 ZeroConv: 已在 _zero_conv 中初始化为 0
        """
        # Encoder
        for m in self.weather_encoder.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm,)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

        # TimedC2F
        for c2f in self.timed_c2f_blocks:
            for m in c2f.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.ones_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.zeros_(c2f.c2f.beta)
            nn.init.zeros_(c2f.c2f.gamma)
            nn.init.normal_(c2f.time_mlp[-1].weight, std=0.02)
            nn.init.zeros_(c2f.time_mlp[-1].bias)

        # ARCA: 7 个 stage, 各自 Kaiming + LN=1/0 + alpha=0 + zero_conv=0
        all_arca = list(self.main_arca) + list(self.down_arca) + [self.mid_arca]
        for arca in all_arca:
            for m in arca.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.ones_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
            # 关键: alpha 初始 0.1 (论文方案第 6 节) — 避免与 zero_conv 0-init 死锁,
            # 同时 0.1 量级残差不会破坏 SD 预训练; 训练中通过 tanh 限到 [-1, 1]
            nn.init.constant_(arca.alpha, 0.1)
            # 防御性: zero_conv 已 0 init, 这里冗余保险
            nn.init.zeros_(arca.zero_conv.weight)
            if arca.zero_conv.bias is not None:
                nn.init.zeros_(arca.zero_conv.bias)

        # 防御性: 再次确认全部 ZeroConv 为 0 (property 兼容旧命名)
        for name in ['zero_conv_stage_0', 'zero_conv_stage_1', 'zero_conv_stage_2',
                     'zero_conv_stage_3', 'zero_conv_down_0', 'zero_conv_down_1',
                     'zero_conv_mid']:
            zc = getattr(self, name)
            nn.init.zeros_(zc.weight)
            if zc.bias is not None:
                nn.init.zeros_(zc.bias)

    # ========================================================================
    # 前向
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
        """
        与原生 ControlNet 完全相同的签名, 内部走退化感知 C2F 链路.

        返回: (down_block_res_samples, mid_block_res_sample)
            down_block_res_samples: tuple of 12 tensors, 严格匹配 SD2 UNet 期望
            mid_block_res_sample:   single tensor
        """
        # 处理 timestep 为 [B] 标量
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep] * sample.shape[0], device=sample.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(sample.device).expand(sample.shape[0])

        # ---- 1. 退化感知编码 ----
        feats = self.weather_encoder(controlnet_cond)
        # feats[0] F64: (B, 320, H/8, W/8)
        # feats[1] F32: (B, 640, H/16, W/16)
        # feats[2] F16: (B, 1280, H/32, W/32)
        # feats[3] F8 : (B, 1280, H/64, W/64)

        # ---- 2. 四级 TimedC2F + ARCA, 产出 4 个主残差 (已是 zero_conv + alpha 校准后) ----
        # 第一步: 4 个 TimedC2F 块处理 feats (雨/雪/雾特征精炼, 含粗细分支 + 时序调制)
        c2f_out = [self.timed_c2f_blocks[i](feats[i], timestep) for i in range(4)]
        # 第二步: 4 个主 stage 各自吃 c2f 输出, 各自有独立 zero_conv + alpha
        r_F64 = self.main_arca[0](c2f_out[0])   # 320ch, 64x64
        r_F32 = self.main_arca[1](c2f_out[1])   # 640ch, 32x32
        r_F16 = self.main_arca[2](c2f_out[2])   # 1280ch, 16x16
        r_F8  = self.main_arca[3](c2f_out[3])   # 1280ch, 8x8
        # 2 downsample 位置 (channel projection) - 各自 ARCA, 直接吃 c2f 输出
        r_down_0 = self.down_arca[0](c2f_out[1])  # 320ch, 32x32 (from F32)
        r_down_1 = self.down_arca[1](c2f_out[2])  # 640ch, 16x16 (from F16)
        # mid - 独立 ARCA, 直接吃 c2f 输出
        r_mid = self.mid_arca(c2f_out[3])         # 1280ch, 8x8

        # ---- 4. 拼装 12 down_res + 1 mid_res, 顺序严格匹配 SD2 UNet ----
        # [pos 0..2]: F64 × 3
        # [pos 3]:    down_0
        # [pos 4..5]: F32 × 2
        # [pos 6]:    down_1
        # [pos 7..8]: F16 × 2
        # [pos 9..11]: F8 × 3
        down_res_samples = (
            r_F64, r_F64, r_F64,        # pos 0, 1, 2
            r_down_0,                   # pos 3
            r_F32, r_F32,               # pos 4, 5
            r_down_1,                   # pos 6
            r_F16, r_F16,               # pos 7, 8
            r_F8, r_F8, r_F8,           # pos 9, 10, 11
        )
        mid_block_res_sample = r_mid

        # ---- 5. scaling (与 ControlNet 一致) ----
        if guess_mode:
            scales = torch.logspace(-1, 0, len(down_res_samples) + 1, device=sample.device)
            scales = scales * conditioning_scale
            down_res_samples = tuple(r * s for r, s in zip(down_res_samples, scales))
            mid_block_res_sample = mid_block_res_sample * scales[-1]
        else:
            down_res_samples = tuple(r * conditioning_scale for r in down_res_samples)
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if not return_dict:
            return (down_res_samples, mid_block_res_sample)

        from diffusers.models.controlnets.controlnet import ControlNetOutput
        return ControlNetOutput(
            down_block_res_samples=down_res_samples,
            mid_block_res_sample=mid_block_res_sample,
        )

    # ========================================================================
    # diffusers 兼容接口
    # ========================================================================

    def save_pretrained(self, save_directory: str, **kwargs):
        """简化版 save_pretrained."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'diffusion_pytorch_model.bin'))
        # 用 diffusers 的 save_config 写 config.json (兼容 ModelMixin/ConfigMixin)
        if hasattr(self, 'save_config'):
            try:
                self.save_config(save_directory)
            except Exception as e:
                # 兜底: 直接 dump _internal_dict
                import json
                with open(os.path.join(save_directory, 'config.json'), 'w') as f:
                    json.dump(dict(self._internal_dict), f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: str | None = None, **kwargs):
        """简化版 from_pretrained."""
        import os
        load_dir = pretrained_model_name_or_path
        if subfolder:
            load_dir = os.path.join(load_dir, subfolder)
        state_dict_path = os.path.join(load_dir, 'diffusion_pytorch_model.bin')
        state_dict = torch.load(state_dict_path, map_location='cpu')
        model = cls()
        model._init_new_modules()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'[from_pretrained] missing keys: {len(missing)} (first 5: {missing[:5]})')
        if unexpected:
            print(f'[from_pretrained] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})')
        return model