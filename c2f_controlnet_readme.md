# 多天气图像修复模型完整架构整理

> 文档版本：2026-07-20（整合 Phase 1~5 全部改进）

---

## 一、整体方案概览

### 1.1 任务定义

针对多天气退化图像（雨/雪/雾）做基于扩散模型的图像修复。输入为 LQ 低质量 RGB 图（512×512），输出为 SD2 UNet 主干预测的噪声张量（4×64×64）。ControlNet 旁路从 LQ 中提取多尺度退化条件，生成可叠加到 UNet 各层残差块的"修图建议"，引导扩散模型还原清晰图像。

### 1.2 整体数据流链路

```
LQ (3, 512, 512)
  → WeatherDegradationEncoder：5 档多尺度特征 + F128 高频注入
  → TimedC2FBlock ×4：粗细双分支 + 时序调制
  → ARCAResidualCalibrator ×7：gate × tanh(α) 双层限幅
  → 拼装层（无参）：7 张修图建议 → 13 个 UNet 注入点
  → 冻结 SD2 UNet + LoRA 微调：每层残差块后叠加对应修图建议
  → 噪声预测输出 (B, 4, 64, 64)
```

### 1.3 极简数据流图

```
LQ (3, 512, 512)
  ├─→ stem_head → feat_512 (64ch, 512×512)
  │     └─→ down_blocks[0:2] → feat_256 (64ch, 256×256)
  │           └─→ down_blocks[2:4] → feat_128 (64ch, 128×128)
  │                 ├─→ proj_128 → f128 (320ch, 128×128)
  │                 │       └─→ f128_refine (双线性降采样 0.5 + 3×3 conv, init=0) → 64×64 ──┐
  │                 └─→ down_blocks[4:6] → feat_64 → proj_64 → f64_base ───────────────────┘
  │                                                                └─→ 加法注入 → f64 (320ch, 64×64)
  ├─→ avg_pool → proj_32 → f32 (640ch, 32×32)
  ├─→ avg_pool → proj_16 → f16 (1280ch, 16×16)
  └─→ avg_pool → proj_8  → f8  (1280ch, 8×8)
  → 4 路并行 TimedC2FBlock → [F64', F32', F16', F8']
  → 7 路独立 ARCAResidualCalibrator (gate × tanh(α) 缩放) → 7 组修图建议
  → 无参拼装层 → 13 个 UNet 层级条件张量
  → 冻结 SD2 UNet + LoRA（每层叠加修图建议）→ 噪声预测输出
```

### 1.4 方案核心思路

整套结构等价于专用 ControlNet 分支：输入退化天气图 LQ → 提取多尺度退化条件特征（含 F128 高频注入）→ 经时序与多感受野增强 → 生成适配 SD2 UNet 各层级的校正残差（修图建议）→ 送入 UNet 每层残差块叠加修正 → 引导扩散模型还原清晰图像。Phase 5 额外通过 LoRA 微调 UNet attention 投影，在不动 backbone 主干权重的前提下释放生成能力。

---

## 二、训练配置

### 2.1 可训练参数量汇总

| 模块 | 参数量 | 关键说明 |
|------|--------|----------|
| WeatherDegradationEncoder | ~5M | 含新增 proj_128（20.8K）+ f128_refine |
| TimedC2FBlock ×4 | 25M | 每个块 ~6.25M |
| ARCAResidualCalibrator ×7 | ~3.5M | 含 7 个 α 标量 + 7 个 gate 标量 |
| ControlNet 旁路小计 | ~33.9M + 14 标量 | |
| UNet LoRA（to_q / to_k / to_v） | ~20M | rank=32, alpha=64 |
| **可训练总量** | **~54M** | |
| SD2 UNet 主体 | 860M | 完全冻结 |
| CLIP text encoder + VAE | ~206M | 完全冻结 |

### 2.2 学习率分层

| 参数类型 | 参数量 | 学习率 | 说明 |
|----------|--------|--------|------|
| 7 个可学习标量 α（残差幅度控制） | 7 | 3e-6 | 极小学习率，稳定残差输出幅度 |
| 7 个可学习标量 gate（直接乘子） | 7 | 5e-4 | 控制残差有效缩放，R > 1 时需降到 0.3~0.5 |
| 编码器 + 全部 C2F + ARCA 主干（不含 α / gate） | 33M | 5e-5 | 主体特征提取模块 |
| ARCA 内部零卷积权重 | 7 组 | 主学习率 | 初始化权重全 0，渐进生效校正 |
| **UNet LoRA params (A, B)** | **~20M** | **1e-4** | Phase 5：低秩适配 UNet attention |
| SD2 UNet 全部权重 | 860M | 不更新 | 完全冻结 |

### 2.3 损失函数

```
total_loss = L_noise_mse
           + latent_l1_weight × L_latent_L1      # 0.2
           + freq_loss_weight × L_FFT_freq       # 0.05
           + lpips_weight × L_LPIPS              # 0.02 (间歇；实际因 OOM 被禁用)
```

| 损失项 | 作用 | 权重 | 计算频率 |
|--------|------|------|----------|
| `F.mse_loss(model_pred, noise)` | 扩散噪声预测（轨迹约束） | 1.0 | 每步 |
| `F.l1_loss(pred_x0, latents)` | 隐空间像素级对齐 | 0.2 | 每步 |
| `F.l1_loss(FFT(pred_x0).abs(), FFT(latents).abs())` | 隐空间频域对齐，保留高频细节 | 0.05 | 每步 |
| `LPIPS(pred_rgb, gt_rgb)` | 感知对齐（高频纹理） | 0.02 | 每 4 步（实际为 0：CUDA OOM） |

- **Phase 1 改进效果**：PSNR 14 dB → 18.1 dB（+4.1 dB），证明根本问题是缺像素级内容约束。
- **Phase 2+4+5 预期**：18.1 dB → 21+ dB（频域约束 + F128 高频注入 + LoRA backbone 适配）。

### 2.4 调度器与混合精度

- **调度器**：`cosine`，warmup 500 步，`lr_num_cycles=0.5`，从 5e-5 衰减到接近 0
- **LoRA 独立学习率**：1e-4，高于主网络 5e-5，加速 attention 适配
- **混合精度**：bf16，替代 fp16 规避 LoRA 触发的 `GradScaler: Attempting to unscale FP16 gradients` 错误，需 Ampere+ GPU（RTX 3090 / A100 / H100）

---

## 三、分模块架构详解

### 模块 1：WeatherDegradationEncoder 退化多尺度编码器

**输入**：LQ 低质量图像 `(B, 3, 512, 512)`

**核心作用**：分层下采样，提取 5 档不同分辨率特征，并通过 F128 高频注入解决下采样链路中雨丝、雪粒等高频结构丢失的问题。

#### 设计动机

LQ 经过三次步长=2 下采样后，128×128 中间层特征中保留的高频纹理在继续下采样到 64×64 时会被均值滤波磨平，导致下游 UNet 残差块接收不到细节修正信号。F128 高频注入的核心思路是把 128×128 阶段的高频信息通过可学习降采样重新注入到 F64。

#### 内部计算流程（Phase 4 改造后）

1. **浅层纹理提取**：2 层 3×3 卷积 + GELU，输出 64 通道 512×512 浅层特征
2. **三次步长=2 下采样**（保持原结构以向后兼容老 checkpoint）：
   - `feat_256 = down_blocks[0:2](feat_512)` → 64 通道 256×256
   - `feat_128 = down_blocks[2:4](feat_256)` → 64 通道 128×128
   - `feat_64 = down_blocks[4:6](feat_128)` → 64 通道 64×64
3. **5 档特征投影**：
   - `f128 = proj_128(feat_128)` → 320 通道 128×128（零初始化，老 checkpoint 加载后等价无 F128）
   - `f64_base = proj_64(feat_64)` → 320 通道 64×64
   - **F128 高频注入**：`f128_up = f128_refine(f128)` → `Upsample(scale_factor=0.5, bilinear) → Conv2d(3×3, init=0)`。代码层名虽叫 `Upsample`，但 `scale_factor=0.5` 实际是 128 → 64 的可学习降采样
   - `f64 = f64_base + f128_up`：注入后 F64 含高频细节
   - `f32 = proj_32(avg_pool(f64, 2))` → 640 通道 32×32
   - `f16 = proj_16(avg_pool(f32, 2))` → 1280 通道 16×16
   - `f8 = proj_8(avg_pool(f16, 2))` → 1280 通道 8×8

**输出**：四尺度基础特征列表 `[F64, F32, F16, F8]`，F64 已含 F128 高频注入。

---

### 模块 2：TimedC2FBlock ×4 时序感知粗细精炼块

**输入**：模块 1 输出的 4 个尺度特征，每个尺度单独送入 1 个独立 C2F 块；额外输入扩散时序步长 timestep。

**核心作用**：引入时序调制与多膨胀感受野，细化粗/细两路特征，强化退化细节表达。

#### 设计动机

传统 ControlNet 旁路缺乏对扩散 timestep 的显式建模，导致不同时序步下的特征表达无差异；单路卷积难以同时捕捉局部纹理与大范围退化分布。C2F 块借鉴 ReviveDiff 的双分支设计：细纹理支路捕获局部雨/雪纹理，粗全局支路用膨胀卷积建模大范围雾气分布。

#### 单块内部流程

1. **时序调制分支**：timestep 经正弦编码 + 两层 Linear + Sigmoid，输出时序权重 γ ∈ (0, 1)
2. **静态 C2F 特征提取**（细/粗双路并行）：
   - **细纹理分支**：归一化 → 1×1 + 深度 3×3 卷积 → SimpleGate 通道相乘 → SCA 通道统计，输出精细局部特征 `x_fine`
   - **粗全局分支**：三层膨胀卷积 `dilation = 2/4/8`，累积感受野覆盖 31×31，输出大范围上下文特征 `x3_coarse`
3. **粗细特征融合**：注意力门控融合 → 深度 3×3 卷积 + β 残差 → FFN 通道变换
4. **时序调制输出**：对 C2F 整体精炼输出乘以系数 `(1 + 0.2γ)` 完成温和的全局缩放（γ ∈ (0, 1)，0.2 系数避免破坏已学特征）
   - 实际实现（`c2f_block.py:268-294`）仅做 `refined * (1 + 0.2γ)` 全局放缩，未对 fine / coarse 分支做差异化门控
   - `TimedC2FBlock` 类 docstring 中描述的 `fine * (0.5 + 0.5γ)` / `coarse * (1 - 0.5γ)` 软调制未在当前 forward 中生效（详见附录 A.3）

**输出**：4 张精炼后多尺度特征 `[F64', F32', F16', F8']`，尺度与通道数与输入完全对应。

---

### 模块 3：ARCAResidualCalibrator ×7 残差校正器

**输入**：模块 2 精炼后的四尺度特征，按表分配给 7 个独立 ARCA 块，每个块独享一套参数。

**核心作用**：将多尺度特征转换为可控幅度的残差校正张量（修图建议），初始输出趋近于 0；通过 `tanh(α)` 与 `gate` 双层限幅约束残差相对 UNet 主干的比例 R。

#### 设计动机

直接送 C2F 精炼特征到 UNet 会出三个问题：① 通道维度不完全匹配（C2F 输出 320/640/1280 ch 与 UNet downsample 过渡位置期望的 320/640 ch 投影不一致）；② 分布不匹配（C2F 是"退化域"分布，UNet 期望"生成域"分布）；③ 致命问题——训练初期 ARCA 随机初始化会输出任意幅值，破坏 SD2 UNet 的预训练权重，引发灾难性遗忘。

**ARCA 三件套的解决思路：**

| 组件 | 解决的问题 |
|------|------------|
| `GroupNorm + DWConv + PWConv` | 通道对齐 + 局部纹理提取 + 跨通道融合，把 C2F 特征"翻译"到 UNet 期望的分布 |
| **零卷积**（zero-init 1×1 conv） | 训练初期输出严格为 0，前向等价"无 ControlNet"，保留 SD2 全部预训练能力 |
| **`tanh(α) × gate` 双层限幅** | 训练中后期残差逐渐增长时，`tanh` 把单次校正在 (-1, 1) 区间，`gate` 进一步控制残差相对 UNet 主干的比例 R |

**为什么是 7 个 ARCA 而不是 4 个或 13 个？**

- **4 个 main_arca**：与 C2F 输出一一对应（粗/中/细/极细 4 个尺度）
- **2 个 down_arca**：UNet 下采样过渡位置需要通道投影（F32 → 320ch、F16 → 640ch），与 main 输出的通道不同，必须独立 ARCA
- **1 个 mid_arca**：中间层单独处理（虽与 main_arca.3 同源，但 mid_block 语义独立）
- **不做 13 个独立 ARCA**：同一 stage 内多个 res block 共享退化特征是合理的归纳偏置，参数量翻倍但表达提升有限；由模块 4 拼装层用无参复制完成 7 → 13 的扩展

#### 7 个 ARCA 块分配映射表

| ARCA 块名称 | 输入特征 | 输出通道 | 输出尺寸 | 供给 UNet 层级位置 |
|-------------|----------|----------|----------|--------------------|
| main_arca.0 | F64' | 320 | 64×64 | UNet down 0/1/2 |
| main_arca.1 | F32' | 640 | 32×32 | UNet down 4/5 |
| main_arca.2 | F16' | 1280 | 16×16 | UNet down 7/8 |
| main_arca.3 | F8'  | 1280 | 8×8 | UNet down 9/10/11 |
| down_arca.0 | F32' | 320 | 32×32 | UNet down 3 |
| down_arca.1 | F16' | 640 | 16×16 | UNet down 6 |
| mid_arca | F8' | 1280 | 8×8 | UNet 中间层 mid_block |

#### 单个 ARCA 内部流程

1. `GroupNorm(32)` 归一化 → 深度 3×3 卷积提取局部纹理 → GELU 激活
2. 1×1 卷积通道映射 → 再次 `GroupNorm` 归一化
3. 零初始化 1×1 卷积：训练初期残差输出严格为 0，不破坏已学 UNet
4. **幅度控制**（双层）：
   - 乘 `tanh(α)`：α 为块专属可学习标量（init = 0.1），tanh 将单次校正在 (-1, 1) 区间
   - 乘 `gate`（直接乘子）：块专属可学习标量（init = 1.0），初始等价无 gate、不破坏已学模型
5. **输出公式**：`residual = gate × tanh(α) × zero_conv_output`

#### Gate 参数演化历程

| 尝试 | 结果 | 教训 |
|------|------|------|
| sigmoid(gate)，init = 4.0 | gradient 消失（sigmoid'(4) = 0.018） | 高 init 值会 saturation |
| sigmoid(gate)，init = 1.0 | 初始行为破坏模型（residual 缩小 27%） | 低 init 会干扰已学模型 |
| **直接乘子 gate，init = 1.0（当前）** | 初始等价于无 gate，无冲击 | 但受 LR 限制，实际学不动 |

**实际结果**：受 LR cosine 衰减 + gate 梯度本身很小影响，gate 几乎不动（变化 < 1e-3），始终卡在 ~1.0。结论：gate 思路需要单独的高 LR 配置（不参与 cosine 衰减），或改用硬约束（alpha clip）。

**输出**：7 组独立残差校正张量（7 条修图建议）。

---

### 模块 4：拼装层（无参接口适配器）

**输入**：模块 3 输出的 7 组校正残差。

**核心作用**：把 7 张语义独立的修图建议整理为 SD2 UNet 期望的 13 个条件张量（12 down + 1 mid）。

#### 设计动机

SD2 UNet 的 `down_block_additional_residuals` 接口要求**精确 13 个**张量：

| UNet 位置    | 通道  | 分辨率 | 位置数 |
|--------------|-------|--------|--------|
| down_block_0 | 320   | 64×64  | 3（res_0、res_1、downsample） |
| down_block_1 | 640   | 32×32  | 3 |
| down_block_2 | 1280  | 16×16  | 3 |
| down_block_3 | 1280  | 8×8    | 3 |
| mid_block    | 1280  | 8×8    | 1 |

但 ARCA 只产出 7 个张量。拼装层是"把 7 个语义独立的修图建议无参复制 / 分配到 13 个 SD2 UNet 期望的位置"的接口适配器，而非特征变换层，因此无可训练参数。

#### 拼装映射规则

1. **12 个 down 层条件**：
   - pos 0/1/2：复用 `main_arca.0` 输出（3 次）
   - pos 3：`down_arca.0`
   - pos 4/5：复用 `main_arca.1` 输出（2 次）
   - pos 6：`down_arca.1`
   - pos 7/8：复用 `main_arca.2` 输出（2 次）
   - pos 9/10/11：复用 `main_arca.3` 输出（3 次）
2. **中间层条件**：`mid_arca` 输出
3. **统一缩放**：全部 13 个张量乘以超参 `conditioning_scale`（默认 1.0）

**输出**：长度为 13 的条件张量列表，完全匹配 SD2 UNet 各残差块输入格式。

---

### 模块 5：SD2 UNet（冻结主干 + LoRA 微调）

**输入**：扩散噪声潜变量、文本提示词编码、模块 4 输出的 13 个校正残差张量。

**核心协作逻辑**：标准 SD2 UNet 前向推理，每一层残差块计算完成后叠加对应位置的修图建议，用"修图建议"修正每层特征。

#### 冻结 / 微调一览

| 状态 | 组件 | 参数量 |
|------|------|--------|
| 🔒 完全冻结 | SD2 UNet 所有 conv / FFN / LayerNorm / time embedding 权重 | 860M |
| 🔒 完全冻结 | CLIP text encoder + VAE encoder / decoder | ~206M |
| ✏️ **新增并训练** | UNet attention 的 `to_q` / `to_k` / `to_v` 上的 LoRA **A 矩阵**（Kaiming init） | ~10M |
| ✏️ **新增并训练** | UNet attention 的 `to_q` / `to_k` / `to_v` 上的 LoRA **B 矩阵**（zero init） | ~10M |
| ✏️ **新增并训练** | ControlNet 旁路（Encoder + C2F + ARCA + α / gate 标量） | ~33.9M + 14 标量 |

#### LoRA 实现要点

- **作用位置**：在 SD2 UNet **每一层 self-attention 与 cross-attention 块的 Q / K / V 三个线性投影**上各叠加一个低秩适配矩阵 ΔW
- **前向计算**：`Y = (W₀ + scaling · B · A) · X`，其中 `scaling = α / rank = 64 / 32 = 2`
  - W₀ 为冻结的 SD2 原始权重
  - A ∈ ℝ^{dim × 32}（Kaiming init）、B ∈ ℝ^{32 × dim}（zero init）
  - 训练第一步 `ΔW = B · A = 0`，前向等价于无 LoRA，不破坏 SD2 预训练
- **不修改的 UNet 部分**：所有 ResBlock、FFN、LayerNorm、time embedding、`to_out` 输出投影、downsample / upsample 卷积——全部冻结
- **rank=32, alpha=64**：缩放因子 2，让 LoRA delta 在前向数值上与原始权重同量级
- **强制 fp32 精度**：避免 autocast 把 LoRA grad 转 fp16 触发 `GradScaler: Attempting to unscale FP16 gradients` 错误
- **训练量 ~20M 参数**（仅 A + B 总和），**独立 LR = 1e-4**（高于主网络 5e-5，加速 attention 适配）
- **手动实现不依赖 peft**：避免 diffusers / peft 版本兼容问题

**输出**：扩散噪声预测结果 `(B, 4, 64, 64)`。

---

## 四、Phase 演进与实验效果

### 4.1 Phase 总览

| Phase | 改动 | PSNR 变化 | 状态 |
|-------|------|-----------|------|
| 基线 | noise MSE + 冻结 UNet | 14.0 dB | ✗ 严重退化 |
| Phase 1 | + latent L1 + LPIPS 损失 | **+4.1 dB → 18.1** | ✓ 已生效 |
| Phase 2 | + cosine LR scheduler | +0~1 dB（稳定化）| ✓ 已生效 |
| Phase 3（失败） | ARCA gate 三种方案 | +0 dB（gate 学不动）| ✗ 暂时搁置 |
| Phase 4 | 多尺度 encoder（F128 高频注入 + 可学习下采样） | 预期 +0.5~1.5 dB | ⏳ 待验证 |
| Phase 5（当前） | UNet LoRA（rank=32）+ bf16 + FFT 频域损失 | 预期 +2~4 dB | ⏳ 训练中 |

### 4.2 Phase 3 失败教训

ARCA gate 的根本死结：

- 直接乘子（无 sigmoid）→ 梯度太小 × cosine 衰减 LR = 变化量 < 1e-3 / 6500 step
- 在 70k 剩余 step 内，gate 最多变化 0.016，远不够达到目标 0.3~0.5

**结论**：gate 思路需要单独的高 LR 配置（不参与 cosine 衰减），或改用硬约束（alpha clip）。

### 4.3 Phase 4 关键改进：encoder 高频注入

放弃 gate 思路，转向直接修改 encoder 注入高频信息：

1. 新增 `proj_128`（20K 参数），零初始化从零学
2. 新增 `f128_refine`：双线性降采样 0.5 + Conv2d(3×3, init=0) 替代 bilinear，保留高频
3. 老权重零改动，向后兼容
4. F128 通过可学习下采样注入到 F64，SD2 UNet pos 0/1/2 间接获得高频增强

详细实现见模块 1。

### 4.4 Phase 5 关键改进：LoRA + FFT 损失

1. **UNet LoRA**：手动实现，不依赖 peft，rank=32 / alpha=64，覆盖 to_q / to_k / to_v
2. **bf16 mixed precision**：fp16 + LoRA 触发 GradScaler 错误，bf16 跳过 GradScaler
3. **FFT 频域损失**：在隐空间频域对齐，保留雨丝 / 雪粒 / 边缘等高频细节
4. **LoRA 独立学习率 1e-4**：高于主网络 5e-5，加速 attention 适配

详细实现见模块 5 与 §2.3 损失函数。

### 4.5 验证集 PSNR 演进（n=4 训练验证仅供参考，n=200 全面 eval 才可信）

| 阶段 | rain | snow | haze | 平均 |
|------|------|------|------|------|
| 基线（14k step） | ~12 | ~13 | ~11 | ~12 |
| Phase 1 后（140k step） | 18.34 | 18.69 | 17.30 | **18.11** |
| Phase 4+5 目标 | 21~22 | 21~22 | 20~21 | **21~22** |

---
