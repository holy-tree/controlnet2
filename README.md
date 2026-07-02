# ControlNet-SR

基于 ControlNet 和 RAM (Recognize Anything Model) 的图像超分辨率项目。

## 项目结构

```
.
├── train_controlnet.py          # ControlNet 训练脚本
├── wavelet_color_fix.py         # 颜色修复工具 (Wavelet/AdaIn)
├── test.py                      # 测试脚本
├── ramseesr/                    # RAM + ESRGAN 模型
│   ├── models/                  # 模型定义
│   │   ├── ram.py              # RAM 图像标注模型
│   │   ├── swin_transformer.py # Swin Transformer
│   │   └── bert.py             # BERT 模型
│   ├── inference.py            # 推理函数
│   └── transform.py            # 数据转换
├── dataloaders/                 # 数据加载
│   ├── paired_dataset.py       # 成对数据集
│   └── realesrgan.py           # RealESRGAN 退化流程
└── scripts/train/              # 训练脚本
```

## 主要功能

- **ControlNet 超分辨率**: 使用 ControlNet 结合低分辨率图像作为条件引导进行超分辨率重建
- **RAM 图像标注**: 集成 Recognize Anything Model 进行图像标签自动识别
- **RealESRGAN 退化**: 模拟真实世界退化过程生成训练数据
- **颜色修复**: 支持 Wavelet 和 AdaIn 两种颜色修复方法

## 训练数据格式

数据集文件夹结构：
```
dataset_root/
├── sr_bicubic/   # 低分辨率图像
├── gt/           # 高分辨率原图
└── gt_tag/       # 图像标签文本
```

## 训练

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7," accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="stable-diffusion-2-1-base" \
  --output_dir="./experiment/ControlNetSR" \
  --root_folders '/path/to/dataset' \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision="fp16" \
  --resolution=512 \
  --learning_rate=5e-5 \
  --train_batch_size=4 \
  --null_text_ratio=0.5
```

## 颜色修复

```python
from wavelet_color_fix import wavelet_color_fix, adain_color_fix

# Wavelet 颜色修复
result = wavelet_color_fix(target_image, source_image)

# AdaIn 颜色修复
result = adain_color_fix(target_image, source_image)
```

## 依赖

- torch
- diffusers
- transformers
- accelerate
- realesrgan / basicsr
- PIL / opencv
