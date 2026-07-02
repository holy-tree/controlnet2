"""
成对图像数据集加载器
================================

支持目录结构:
    dataset_root/
    ├── rain/{train,test}/{GT,LQ}/
    ├── snow/{train,test}/{GT,LQ}/
    └── haze/{train,test}/{GT,LQ}/

特性:
    - 自动按文件名匹配 GT / LQ 图像对
    - 支持动态 prompt (按天气类型 rain/snow/haze 区分)
    - prompt 开关 (use_prompt=False 时全部使用空 prompt)
    - prompt 比例 (prompt_ratio: 0.15 ~ 0.25)
    - 兼容原始 null_text_ratio (CFG 训练)
"""

import glob
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils import data as data
from torchvision import transforms

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 默认各天气的 prompt 描述 (当 use_prompt=True 时使用)
DEFAULT_WEATHER_PROMPTS: Dict[str, str] = {
    "rain": "rainy scene, rain streaks on the image, wet surfaces, overcast sky",
    "snow": "snowy scene, snowflakes covering the image, cold atmosphere, white noise",
    "haze": "hazy scene, foggy atmosphere, low visibility, grayish tone",
}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


class PairedCaptionDataset(data.Dataset):
    """
    从 dataset_root/{weather}/{split}/{GT,LQ}/ 加载图像对。

    Args:
        dataset_root: 数据集根目录 (例如 ./datasets)
        weather_types: 要加载的天气类型列表, 例如 ['rain', 'snow', 'haze']
        splits: 要加载的划分列表, 例如 ['train']
        tokenizer: HuggingFace tokenizer,用于将文本转为 token ids
        null_text_ratio: 空 prompt 比例 (CFG 训练), 默认 0.5
        use_prompt: 是否使用 prompt 文本 (False: 全部空 prompt)
        prompt_ratio: 启用 prompt 时,使用天气 prompt 的比例 (0.15 ~ 0.25)
        weather_prompts: 自定义各天气的 prompt 描述 (可选)
        resolution: 训练分辨率 (默认 512), 会 Resize 短边到此值后 CenterCrop
        weather_num_samples: 按天气类型限制样本数,例如 {'rain': 10, 'snow': 5, 'haze': 3}
                            None 或不指定表示不限制。<=0 同样视为不限制。
    """

    def __init__(
        self,
        dataset_root: str = "",
        weather_types: List[str] = None,
        splits: List[str] = None,
        tokenizer=None,
        null_text_ratio: float = 0.5,
        use_prompt: bool = False,
        prompt_ratio: float = 0.2,
        weather_prompts: Dict[str, str] = None,
        resolution: int = 512,
        weather_num_samples: Dict[str, int] = None,
    ):
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.null_text_ratio = null_text_ratio
        self.use_prompt = use_prompt
        self.prompt_ratio = max(0.0, min(1.0, prompt_ratio))
        self.resolution = resolution
        self.weather_num_samples = weather_num_samples or {}

        if weather_types is None:
            weather_types = ["rain", "snow", "haze"]
        if splits is None:
            splits = ["train"]

        self.weather_types = list(weather_types)
        self.splits = list(splits)

        # 合并自定义与默认 prompt
        self.weather_prompts = dict(DEFAULT_WEATHER_PROMPTS)
        if weather_prompts:
            self.weather_prompts.update(weather_prompts)

        # 加载所有图像对: list of (gt_path, lq_path, weather)
        self.samples: List[Tuple[Path, Path, str]] = []
        for weather in self.weather_types:
            for split in self.splits:
                gt_dir = self.dataset_root / weather / split / "GT"
                lq_dir = self.dataset_root / weather / split / "LQ"
                if not gt_dir.is_dir() or not lq_dir.is_dir():
                    print(f"[跳过] {gt_dir} 或 {lq_dir} 不存在")
                    continue

                gt_map = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and is_image(p)}
                lq_map = {p.stem: p for p in lq_dir.iterdir() if p.is_file() and is_image(p)}

                matched = 0
                for stem in sorted(gt_map.keys() & lq_map.keys()):
                    self.samples.append((gt_map[stem], lq_map[stem], weather))
                    matched += 1

                print(f"[数据集] {weather}/{split}: 匹配 {matched} 对")

        if not self.samples:
            raise FileNotFoundError(
                f"在 {self.dataset_root} 下未找到任何匹配的图像对, "
                f"请检查目录结构是否为 {{weather}}/{{split}}/{{GT,LQ}}/"
            )

        # ===== 按 weather 限制每个天气的样本数 =====
        # 保持 weather 顺序稳定: 遍历 weather_types, 截取对应样本
        if self.weather_num_samples:
            grouped: Dict[str, List[Tuple[Path, Path, str]]] = {w: [] for w in self.weather_types}
            for sample in self.samples:
                if sample[2] in grouped:
                    grouped[sample[2]].append(sample)

            new_samples: List[Tuple[Path, Path, str]] = []
            for weather in self.weather_types:
                limit = self.weather_num_samples.get(weather, -1)
                weather_samples = grouped[weather]
                if limit is not None and limit > 0 and limit < len(weather_samples):
                    print(f"[数据集] {weather}: 截断 {len(weather_samples)} -> {limit} 样本")
                    new_samples.extend(weather_samples[:limit])
                else:
                    new_samples.extend(weather_samples)
            self.samples = new_samples

            print(f"[数据集] 最终训练样本数: {len(self.samples)}")
            for w in self.weather_types:
                cnt = sum(1 for s in self.samples if s[2] == w)
                limit_str = f"/{self.weather_num_samples[w]}" if self.weather_num_samples.get(w, -1) > 0 else ""
                print(f"  - {w}: {cnt}{limit_str}")

        # 图像预处理: 短边 Resize 到 resolution, 再中心裁剪到 resolution×resolution
        # 这样无论原始尺寸多大,输出都是固定大小,可以 batch
        # GT 和 LQ 用相同变换,保持像素级对应关系
        self.preprocess = transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution),
        ])
        self.to_tensor = transforms.ToTensor()

    def _make_prompt(self, weather: str) -> str:
        """
        根据 use_prompt / prompt_ratio / null_text_ratio 决定 prompt:
            - use_prompt=False  -> 全部空
            - use_prompt=True   -> prompt_ratio 概率用天气 prompt, 其余为空
            - 额外叠加 null_text_ratio (来自原始 CFG 训练)
        """
        # 整体逻辑:先按 use_prompt 决定是否启用, 再叠加 null_text_ratio
        # 实际最终空 prompt 概率 = 1 - use_prompt + use_prompt * (1 - prompt_ratio)
        if not self.use_prompt:
            return ""

        # use_prompt=True 时:
        #   - prompt_ratio 概率使用天气 prompt
        #   - 1 - prompt_ratio 概率为空 (用于 CFG 训练)
        if random.random() < self.prompt_ratio:
            return self.weather_prompts.get(weather, "")
        return ""

    def tokenize_caption(self, caption: str = "") -> torch.Tensor:
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        result = inputs.input_ids
        # 兼容返回 list 的 tokenizer (测试场景)
        if not torch.is_tensor(result):
            result = torch.tensor(result)
        return result

    def __getitem__(self, index):
        gt_path, lq_path, weather = self.samples[index]

        gt_img = Image.open(gt_path).convert("RGB")
        gt_img = self.preprocess(gt_img)
        gt_img = self.to_tensor(gt_img)

        lq_img = Image.open(lq_path).convert("RGB")
        lq_img = self.preprocess(lq_img)
        lq_img = self.to_tensor(lq_img)

        prompt = self._make_prompt(weather)

        example = {
            "conditioning_pixel_values": lq_img,           # LQ, [0, 1]
            "pixel_values": gt_img * 2.0 - 1.0,           # GT, [-1, 1]
            "input_ids": self.tokenize_caption(prompt).squeeze(0),
            "weather": weather,
        }
        return example

    def __len__(self):
        return len(self.samples)