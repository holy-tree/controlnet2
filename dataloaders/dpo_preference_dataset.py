"""
ControlNet-DPO 偏好对 Dataset
================================

数据布局:
    dataset_root/{weather}/{split}/
    ├── GT/                      # 真实原图
    ├── LQ/                      # 退化条件图
    └── candidates/{stem}/
        ├── cand_000.png         # N 张候选恢复图
        ├── ...
        └── score.json           # 4 维 reward 全量分数 (build_preference.py 产出)

核心特性:
    1. 实时用 reward_weights 聚合分数, 每条样本不预选 winner/loser
    2. 在 Top-k% / Bottom-k% 池内随机抽 winner / loser, 增加偏好对多样性
    3. 弱偏好过滤 (score_gap < min_gap 自动跳过)
    4. winner / loser / LQ / GT 四图共享 random.Random(seed) 做几何增强
       (Flip / Scale / CenterCrop), 严禁颜色增强
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils import data as data
from torchvision import transforms
from torchvision.transforms import functional as TF

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTENSIONS


class DPOPreferenceDataset(data.Dataset):
    """
    Args:
        dataset_root:  数据集根目录
        weather_types: 参与训练的天氣类型列表
        splits:        参与训练的数据划分
        tokenizer:     HF tokenizer
        resolution:    训练分辨率
        candidates_subdir: build_preference.py 写出的子目录名
        reward_weights: {psnr, ssim, lpips, clip_iqa} 聚合权重
                      (注意: lpips 越小越好, 权重应为负)
        top_k_ratio:    winner 池 = 聚合分数前 k% 样本
        bottom_k_ratio: loser  池 = 聚合分数后 k% 样本
        min_gap:        弱偏好过滤阈值 (winner_score - loser_score < min_gap 视为无效)
        augment_geo:    是否启用几何增强
        geo_flip_prob:  水平翻转概率
        geo_scale_range: (low, high) 短边随机缩放比例
        null_text_ratio: 全部空 prompt 的比例 (DPO 阶段应设为 0)
        use_prompt / prompt_ratio / weather_prompts: 与 SFT 阶段对齐
        weather_num_samples: 限制每种天气的样本数
        max_skip:       __getitem__ 中跳过弱偏好样本的最大尝试次数
    """

    def __init__(
        self,
        dataset_root: str,
        weather_types: List[str],
        splits: List[str],
        tokenizer,
        resolution: int = 512,
        candidates_subdir: str = "candidates",
        reward_weights: Dict[str, float] = None,
        top_k_ratio: float = 0.30,
        bottom_k_ratio: float = 0.30,
        min_gap: float = 0.02,
        normalize: bool = True,
        augment_geo: bool = True,
        geo_flip_prob: float = 0.5,
        geo_scale_range: Tuple[float, float] = (0.9, 1.1),
        null_text_ratio: float = 0.0,
        use_prompt: bool = False,
        prompt_ratio: float = 0.2,
        weather_prompts: Dict[str, str] = None,
        weather_num_samples: Dict[str, int] = None,
        max_skip: int = 8,
    ):
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.candidates_subdir = candidates_subdir
        self.reward_weights = reward_weights or {
            "psnr": 0.25, "ssim": 0.25, "lpips": -0.30, "clip_iqa": 0.20,
        }
        self.normalize = normalize
        self.top_k_ratio = max(0.0, min(1.0, top_k_ratio))
        self.bottom_k_ratio = max(0.0, min(1.0, bottom_k_ratio))
        self.min_gap = min_gap
        self.augment_geo = augment_geo
        self.geo_flip_prob = geo_flip_prob
        self.geo_scale_range = geo_scale_range
        self.null_text_ratio = null_text_ratio
        self.use_prompt = use_prompt
        self.prompt_ratio = max(0.0, min(1.0, prompt_ratio))
        self.weather_num_samples = weather_num_samples or {}
        self.max_skip = max_skip

        # 严禁颜色增强
        for forbidden in ("color_jitter", "brightness", "contrast", "saturation", "hue"):
            assert not getattr(self, forbidden, False), (
                f"检测到颜色增强参数 {forbidden}, DPO 阶段严禁启用"
            )

        # 合并默认 prompt
        self.weather_prompts = dict(_DEFAULT_WEATHER_PROMPTS)
        if weather_prompts:
            self.weather_prompts.update(weather_prompts)

        if weather_types is None:
            weather_types = ["rain", "snow", "haze"]
        if splits is None:
            splits = ["train"]
        self.weather_types = list(weather_types)
        self.splits = list(splits)

        # 扫描样本
        # candidates_subdir 支持相对 (拼到 dataset_root/{weather}/{split}/<subdir>/) 和绝对两种模式,
        # 与 scripts/build_preference.py 中的保存逻辑保持一致.
        self.candidates_root = Path(candidates_subdir)
        self.candidates_is_absolute = self.candidates_root.is_absolute()
        self.samples: List[Dict] = []
        for weather in self.weather_types:
            for split in self.splits:
                gt_dir = self.dataset_root / weather / split / "GT"
                lq_dir = self.dataset_root / weather / split / "LQ"
                if self.candidates_is_absolute:
                    cand_dir = self.candidates_root / weather / split
                else:
                    cand_dir = self.dataset_root / weather / split / self.candidates_root
                if not (gt_dir.is_dir() and lq_dir.is_dir() and cand_dir.is_dir()):
                    logger_warning(
                        f"[跳过] {weather}/{split} 缺少 GT/LQ/candidates 目录"
                    )
                    continue

                gt_map = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and is_image(p)}
                for stem, gt_path in gt_map.items():
                    lq_path = lq_dir / gt_path.name
                    sample_cand_dir = cand_dir / stem
                    score_path = sample_cand_dir / "score.json"
                    if not lq_path.is_file() or not score_path.is_file():
                        continue
                    self.samples.append({
                        "weather": weather,
                        "split": split,
                        "stem": stem,
                        "lq_path": lq_path,
                        "gt_path": gt_path,
                        "cand_dir": sample_cand_dir,
                        "score_path": score_path,
                    })

        # 按 weather 限制样本数
        if self.weather_num_samples:
            grouped: Dict[str, List[Dict]] = {w: [] for w in self.weather_types}
            for s in self.samples:
                grouped[s["weather"]].append(s)
            new_samples: List[Dict] = []
            for weather in self.weather_types:
                limit = self.weather_num_samples.get(weather, 0)
                ws = grouped[weather]
                if limit and limit > 0 and limit < len(ws):
                    new_samples.extend(ws[:limit])
                else:
                    new_samples.extend(ws)
            self.samples = new_samples

        if not self.samples:
            raise FileNotFoundError(
                f"在 {self.dataset_root} 下未找到任何带 score.json 的偏好样本, "
                f"请先运行 scripts/build_preference.py"
            )

        # 输出 dataset 统计
        logger_warning(
            f"[DPO Dataset] 共 {len(self.samples)} 条偏好样本 "
            f"(candidates={candidates_subdir}, top_k={self.top_k_ratio}, "
            f"bottom_k={self.bottom_k_ratio}, min_gap={self.min_gap})"
        )
        for w in self.weather_types:
            cnt = sum(1 for s in self.samples if s["weather"] == w)
            logger_warning(f"  - {w}: {cnt}")

    def __len__(self):
        return len(self.samples)

    # ============================================================
    # 几何共享增强
    # ============================================================
    def _geo_augment(self, pil_img, rng: random.Random):
        """
        只做几何增强 (Flip / Scale / CenterCrop), 不碰颜色.
        rng: 独立的 random.Random, 保证四图共享同一决策.
        """
        # 1) 水平翻转
        if rng.random() < self.geo_flip_prob:
            pil_img = TF.hflip(pil_img)

        # 2) 短边随机缩放
        w, h = pil_img.size
        scale = rng.uniform(self.geo_scale_range[0], self.geo_scale_range[1])
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)

        # 3) 中心裁回固定分辨率
        pil_img = TF.center_crop(pil_img, self.resolution)

        # 4) 保证尺寸一致
        if pil_img.size != (self.resolution, self.resolution):
            pil_img = pil_img.resize(
                (self.resolution, self.resolution), Image.BILINEAR
            )
        return pil_img

    def _to_neg11(self, pil_img):
        """[0, 1] -> [-1, 1]"""
        return transforms.functional.to_tensor(pil_img) * 2.0 - 1.0

    def _to_01(self, pil_img):
        """[0, 255] -> [0, 1]"""
        return transforms.functional.to_tensor(pil_img)

    # ============================================================
    # 加载候选 (内存优化: 一次性读全部 cand, score 已在 __init__ 时读入)
    # ============================================================
    def _load_candidates(self, sample: Dict) -> Tuple[List[Dict], List[torch.Tensor]]:
        """读 score.json 并按当前 reward_weights 聚合分数; 按需加载候选图.

        聚合方式:
          - 默认 (normalize=false): 简单加权和, 受 PSNR 等大数值指标主导
          - normalize=true: 候选内 min-max 归一化后加权和, 各指标贡献与权重对齐,
            lpips 因为越小越好自动翻转
        """
        with open(sample["score_path"], "r", encoding="utf-8") as f:
            score_payload = json.load(f)
        cands_meta = score_payload["candidates"]
        # 候选内 min-max 归一化后加权和 (避免 PSNR 等数值大的指标主导)
        if self.normalize:
            scores = self._aggregate_normalized(cands_meta)
        else:
            # 简单加权和 (score.json 缺字段时容错默认 0)
            scores = []
            for c in cands_meta:
                s = 0.0
                for k, w in self.reward_weights.items():
                    v = c.get(k, 0.0)
                    # 兜底: score.json 出现 None/NaN/Inf 时按 0 处理
                    if v is None or not math.isfinite(v):
                        v = 0.0
                    s += w * v
                scores.append(s)
        scores_t = torch.tensor(scores)

        # Top-k / Bottom-k 池
        n = len(cands_meta)
        k_top = max(1, int(n * self.top_k_ratio))
        k_bot = max(1, int(n * self.bottom_k_ratio))

        order = torch.argsort(scores_t)  # 升序
        top_pool = order[-k_top:]        # 分数最大
        bot_pool = order[:k_bot]         # 分数最小

        return cands_meta, scores_t, top_pool, bot_pool

    def _aggregate_normalized(self, cands_meta: List[Dict]) -> List[float]:
        """候选内 min-max 归一化后加权和.

        流程:
          1. 每个 metric 沿候选方向 min-max 归一到 [0, 1]
          2. lpips 越小越好 → 翻转 (1 - norm)
          3. 加权求和, 权重正则化到 sum(|w|)=1 不必要 (代码略),
             实际只要各项不远离 0~1 即可

        例: 候选 A,B,C 的 PSNR=[24, 27, 25], lpips=[0.3, 0.15, 0.2]
             → norm_psnr=[0.0, 1.0, 0.33], norm_lpips=[0.0, 1.0, 0.67]
             → 翻转 lpips (越小越好): [1.0, 0.0, 0.33]
        """
        from math import isclose
        n = len(cands_meta)
        scores = [0.0] * n
        for m, w in self.reward_weights.items():
            if abs(w) < 1e-12:
                continue
            vals = [c.get(m, 0.0) for c in cands_meta]
            if not vals:
                continue
            lo, hi = min(vals), max(vals)
            if isclose(hi, lo, abs_tol=1e-9):
                norm = [0.5] * n      # 全相同, 中性贡献
            else:
                norm = [(v - lo) / (hi - lo) for v in vals]
            # lpips 越小越好 → 翻转
            if m == "lpips":
                norm = [1.0 - v for v in norm]
            for i, v in enumerate(norm):
                scores[i] += w * v
        return scores

    # ============================================================
    # 弱偏好过滤 + 抽样
    # ============================================================
    def _sample_pair(self, cands_meta, scores_t, top_pool, bot_pool, rng):
        # gap 检查
        score_w = float(scores_t[top_pool].max())
        score_l = float(scores_t[bot_pool].min())
        if score_w - score_l < self.min_gap:
            return None

        winner_idx = int(top_pool[rng.randint(0, len(top_pool) - 1)].item())
        loser_idx = int(bot_pool[rng.randint(0, len(bot_pool) - 1)].item())
        if winner_idx == loser_idx and len(top_pool) > 1:
            # 极端情况下 top/bot 池只有 1 个样本且重叠, 强制换一个 loser
            for _ in range(5):
                cand = int(bot_pool[rng.randint(0, len(bot_pool) - 1)].item())
                if cand != winner_idx:
                    loser_idx = cand
                    break
        return winner_idx, loser_idx

    # ============================================================
    # prompt 生成 (与 SFT 保持一致, 用同一个 random 状态)
    # ============================================================
    def _make_prompt(self, weather: str, rng: random.Random):
        """
        DPO 阶段通常 null_text_ratio=0, 但保留接口兼容 SFT.
        注意: winner / loser 共享同一 prompt, 所以由 collate_fn 复制 input_ids 即可.
        """
        if random.random() < self.null_text_ratio:
            return ""
        if not self.use_prompt:
            return ""
        if rng.random() < self.prompt_ratio:
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
        if not torch.is_tensor(result):
            result = torch.tensor(result)
        return result

    # ============================================================
    # __getitem__
    # ============================================================
    def __getitem__(self, index):
        # 弱偏好时向后跳
        for _ in range(self.max_skip):
            try:
                item = self._get(index)
                return item
            except _WeakPreferenceError:
                index = (index + 1) % len(self.samples)
        # max_skip 用尽仍失败, 在整个 dataset 扫一遍找任何可用样本 (force=True 模式)
        for offset in range(len(self.samples)):
            idx = (index + offset) % len(self.samples)
            try:
                return self._get(idx, force=True)
            except _WeakPreferenceError:
                continue
        # 实在找不到 (整个 dataset 都有问题), 抛错
        raise RuntimeError(
            "DPOPreferenceDataset: 没有任何 score.json 含 ≥2 个有效候选, "
            "请先运行 scripts/build_preference.py 生成偏好数据"
        )

    def _get(self, index, force=False):
        sample = self.samples[index]
        cands_meta, scores_t, top_pool, bot_pool = self._load_candidates(sample)

        # 防御: score.json "candidates" 为空 (例如 build_preference 把所有 NaN 候选过滤掉了)
        # → 至少要 2 张候选才能组成 winner/loser 对
        if len(cands_meta) < 2:
            raise _WeakPreferenceError(
                f"score.json 候选数 {len(cands_meta)} < 2, 跳过"
            )

        rng = random.Random(random.randint(0, 2**31 - 1))
        pair = self._sample_pair(cands_meta, scores_t, top_pool, bot_pool, rng)
        if pair is None and not force:
            raise _WeakPreferenceError("score gap < min_gap")
        winner_idx, loser_idx = pair

        # =================== 共享种子几何增强 ===================
        # 同一 rng 实例: 四图共享 flip / scale 决策
        win_pil = Image.open(
            sample["cand_dir"] / f"cand_{winner_idx:03d}.png"
        ).convert("RGB")
        lose_pil = Image.open(
            sample["cand_dir"] / f"cand_{loser_idx:03d}.png"
        ).convert("RGB")
        lq_pil = Image.open(sample["lq_path"]).convert("RGB")
        gt_pil = Image.open(sample["gt_path"]).convert("RGB")

        if self.augment_geo:
            win_pil = self._geo_augment(win_pil, rng)
            lose_pil = self._geo_augment(lose_pil, rng)
            lq_pil = self._geo_augment(lq_pil, rng)
            gt_pil = self._geo_augment(gt_pil, rng)
        else:
            # 仅做尺寸对齐
            win_pil = TF.center_crop(win_pil, self.resolution)
            lose_pil = TF.center_crop(lose_pil, self.resolution)
            lq_pil = TF.center_crop(lq_pil, self.resolution)
            gt_pil = TF.center_crop(gt_pil, self.resolution)
            for p in (win_pil, lose_pil, lq_pil, gt_pil):
                if p.size != (self.resolution, self.resolution):
                    p.resize((self.resolution, self.resolution), Image.BILINEAR)

        prompt = self._make_prompt(sample["weather"], rng)

        return {
            "pixel_values_winner": self._to_neg11(win_pil),
            "pixel_values_loser":  self._to_neg11(lose_pil),
            "pixel_values_gt":     self._to_neg11(gt_pil),
            "conditioning_pixel_values": self._to_01(lq_pil),
            "input_ids": self.tokenize_caption(prompt).squeeze(0),
            "weather":   sample["weather"],
            "score_w":   float(scores_t[winner_idx].item()),
            "score_l":   float(scores_t[loser_idx].item()),
        }


# ============================================================
# Collate: 把 winner/loser 沿 batch 维 cat, LQ / input_ids 复制 2 份
# ============================================================
def collate_fn_dpo(examples):
    """
    输入: list of dict (来自 DPOPreferenceDataset)
    输出: dict with
        pixel_values:            [2B, 3, H, W] in [-1, 1]  (前 B winner, 后 B loser)
        conditioning_pixel_values: [2B, 3, H, W] in [0, 1]
        input_ids:               [2B, L]
        pixel_values_gt:         [B, 3, H, W] (SFT 正则用, 可选)
    """
    win = torch.stack([e["pixel_values_winner"] for e in examples])
    lose = torch.stack([e["pixel_values_loser"] for e in examples])
    pixel_values = torch.cat([win, lose], dim=0).contiguous().float()

    lq = torch.stack([e["conditioning_pixel_values"] for e in examples])
    lq = torch.cat([lq, lq], dim=0).contiguous().float()

    ids = torch.stack([e["input_ids"] for e in examples])
    ids = torch.cat([ids, ids], dim=0)

    out = {
        "pixel_values":              pixel_values,
        "conditioning_pixel_values": lq,
        "input_ids":                 ids,
    }
    # 可选: GT 用于 SFT 正则
    if "pixel_values_gt" in examples[0]:
        out["pixel_values_gt"] = torch.stack(
            [e["pixel_values_gt"] for e in examples]
        ).contiguous().float()
    return out


# ============================================================
# Helpers
# ============================================================
class _WeakPreferenceError(Exception):
    pass


_DEFAULT_WEATHER_PROMPTS: Dict[str, str] = {
    "rain": "rainy scene, rain streaks on the image, wet surfaces, overcast sky",
    "snow": "snowy scene, snowflakes covering the image, cold atmosphere, white noise",
    "haze": "hazy scene, foggy atmosphere, low visibility, grayish tone",
}


def logger_warning(msg: str):
    # 与 dataloaders/paired_dataset.py 的 print 风格保持一致
    print(msg)
