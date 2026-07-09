# Sobel 细节筛选工具

按 Sobel 梯度幅值对 GT 图像排序, 选出"细节最丰富"的图像, 把入选路径写入 JSON.
用于 SFT / DPO 训练前的"纹理筛选"步骤 (例如每种天气保留 7000 张).

## 用法

### 多天气模式 (推荐, 与项目数据结构对齐)

```bash
python scripts/filter_dataset/filter_by_sobel.py --gt_root D:\Projects\pycharm\WeaFU-main\dataprocess --weather_types rain snow haze --splits train --top_k 7 --output_json ./experiment/filtered/top7000_per_weather.json --num_workers 8 --metric mean
```

目录结构假设: `{gt_root}/{weather}/{split}/GT/*.png`
对每个 `(weather, split)` 组合分别排序, 各取 `--top_k`,
最终输出一个统一 JSON, 每条 sample 含 `weather` / `split` 字段.

### 单目录模式 (向后兼容)

```bash
python scripts/filter_dataset/filter_by_sobel.py \
    --gt_root D:/data/weafu/rain/train/GT \
    --top_k 7000 \
    --output_json ./experiment/filtered/rain.json
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--gt_root` | (必填) | 单目录模式: GT 目录; 多天气模式: 数据集根目录 |
| `--output_json` | (必填) | 输出 JSON 路径 |
| `--top_k` | 7000 | 多天气模式: 每种天气保留 N 张; 单目录: 总保留 N 张 |
| `--weather_types` | None | 启用多天气模式 |
| `--splits` | `train` | 参与筛选的划分 |
| `--metric` | `mean` | `mean` / `std` / `mean_std` |
| `--num_workers` | 8 | 并行 worker 数 |
| `--resize` | None | 短边 resize 后再算 (加速) |
| `--min_score` | None | 最低分数阈值 |
| `--save_scores` | False | 单目录模式: 写全量分数 (debug) |

## 指标选择

- `mean`: Sobel 幅值均值 — 整体纹理/对比度, 推荐
- `std`: Sobel 幅值标准差 — 边缘分布的离散度
- `mean_std`: 复合指标, 同时考虑整体强度和分布

## 多天气模式输出 JSON

```json
{
  "meta": {
    "gt_root": "D:/data/weafu",
    "weather_types": ["rain", "snow", "haze"],
    "splits": ["train"],
    "per_weather_top_k": 7000,
    "total_selected": 21000,
    "metric": "mean"
  },
  "weather_stats": {
    "rain/train":  {"total_scanned": 8521, "num_valid": 8510, "num_invalid": 11,
                    "num_selected": 7000, "score_min": 12.3, "score_max": 88.1, "score_mean": 41.2},
    "snow/train":  {"total_scanned": 7342, "num_valid": 7342, "num_invalid": 0,
                    "num_selected": 7000, "score_min": 11.0, "score_max": 79.5, "score_mean": 38.7},
    "haze/train":  {"total_scanned": 6921, "num_valid": 6918, "num_invalid": 3,
                    "num_selected": 6921, "score_min":  9.4, "score_max": 72.1, "score_mean": 35.9}
  },
  "samples": [
    {"path": "D:/.../rain/train/GT/0001.png", "score": 88.10, "weather": "rain", "split": "train"},
    ...
  ]
}
```

注意 `num_selected` 可能小于 `top_k` (当该天气样本不足时).

## 配合 DPO 流程

1. 本脚本筛选细节丰富的 GT 子集 (每种天气 7000 张)
2. 把 JSON 中 `samples` 列表按 `weather` 字段分组, 拷/链到
   `dataset_root/{weather}/train/GT_top/` 子目录 (可选)
3. 用 `scripts/build_preference.py` 在子集上构建偏好对
4. 用 `train_controlnet.py --train_method dpo` 做 DPO 微调
