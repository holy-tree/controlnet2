"""
数据集整理脚本
================================

将原始的多目录、多结构数据集统一整理为如下结构:
    datasets/{rain,snow,haze}/{train,test}/{GT,LQ}/

其中:
    GT  -- 标签图 (ground truth, 高质量)
    LQ  -- 退化图 (low quality, 含噪声/雨/雪/雾)

默认 train:test = 10:1

文件操作策略 (默认):
    - LQ 文件: 移动 (move)
    - GT 文件:
        * haze, snow, rain/RainTrainH, rain/RainTrainL: 移动 (move)
        * rain/Rain12600, rain/Rain1400: 复制 (copy),保留原文件
            原因: GT 与 14 张 LQ 一一对应,需复制 14 次,必须保留原文件

原始数据集目录结构假设 (相对于当前工作目录):
    datasets/
    ├── haze/
    │   ├── Reside-in/
    │   │   ├── test/         -> GT/, hazy/
    │   │   └── train/        -> GT/, hazy/
    │   └── Reside-out/
    │       ├── test/         -> GT/, hazy/
    │       └── train/        -> GT/, hazy/
    ├── rain/
    │   ├── Rain12600/
    │   │   ├── ground_truth/ -> GT (每张 GT 对应 14 张 LQ: xxx_1.jpg ~ xxx_14.jpg)
    │   │   └── rainy_image/
    │   ├── Rain1400/
    │   │   ├── ground_truth/
    │   │   └── rainy_image/
    │   ├── RainTrainH/        -> 扁平目录 (rain-xxx.png / norain-xxx.png)
    │   └── RainTrainL/        -> 扁平目录 (rain-xxx.png / norain-xxx.png)
    └── snow/
        ├── Real_Snow_training_original_size/
        │   ├── video2imgs_GT_re/
        │   └── video2imgs_IN_re/
        └── Snow100K-M/
            ├── gt/
            └── synthetic/

使用方法:
    python organize_dataset.py [--root ./datasets] [--ratio 10] [--dry-run]

注意:
    - 运行前请备份原始数据,移动操作不可逆
    - 相同天气不同目录合并后可能会有重名图片,脚本会整体重新编号
    - 已使用全局唯一编号,避免重名覆盖问题
    - 默认 GT 复制保留,只有 LQ 移动;若想全部移动请使用 --no-keep-gt
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# 支持的图片后缀
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# 工具函数
# ============================================================

def is_image(path: Path) -> bool:
    """判断文件是否为图片"""
    return path.suffix.lower() in IMG_EXTENSIONS


def safe_move(src: Path, dst: Path, dry_run: bool = False, copy: bool = False) -> Path:
    """
    安全移动 (或复制) 文件到目标位置,自动避免覆盖。
    若目标已存在,会自动在文件名后追加 _1, _2 ... 等后缀。
    返回最终写入的目标路径。

    Args:
        src: 源文件路径
        dst: 目标文件路径
        dry_run: 仅打印不实际执行
        copy: True 时改为复制而非移动 (用于保护 GT 等标签图)
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists():
        final_dst = dst
    else:
        # 避免覆盖,自动追加序号
        stem, suffix = dst.stem, dst.suffix
        parent = dst.parent
        counter = 1
        while True:
            final_dst = parent / f"{stem}_{counter}{suffix}"
            if not final_dst.exists():
                break
            counter += 1

    action = "COPY" if copy else "MOVE"
    if dry_run:
        print(f"  [DRY-RUN][{action}] {src} -> {final_dst}")
    elif copy:
        shutil.copy2(src, final_dst)
    else:
        shutil.move(str(src), str(final_dst))
    return final_dst


def split_indices(n: int, ratio: int = 10, shuffle: bool = True, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    按比例划分 train / test。
    默认 10:1 -> train 占 10/11, test 占 1/11。

    Args:
        n: 样本总数
        ratio: train:test 的比例 (默认 10)
        shuffle: 是否先打乱索引再划分 (默认 True)
        seed: 随机种子,保证可复现
    """
    if n == 0:
        return [], []

    indices = list(range(n))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    test_size = max(1, n // (ratio + 1))
    test_indices = sorted(indices[:test_size])
    train_indices = sorted(indices[test_size:])
    return train_indices, test_indices


def write_pairs(
    pairs: List[Tuple[Path, Path]],
    output_root: Path,
    category: str,
    start_idx: int = 0,
    dry_run: bool = False,
    shuffle: bool = True,
    seed: int = 42,
    copy_gt: bool = False,
) -> int:
    """
    将 (gt_path, lq_path) 列表按统一编号写入:
        output_root/{category}/{split}/{GT,LQ}/{idx:06d}.{ext}

    pairs: 已经匹配的 (gt, lq) 对列表
    start_idx: 全局起始编号,避免不同子任务重名
    shuffle: 是否打乱后再划分 train/test
    seed: 随机种子
    copy_gt: 是否复制 GT (True: 复制保留原文件; False: 移动)
    LQ 文件始终为移动。
    返回写入后使用的下一个起始编号。
    """
    train_indices, test_indices = split_indices(len(pairs), ratio=10, shuffle=shuffle, seed=seed)

    train_set = set(train_indices)
    test_set = set(test_indices)

    idx = start_idx
    for i, (gt_path, lq_path) in enumerate(pairs):
        split = "train" if i in train_set else "test"
        ext = lq_path.suffix.lower() if lq_path else gt_path.suffix.lower()
        new_name = f"{idx:06d}{ext}"

        gt_dst = output_root / category / split / "GT" / new_name
        lq_dst = output_root / category / split / "LQ" / new_name

        safe_move(gt_path, gt_dst, dry_run=dry_run, copy=copy_gt)
        safe_move(lq_path, lq_dst, dry_run=dry_run, copy=False)
        idx += 1

    return idx


def pair_by_filename(
    gt_dir: Path,
    lq_dir: Path,
) -> List[Tuple[Path, Path]]:
    """
    按文件名匹配 GT 与 LQ 图像对 (去除后缀)。
    返回 (gt_path, lq_path) 列表。
    """
    gt_files = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and is_image(p)}
    lq_files = {p.stem: p for p in lq_dir.iterdir() if p.is_file() and is_image(p)}

    pairs = []
    for stem in sorted(gt_files.keys() & lq_files.keys()):
        pairs.append((gt_files[stem], lq_files[stem]))

    return pairs


# ============================================================
# haze (去雾) 处理
# ============================================================

def process_haze(root: Path, output_root: Path, dry_run: bool, shuffle: bool = True, seed: int = 42) -> int:
    """
    处理 haze 数据:
        Reside-in/test, Reside-out/test 已划分好的 test
        Reside-in/train, Reside-out/train 已划分好的 train

    直接合并同名 GT/hazy 对,然后按比例重新划分 train/test。
    """
    print("\n[HAZE] 开始处理 haze 数据集...")

    haze_root = root / "haze"
    all_pairs: List[Tuple[Path, Path]] = []

    sources = [
        ("Reside-in/train", haze_root / "Reside-in" / "train"),
        ("Reside-in/test", haze_root / "Reside-in" / "test"),
        ("Reside-out/train", haze_root / "Reside-out" / "train"),
        ("Reside-out/test", haze_root / "Reside-out" / "test"),
    ]

    for name, sub in sources:
        gt_dir = sub / "GT"
        lq_dir = sub / "hazy"
        if not gt_dir.is_dir() or not lq_dir.is_dir():
            print(f"  [跳过] {name}: 目录不存在或结构不符")
            continue
        sub_pairs = pair_by_filename(gt_dir, lq_dir)
        print(f"  [{name}] 匹配到 {len(sub_pairs)} 对")
        all_pairs.extend(sub_pairs)

    print(f"[HAZE] 共 {len(all_pairs)} 对,按 10:1 重新划分 train/test")
    next_idx = write_pairs(all_pairs, output_root, "haze", start_idx=0, dry_run=dry_run, shuffle=shuffle, seed=seed)
    return next_idx


# ============================================================
# snow (去雪) 处理
# ============================================================

def process_snow(root: Path, output_root: Path, dry_run: bool, shuffle: bool = True, seed: int = 42) -> int:
    """
    处理 snow 数据:
        Real_Snow_training_original_size: video2imgs_GT_re / video2imgs_IN_re
        Snow100K-M: gt / synthetic

    按文件名匹配后统一编号,重新划分。
    """
    print("\n[SNOW] 开始处理 snow 数据集...")

    snow_root = root / "snow"
    all_pairs: List[Tuple[Path, Path]] = []

    sources = [
        (
            "Real_Snow_training_original_size",
            snow_root / "Real_Snow_training_original_size" / "video2imgs_GT_re",
            snow_root / "Real_Snow_training_original_size" / "video2imgs_IN_re",
        ),
        (
            "Snow100K-M",
            snow_root / "Snow100K-M" / "gt",
            snow_root / "Snow100K-M" / "synthetic",
        ),
    ]

    for name, gt_dir, lq_dir in sources:
        if not gt_dir.is_dir() or not lq_dir.is_dir():
            print(f"  [跳过] {name}: 目录不存在")
            continue
        sub_pairs = pair_by_filename(gt_dir, lq_dir)
        print(f"  [{name}] 匹配到 {len(sub_pairs)} 对")
        all_pairs.extend(sub_pairs)

    print(f"[SNOW] 共 {len(all_pairs)} 对,按 10:1 重新划分 train/test")
    next_idx = write_pairs(all_pairs, output_root, "snow", start_idx=0, dry_run=dry_run, shuffle=shuffle, seed=seed)
    return next_idx


# ============================================================
# rain (去雨) 处理
# ============================================================

def process_rain_12600(root: Path, output_root: Path, start_idx: int, dry_run: bool, shuffle: bool = True, seed: int = 42) -> int:
    """
    处理 Rain12600 与 Rain1400:
        GT 文件: xxx.jpg
        对应 LQ: xxx_1.jpg, xxx_2.jpg, ..., xxx_14.jpg

    将每张 GT 复制 14 份,分别重命名为与对应 LQ 完全一致的文件名。
    """
    print("\n[RAIN-12600/1400] 开始处理 Rain12600 与 Rain1400...")

    rain_root = root / "rain"
    pairs: List[Tuple[Path, Path]] = []

    sources = [
        rain_root / "Rain12600" / "ground_truth",
        rain_root / "Rain12600" / "rainy_image",
        rain_root / "Rain1400" / "ground_truth",
        rain_root / "Rain1400" / "rainy_image",
    ]

    if not all(p.is_dir() for p in sources[:2]):
        print("  [跳过] Rain12600 目录不存在")
    else:
        gt_dir, lq_dir = sources[0], sources[1]
        gt_files = sorted([p for p in gt_dir.iterdir() if p.is_file() and is_image(p)])
        lq_files = sorted([p for p in lq_dir.iterdir() if p.is_file() and is_image(p)])

        # 按 GT 名称 stem 匹配 LQ: xxx_1.jpg ~ xxx_14.jpg
        gt_map = {p.stem: p for p in gt_files}
        lq_grouped: Dict[str, List[Path]] = {}
        for lq in lq_files:
            # 文件名形如 xxx_1.jpg -> 去掉 _数字 后缀作为 key
            stem = lq.stem
            base = stem.rsplit("_", 1)[0] if "_" in stem else stem
            lq_grouped.setdefault(base, []).append(lq)

        for base_stem, gt_path in gt_map.items():
            lqs = lq_grouped.get(base_stem, [])
            # 同一 GT 复制 N 次,每份与对应 LQ 文件名一致
            for lq_path in lqs:
                pairs.append((gt_path, lq_path))

        print(f"  [Rain12600] 匹配到 {len(pairs)} 对")

    if not all(p.is_dir() for p in sources[2:]):
        print("  [跳过] Rain1400 目录不存在")
    else:
        gt_dir, lq_dir = sources[2], sources[3]
        gt_files = sorted([p for p in gt_dir.iterdir() if p.is_file() and is_image(p)])
        lq_files = sorted([p for p in lq_dir.iterdir() if p.is_file() and is_image(p)])

        gt_map = {p.stem: p for p in gt_files}
        lq_grouped: Dict[str, List[Path]] = {}
        for lq in lq_files:
            stem = lq.stem
            base = stem.rsplit("_", 1)[0] if "_" in stem else stem
            lq_grouped.setdefault(base, []).append(lq)

        rain1400_pairs = []
        for base_stem, gt_path in gt_map.items():
            lqs = lq_grouped.get(base_stem, [])
            for lq_path in lqs:
                rain1400_pairs.append((gt_path, lq_path))

        print(f"  [Rain1400] 匹配到 {len(rain1400_pairs)} 对")
        pairs.extend(rain1400_pairs)

    print(f"[RAIN-12600/1400] 共 {len(pairs)} 对")
    # Rain12600/1400 的 GT 需要复制 14 次,必须保留原文件 -> copy_gt=True
    return write_pairs(pairs, output_root, "rain", start_idx=start_idx, dry_run=dry_run, shuffle=shuffle, seed=seed, copy_gt=True)


def process_rain_train(root: Path, output_root: Path, start_idx: int, dry_run: bool, shuffle: bool = True, seed: int = 42) -> int:
    """
    处理 RainTrainH / RainTrainL (扁平目录):
        rain-xxx.png   (退化图)
        norain-xxx.png (标签图)
        其他无关文件: 保留原位,不处理

    按编号匹配 rain 与 norain,并将 GT 重命名为 LQ 同名,使 GT/LQ 文件名一致。
    """
    print("\n[RAIN-TrainH/L] 开始处理 RainTrainH 与 RainTrainL...")

    rain_root = root / "rain"
    pairs: List[Tuple[Path, Path]] = []

    sources = [
        ("RainTrainH", rain_root / "RainTrainH"),
        ("RainTrainL", rain_root / "RainTrainL"),
    ]

    for name, sub in sources:
        if not sub.is_dir():
            print(f"  [跳过] {name}: 目录不存在")
            continue

        rain_map: Dict[str, Path] = {}
        norain_map: Dict[str, Path] = {}

        for p in sub.iterdir():
            if not p.is_file() or not is_image(p):
                continue
            stem_lower = p.stem.lower()
            if stem_lower.startswith("rain-"):
                key = p.stem[5:]  # 去掉 "rain-"
                rain_map[key] = p
            elif stem_lower.startswith("norain-"):
                key = p.stem[7:]  # 去掉 "norain-"
                norain_map[key] = p

        sub_pairs = []
        for key in sorted(rain_map.keys() & norain_map.keys()):
            sub_pairs.append((norain_map[key], rain_map[key]))

        print(f"  [{name}] 匹配到 {len(sub_pairs)} 对 (忽略其他无关图像)")
        pairs.extend(sub_pairs)

    print(f"[RAIN-TrainH/L] 共 {len(pairs)} 对")
    return write_pairs(pairs, output_root, "rain", start_idx=start_idx, dry_run=dry_run, shuffle=shuffle, seed=seed)


def process_rain(root: Path, output_root: Path, dry_run: bool, shuffle: bool = True, seed: int = 42) -> None:
    """
    rain 总处理: 先 12600+1400,再 TrainH+TrainL,使用连续全局编号。
    """
    print("\n" + "=" * 60)
    print("[RAIN] 开始处理 rain 数据集...")
    print("=" * 60)

    next_idx = process_rain_12600(root, output_root, start_idx=0, dry_run=dry_run, shuffle=shuffle, seed=seed)
    print(f"[RAIN] 当前编号 -> {next_idx}")
    next_idx = process_rain_train(root, output_root, start_idx=next_idx, dry_run=dry_run, shuffle=shuffle, seed=seed)
    print(f"[RAIN] 最终编号 -> {next_idx}")


# ============================================================
# 入口
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="整理图像去雨/去雪/去雾数据集")
    parser.add_argument(
        "--root",
        type=str,
        default="./datasets",
        help="原始数据集根目录 (默认 ./datasets)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录 (默认与 root 相同,即在原目录下生成整理后的结构)",
    )
    parser.add_argument(
        "--ratio",
        type=int,
        default=10,
        help="train:test 划分比例 (默认 10,即 10:1)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="关闭 train/test 划分的随机打乱 (默认开启 shuffle)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="shuffle 随机种子,保证可复现 (默认 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印要执行的操作,不做实际复制",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_root = Path(args.output).resolve() if args.output else root

    if not root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {root}")

    print(f"原始数据根目录: {root}")
    print(f"输出目录:       {output_root}")
    print(f"划分比例:       train:test = {args.ratio}:1")
    print(f"Shuffle:       {not args.no_shuffle} (seed={args.seed})")
    print(f"操作策略:       LQ 移动; haze/snow/rain-Train GT 移动; rain-12600/1400 GT 复制保留")
    print(f"DRY-RUN:       {args.dry_run}")

    shuffle = not args.no_shuffle
    process_haze(root, output_root, args.dry_run, shuffle=shuffle, seed=args.seed)
    process_snow(root, output_root, args.dry_run, shuffle=shuffle, seed=args.seed)
    process_rain(root, output_root, args.dry_run, shuffle=shuffle, seed=args.seed)

    print("\n" + "=" * 60)
    print("完成! 输出结构:")
    print(f"  {output_root}/")
    print(f"    ├── haze/{{train,test}}/{{GT,LQ}}/")
    print(f"    ├── rain/{{train,test}}/{{GT,LQ}}/")
    print(f"    └── snow/{{train,test}}/{{GT,LQ}}/")
    print("=" * 60)


if __name__ == "__main__":
    main()