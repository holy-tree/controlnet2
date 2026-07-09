"""
测试集整理脚本 (服务器原始结构 -> 标准结构)
====================================================

服务器原始结构:
    datasets/
    ├── haze/
    │   ├── SOTS/
    │   │   ├── nyuhaze500/{gt,hazy}/     # hazy 命名: xxxx_y.png (y=1..10), 1:10 对应
    │   │   └── outdoor/{gt,hazy}/        # hazy 命名: xxxx_0.8_0.2.jpg, GT 命名: xxxx.png
    ├── rain/
    │   ├── Rain100H/rainy/               # 同级目录有 rain-xxx.png 与 norain-xxx.png
    │   ├── Rain100L/rainy/
    │   ├── outdoor/{gt,input}/
    │   └── raindrop/
    │       ├── test_a/{data,gt}/         # data/xx_rain.jpg, gt/xx_clean.jpg
    │       └── test_b/{data,gt}/
    └── snow/
        ├── Snow100K-L/{gt,synthetic}/
        └── Snow100K-S/{gt,synthetic}/

目标结构:
    datasets_test/
    ├── haze/
    │   ├── SOTS_nyuhaze500/{gt,lq}/      # hazy -> lq, GT 不变
    │   └── SOTS_outdoor/{gt,lq}/         # hazy -> lq, GT 不变
    ├── rain/
    │   ├── Rain100H/{gt,lq}/
    │   ├── Rain100L/{gt,lq}/
    │   ├── outdoor/{gt,lq}/              # input -> lq
    │   └── raindrop/{gt,lq}/             # test_a + test_b 融合,统一命名
    └── snow/
        ├── Snow100K-L/{gt,lq}/           # synthetic -> lq
        └── Snow100K-S/{gt,lq}/           # synthetic -> lq

使用方法:
    python organize_testset.py [--root /path/to/datasets] [--output /path/to/output]

注意:
    - 默认所有文件复制 (copy),不破坏原始数据
    - rain/Rain100H 与 Rain100L: 把同级 norain-xxx.png 移入新建的 gt/ 目录并重命名为 rain-xxx.png
    - raindrop/test_a + test_b 统一命名: 使用 prefix 区分, 如 a_001.png / b_001.png
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTENSIONS


def safe_copy(src: Path, dst: Path, dry_run: bool = False) -> Path:
    """复制文件到目标位置, 自动避免覆盖."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        final = dst
    else:
        stem, suf = dst.stem, dst.suffix
        i = 1
        while True:
            final = dst.parent / f"{stem}_{i}{suf}"
            if not final.exists():
                break
            i += 1
    if dry_run:
        print(f"  [DRY] COPY {src.name} -> {final}")
    else:
        shutil.copy2(src, final)
    return final


def safe_move(src: Path, dst: Path, dry_run: bool = False) -> Path:
    """移动文件到目标位置, 自动避免覆盖."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        final = dst
    else:
        stem, suf = dst.stem, dst.suffix
        i = 1
        while True:
            final = dst.parent / f"{stem}_{i}{suf}"
            if not final.exists():
                break
            i += 1
    if dry_run:
        print(f"  [DRY] MOVE {src.name} -> {final}")
    else:
        shutil.move(str(src), str(final))
    return final


# ============================================================
# haze 处理
# ============================================================

def organize_haze(root: Path, out_root: Path, dry_run: bool) -> int:
    """haze: nyuhaze500 + outdoor. hazy 复制为 lq."""
    print("\n[HAZE] 处理 haze 数据集...")
    haze_root = root / "haze"
    count = 0

    # nyuhaze500
    # GT:LQ = 1:10, hazy 命名为 xxxx_y.png (y=1..10)
    # 每张 GT 复制 10 份, 分别与 10 张 LQ 同名匹配
    src = haze_root / "SOTS" / "nyuhaze500"
    if src.is_dir():
        gt_dir = src / "gt"
        hazy_dir = src / "hazy"
        dst = out_root / "haze" / "SOTS_nyuhaze500"
        if gt_dir.is_dir() and hazy_dir.is_dir():
            gt_map = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and is_image(p)}
            hazy_map = {p.stem: p for p in hazy_dir.iterdir() if p.is_file() and is_image(p)}
            print(f"  [nyuhaze500] GT={len(gt_map)} files, Hazy={len(hazy_map)} files")

            # 按 GT 名分组 LQ: 例如 GT 1400 对应 hazy 1400_1 ~ 1400_10
            # hazy 命名: xxxx_y.png  -> stem 是 "xxxx_y", 去掉 _y 找到对应 GT stem "xxxx"
            hazy_grouped: Dict[str, List[Path]] = {}
            for hazy_stem, hazy_path in hazy_map.items():
                # 提取 stem 前缀 (去掉 _数字 后缀)
                if "_" in hazy_stem:
                    gt_stem = hazy_stem.rsplit("_", 1)[0]
                else:
                    gt_stem = hazy_stem
                hazy_grouped.setdefault(gt_stem, []).append(hazy_path)

            # 遍历 GT, 每张复制 N 份 (N = 该 GT 对应的 LQ 数量)
            for gt_stem, gt_path in gt_map.items():
                lqs = hazy_grouped.get(gt_stem, [])
                if not lqs:
                    print(f"    [警告] GT {gt_stem} 没有对应的 LQ, 跳过")
                    continue
                # 每张 GT 复制 len(lqs) 次, 文件名与对应 LQ 一致
                for lq_path in lqs:
                    # LQ 的最终文件名 (用 LQ 自身的 stem + suffix)
                    lq_final_name = lq_path.name
                    safe_copy(lq_path, dst / "lq" / lq_final_name, dry_run)
                    safe_copy(gt_path, dst / "gt" / lq_final_name, dry_run)
                    count += 2  # 算一对
            print(f"  [nyuhaze500] 处理完成, 共生成 {count} 文件 (含 GT 复制)")
        else:
            print(f"  [跳过 nyuhaze500] GT/hazy 目录不全")
    else:
        print(f"  [跳过 nyuhaze500] 目录不存在: {src}")

    # outdoor
    src = haze_root / "SOTS" / "outdoor"
    if src.is_dir():
        gt_dir = src / "gt"
        hazy_dir = src / "hazy"
        dst = out_root / "haze" / "SOTS_outdoor"
        if gt_dir.is_dir() and hazy_dir.is_dir():
            print(f"  [outdoor] GT={len(list(gt_dir.iterdir()))} files, Hazy={len(list(hazy_dir.iterdir()))} files")
            for f in gt_dir.iterdir():
                if f.is_file() and is_image(f):
                    safe_copy(f, dst / "gt" / f.name, dry_run)
                    count += 1
            for f in hazy_dir.iterdir():
                if f.is_file() and is_image(f):
                    safe_copy(f, dst / "lq" / f.name, dry_run)
                    count += 1
        else:
            print(f"  [跳过 outdoor] GT/hazy 目录不全")
    else:
        print(f"  [跳过 outdoor] 目录不存在: {src}")

    print(f"[HAZE] 共处理 {count} 个文件")
    return count


# ============================================================
# rain 处理
# ============================================================

def _process_rain_flat(src_dir: Path, dst: Path, dry_run: bool) -> int:
    """
    Rain100H / Rain100L 处理:
        服务器结构: Rain100H/rainy/rain-xxx.png, Rain100H/norain-xxx.png (norain 在 rainy 同级)
        也兼容:    全部文件混在同一目录

        -> 将 norain-xxx.png 复制为 gt/rain-xxx.png (重命名, 复制而非移动以保护原始数据)
        -> 将 rain-xxx.png 复制为 lq/rain-xxx.png (复制保护)
    """
    if not src_dir.is_dir():
        print(f"  [跳过] {src_dir} 不存在")
        return 0

    rain_map: Dict[str, Path] = {}
    norain_map: Dict[str, Path] = {}

    # 1. 扫描 src_dir (通常是 rainy/ 目录) 里的 rain-xxx.png
    for p in src_dir.iterdir():
        if not p.is_file() or not is_image(p):
            continue
        sl = p.stem.lower()
        if sl.startswith("rain-"):
            rain_map[p.stem[5:]] = p

    # 2. 扫描 src_dir.parent (例如 Rain100H/) 里的 norain-xxx.png
    #    服务器上 norain 与 rainy 是同级目录
    parent_dir = src_dir.parent
    if parent_dir.is_dir():
        for p in parent_dir.iterdir():
            if not p.is_file() or not is_image(p):
                continue
            sl = p.stem.lower()
            if sl.startswith("norain-"):
                norain_map[p.stem[7:]] = p

    # 3. 兜底: 若 src_dir 里也有 norain (兼容旧结构), 也接受
    for p in src_dir.iterdir():
        if not p.is_file() or not is_image(p):
            continue
        sl = p.stem.lower()
        if sl.startswith("norain-"):
            norain_map.setdefault(p.stem[7:], p)

    print(f"  [{src_dir.name}] rain={len(rain_map)} norain={len(norain_map)}")

    if not norain_map:
        print(f"  [跳过 {src_dir.name}] 未找到 norain-xxx.png (GT)")
        return 0

    count = 0
    for key in sorted(rain_map.keys() & norain_map.keys()):
        # norain -> gt/rain-xxx.png (复制 + 重命名)
        safe_copy(norain_map[key], dst / "gt" / f"rain-{key}{norain_map[key].suffix}", dry_run)
        # rain -> lq/rain-xxx.png (复制 + 重命名, 文件名已一致)
        safe_copy(rain_map[key], dst / "lq" / f"rain-{key}{rain_map[key].suffix}", dry_run)
        count += 1
    return count


def _process_outdoor(src_dir: Path, dst: Path, dry_run: bool) -> int:
    """
    rain/outdoor:
        gt/xxxx.png
        input/yyyy (hazy)
    命名上 GT 为 xxxx.png, hazy 一般为 xxxx.png (同名匹配), 也可能不同.
    此处按 stem 匹配 (假设同名).
    """
    if not src_dir.is_dir():
        print(f"  [跳过] {src_dir} 不存在")
        return 0

    gt_dir = src_dir / "gt"
    input_dir = src_dir / "input"
    if not gt_dir.is_dir() or not input_dir.is_dir():
        print(f"  [跳过 outdoor] GT/input 目录不全")
        return 0

    gt_map = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and is_image(p)}
    input_map = {p.stem: p for p in input_dir.iterdir() if p.is_file() and is_image(p)}

    common = sorted(set(gt_map.keys()) & set(input_map.keys()))
    print(f"  [outdoor] GT={len(gt_map)} input={len(input_map)} common={len(common)}")

    count = 0
    for stem in common:
        safe_copy(gt_map[stem], dst / "gt" / f"{stem}{gt_map[stem].suffix}", dry_run)
        safe_copy(input_map[stem], dst / "lq" / f"{stem}{input_map[stem].suffix}", dry_run)
        count += 1
    return count


def _process_raindrop(src_dir: Path, dst: Path, dry_run: bool, prefix: str = "") -> int:
    """
    raindrop/test_a 与 test_b:
        data/xx_rain.jpg     -> lq/xx_rain.jpg (统一为原名, 或者用 prefix 重命名防冲突)
        gt/xx_clean.jpg      -> gt/xx_rain.jpg (重命名为与 lq 一致)
    融合多个 test 时加 prefix 以避免重名冲突.
    """
    if not src_dir.is_dir():
        print(f"  [跳过] {src_dir} 不存在")
        return 0

    data_dir = src_dir / "data"
    gt_dir = src_dir / "gt"
    if not data_dir.is_dir() or not gt_dir.is_dir():
        print(f"  [跳过 {src_dir.name}] data/gt 目录不全")
        return 0

    data_map = {p.stem: p for p in data_dir.iterdir() if p.is_file() and is_image(p)}
    gt_map = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and is_image(p)}

    print(f"  [raindrop/{src_dir.name}] data={len(data_map)} gt={len(gt_map)}")

    count = 0
    # 命名规范: 把 _rain / _clean 后缀去掉, 统一用 stem; 加 prefix 防重名
    # 假设原始: 1_rain.jpg / 1_clean.jpg
    #   -> lq/prefix_1.png, gt/prefix_1.png
    # 用 data 的 stem 作为 key (假设 _rain 后缀可去掉得到唯一编号)
    for data_stem, data_path in sorted(data_map.items()):
        # 去掉可能的 _rain 后缀
        base = re.sub(r"_rain$", "", data_stem, flags=re.IGNORECASE)
        # 在 gt map 中找匹配 (去 _clean 后缀)
        gt_candidate = None
        for gt_stem, gt_path in gt_map.items():
            if re.sub(r"_clean$", "", gt_stem, flags=re.IGNORECASE) == base:
                gt_candidate = gt_path
                break

        if gt_candidate is None:
            # 不匹配, 跳过
            continue

        # 统一命名: prefix + base, 扩展名取 gt 的 (因为 gt 通常是 png)
        ext = gt_candidate.suffix.lower()
        new_name = f"{prefix}{base}{ext}"
        safe_copy(data_path, dst / "lq" / new_name, dry_run)
        safe_copy(gt_candidate, dst / "gt" / new_name, dry_run)
        count += 1
    return count


def organize_rain(root: Path, out_root: Path, dry_run: bool) -> int:
    print("\n[RAIN] 处理 rain 数据集...")
    rain_root = root / "rain"
    count = 0

    # Rain100H, Rain100L
    for sub in ["Rain100H", "Rain100L"]:
        src = rain_root / sub / "rainy"
        if not src.is_dir():
            # 兼容: 也可能在 Rain100H 直接同级
            src = rain_root / sub
        dst = out_root / "rain" / sub
        count += _process_rain_flat(src, dst, dry_run)

    # outdoor
    src = rain_root / "outdoor"
    dst = out_root / "rain" / "outdoor"
    count += _process_outdoor(src, dst, dry_run)

    # raindrop (test_a + test_b 融合)
    rd_dir = rain_root / "raindrop"
    if rd_dir.is_dir():
        dst = out_root / "rain" / "raindrop"
        # test_a -> prefix "a_", test_b -> prefix "b_"
        c_a = _process_raindrop(rd_dir / "test_a", dst, dry_run, prefix="a_")
        c_b = _process_raindrop(rd_dir / "test_b", dst, dry_run, prefix="b_")
        print(f"  [raindrop] test_a={c_a}, test_b={c_b}")
        count += c_a + c_b
    else:
        print(f"  [跳过 raindrop] {rd_dir} 不存在")

    print(f"[RAIN] 共处理 {count} 个文件")
    return count


# ============================================================
# snow 处理
# ============================================================

def organize_snow(root: Path, out_root: Path, dry_run: bool) -> int:
    print("\n[SNOW] 处理 snow 数据集...")
    snow_root = root / "snow"
    count = 0

    for sub in ["Snow100K-L", "Snow100K-S"]:
        src = snow_root / sub
        if not src.is_dir():
            print(f"  [跳过] {src} 不存在")
            continue

        gt_dir = src / "gt"
        syn_dir = src / "synthetic"
        dst = out_root / "snow" / sub
        if not gt_dir.is_dir() or not syn_dir.is_dir():
            print(f"  [跳过 {sub}] GT/synthetic 目录不全")
            continue

        gt_map = {p.stem: p for p in gt_dir.iterdir() if p.is_file() and is_image(p)}
        syn_map = {p.stem: p for p in syn_dir.iterdir() if p.is_file() and is_image(p)}
        common = sorted(set(gt_map.keys()) & set(syn_map.keys()))
        print(f"  [{sub}] GT={len(gt_map)} synthetic={len(syn_map)} common={len(common)}")

        for stem in common:
            safe_copy(gt_map[stem], dst / "gt" / f"{stem}{gt_map[stem].suffix}", dry_run)
            safe_copy(syn_map[stem], dst / "lq" / f"{stem}{syn_map[stem].suffix}", dry_run)
            count += 1

    print(f"[SNOW] 共处理 {count} 个文件")
    return count


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="整理服务器测试集 (haze/rain/snow) 到统一 GT/LQ 结构"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="原始数据集根目录 (含 haze/, rain/, snow/ 的目录)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录 (默认与 root 同级, 名为 datasets_test)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印不实际执行",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {root}")

    output_root = Path(args.output).resolve() if args.output else root.parent / "datasets_test"
    print(f"原始数据根目录: {root}")
    print(f"输出目录:       {output_root}")
    print(f"DRY-RUN:        {args.dry_run}")

    # 各天气独立输出, 不互相覆盖
    organize_haze(root, output_root, args.dry_run)
    organize_rain(root, output_root, args.dry_run)
    organize_snow(root, output_root, args.dry_run)

    print("\n" + "=" * 60)
    print("完成! 输出结构:")
    print(f"  {output_root}/")
    print(f"    haze/SOTS_nyuhaze500/{{gt,lq}}/")
    print(f"    haze/SOTS_outdoor/{{gt,lq}}/")
    print(f"    rain/Rain100H/{{gt,lq}}/")
    print(f"    rain/Rain100L/{{gt,lq}}/")
    print(f"    rain/outdoor/{{gt,lq}}/")
    print(f"    rain/raindrop/{{gt,lq}}/        # test_a + test_b 融合")
    print(f"    snow/Snow100K-L/{{gt,lq}}/")
    print(f"    snow/Snow100K-S/{{gt,lq}}/")
    print("=" * 60)


if __name__ == "__main__":
    main()