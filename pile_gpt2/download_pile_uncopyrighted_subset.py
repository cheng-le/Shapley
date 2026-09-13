"""从 HuggingFace monology/pile-uncopyrighted 流式抽样到 data/Pile。

这个脚本不会下载 335GB 全量数据，而是按 pile_set_name 分层抽取一小部分样本，
生成 GPT2 实验可直接读取的本地 jsonl：

    data/Pile/train.jsonl
    data/Pile/val.jsonl
    data/Pile/subsets/*.jsonl

用法示例：

    /home/lcheng/anaconda3/envs/torch_5090/bin/python pile_gpt2/download_pile_uncopyrighted_subset.py \
      --per-subset-train 50 --per-subset-val 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


PILE_UNCOPYRIGHTED_SUBSETS = [
    "Pile-CC",
    "PubMed Central",
    "ArXiv",
    "Github",
    "FreeLaw",
    "StackExchange",
    "USPTO Backgrounds",
    "PubMed Abstracts",
    "Gutenberg (PG-19)",
    "Wikipedia (en)",
    "DM Mathematics",
    "Ubuntu IRC",
    "EuroParl",
    "HackerNews",
    "PhilPapers",
    "NIH ExPorter",
    "Enron Emails",
]


REMOVED_FROM_UNCOPYRIGHTED = [
    "Books3",
    "BookCorpus2",
    "OpenSubtitles",
    "YoutubeSubtitles",
    "OpenWebText2",
]


def slug(name: str) -> str:
    """把子集名转成稳定文件名。"""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="流式抽样 monology/pile-uncopyrighted 到 data/Pile。")
    parser.add_argument("--dataset-name", type=str, default="monology/pile-uncopyrighted")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out-dir", type=Path, default=Path("data/Pile"))
    parser.add_argument("--per-subset-train", type=int, default=50)
    parser.add_argument("--per-subset-val", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=2_000_000, help="最多扫描多少行；网络慢时可调小。")
    parser.add_argument("--retries", type=int, default=8, help="HuggingFace 流式连接中断后的重试次数。")
    parser.add_argument("--retry-sleep", type=float, default=5.0, help="每次重试前等待秒数。")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("请先安装 datasets：python -m pip install datasets") from exc
    if importlib.util.find_spec("zstandard") is None:
        raise RuntimeError(
            "monology/pile-uncopyrighted 的源文件是 .zst 压缩格式，当前 Python 环境缺少 zstandard。\n"
            "请先安装：/home/lcheng/anaconda3/envs/torch_5090/bin/python -m pip install zstandard"
        )

    random.seed(args.seed)
    need = args.per_subset_train + args.per_subset_val
    buckets: Dict[str, List[str]] = defaultdict(list)
    seen: Dict[str, Set[int]] = defaultdict(set)
    target = set(PILE_UNCOPYRIGHTED_SUBSETS)

    scanned = 0
    attempts = 0
    while scanned < args.max_rows and not all(len(buckets[name]) >= need for name in target):
        try:
            ds = load_dataset(args.dataset_name, split=args.split, streaming=True)
            for row in ds:
                scanned += 1
                meta = row.get("meta") or {}
                subset = meta.get("pile_set_name") if isinstance(meta, dict) else None
                text = str(row.get("text", "")).strip()
                if subset in target and text and len(buckets[subset]) < need:
                    text_key = hash(text)
                    if text_key not in seen[subset]:
                        seen[subset].add(text_key)
                        buckets[subset].append(text)
                if scanned % 10_000 == 0:
                    filled = sum(1 for name in target if len(buckets[name]) >= need)
                    print(f"已扫描 {scanned} 行，已满足 {filled}/{len(target)} 个子集。", flush=True)
                if all(len(buckets[name]) >= need for name in target):
                    break
                if scanned >= args.max_rows:
                    break
            break
        except Exception as exc:
            attempts += 1
            filled = sum(1 for name in target if len(buckets[name]) >= need)
            if attempts > args.retries:
                print(f"流式读取失败且已超过重试次数；保留已抽到的 {filled}/{len(target)} 个子集。错误: {exc}", flush=True)
                break
            print(
                f"流式读取中断，第 {attempts}/{args.retries} 次重试；"
                f"已扫描 {scanned} 行，已满足 {filled}/{len(target)} 个子集。错误: {exc}",
                flush=True,
            )
            time.sleep(args.retry_sleep)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subset_dir = args.out_dir / "subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)

    train_rows = []
    val_rows = []
    summary = []
    for subset in PILE_UNCOPYRIGHTED_SUBSETS:
        rows = buckets.get(subset, [])
        random.shuffle(rows)
        train = rows[: args.per_subset_train]
        val = rows[args.per_subset_train : args.per_subset_train + args.per_subset_val]
        subset_path = subset_dir / f"{slug(subset)}.jsonl"
        with subset_path.open("w", encoding="utf-8") as f:
            for text in rows:
                f.write(json.dumps({"text": text, "meta": {"pile_set_name": subset}}, ensure_ascii=False) + "\n")
        train_rows.extend({"text": text, "meta": {"pile_set_name": subset}} for text in train)
        val_rows.extend({"text": text, "meta": {"pile_set_name": subset}} for text in val)
        summary.append({"subset": subset, "available": len(rows), "train": len(train), "val": len(val), "file": str(subset_path)})

    random.shuffle(train_rows)
    random.shuffle(val_rows)
    with (args.out_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "val.jsonl").open("w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "pile_uncopyrighted_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset_name,
                "scanned_rows": scanned,
                "removed_from_uncopyrighted": REMOVED_FROM_UNCOPYRIGHTED,
                "subsets": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"写入训练样本: {len(train_rows)} -> {args.out_dir / 'train.jsonl'}")
    print(f"写入验证样本: {len(val_rows)} -> {args.out_dir / 'val.jsonl'}")
    missing = [item["subset"] for item in summary if item["available"] == 0]
    if missing:
        print("未抽到的子集:", ", ".join(missing))


if __name__ == "__main__":
    main()
