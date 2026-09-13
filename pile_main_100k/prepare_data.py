"""Build a deterministic 10k/2k random sample of Pile Uncopyrighted.

Rows are drawn after a large-buffer streaming shuffle.  Output is first written
to a staging directory so an interrupted download never damages Pile_large.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="monology/pile-uncopyrighted")
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--val-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--shuffle-buffer", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("data/Pile_large_10k_staging"))
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--target", type=Path, default=Path("data/Pile_large"))
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    needed = args.train_size + args.val_size
    stream = load_dataset(args.dataset, split="train", streaming=True)
    stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    rows: list[dict] = []
    seen: set[str] = set()
    for scanned, row in enumerate(stream, start=1):
        text = str(row.get("text", "")).strip()
        meta = row.get("meta") or {}
        domain = meta.get("pile_set_name", "Unknown") if isinstance(meta, dict) else "Unknown"
        if len(text) < 32:
            continue
        digest = stable_id(text)
        if digest in seen:
            continue
        seen.add(digest)
        rows.append({"text": text, "meta": {"pile_set_name": domain}, "sha256": digest})
        if len(rows) % 1000 == 0:
            print(f"accepted={len(rows):,}/{needed:,} scanned={scanned:,}", flush=True)
        if len(rows) == needed:
            break
    if len(rows) != needed:
        raise RuntimeError(f"Only obtained {len(rows)} unique rows; expected {needed}.")

    random.Random(args.seed).shuffle(rows)
    train_rows = rows[: args.train_size]
    val_rows = rows[args.train_size :]
    if args.output.exists():
        shutil.rmtree(args.output)
    subset_dir = args.output / "subsets"
    subset_dir.mkdir(parents=True)
    write_jsonl(args.output / "train.jsonl", train_rows)
    write_jsonl(args.output / "val.jsonl", val_rows)
    by_domain: dict[str, list[dict]] = {}
    for row in rows:
        by_domain.setdefault(row["meta"]["pile_set_name"], []).append(row)
    for index, (domain, domain_rows) in enumerate(sorted(by_domain.items())):
        write_jsonl(subset_dir / f"{index:02d}.jsonl", domain_rows)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    split_stats = {}
    for name, split_rows in (("train", train_rows), ("val", val_rows)):
        token_count = 0
        domains = Counter()
        for row in split_rows:
            token_count += len(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
            domains[row["meta"]["pile_set_name"]] += 1
        split_stats[name] = {
            "corpora": len(split_rows),
            "gpt2_tokens": token_count,
            "bytes": sum(len(row["text"].encode("utf-8")) for row in split_rows),
            "domains": dict(sorted(domains.items())),
        }
    manifest = {
        "dataset": args.dataset,
        "sampling": "streaming shuffle with fixed-size random buffer, then deterministic split",
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "splits": split_stats,
        "unique_sha256": len(seen),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)

    if args.install:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = args.target.with_name(f"{args.target.name}_backup_{stamp}")
        if args.target.exists():
            args.target.rename(backup)
            print(f"backup={backup}", flush=True)
        args.output.rename(args.target)
        print(f"installed={args.target}", flush=True)


if __name__ == "__main__":
    main()
