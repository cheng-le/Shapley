"""Tokenize Pile once into memory-mapped arrays for isolated retraining cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pile_main_100k.common import dataset_fingerprint, load_rows, tokenizer


def write_split(rows, tok, output: Path, split: str, length: int, batch_size: int):
    ids = np.lib.format.open_memmap(output / f"{split}_ids.npy", mode="w+", dtype=np.int32, shape=(len(rows), length))
    masks = np.lib.format.open_memmap(output / f"{split}_mask.npy", mode="w+", dtype=np.uint8, shape=(len(rows), length))
    for start in range(0, len(rows), batch_size):
        encoded = tok([row["text"] for row in rows[start:start + batch_size]], padding="max_length", truncation=True,
                      max_length=length, return_tensors="np")
        ids[start:start + len(encoded["input_ids"])] = encoded["input_ids"].astype(np.int32)
        masks[start:start + len(encoded["attention_mask"])] = encoded["attention_mask"].astype(np.uint8)
        if start % (batch_size * 20) == 0:
            print(f"tokenize {split}={min(start + batch_size, len(rows)):,}/{len(rows):,}", flush=True)
    ids.flush(); masks.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("pile_main_100k/token_cache"))
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    train, val = load_rows("train"), load_rows("val")
    args.output.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    write_split(train, tok, args.output, "train", args.length, args.batch_size)
    write_split(val, tok, args.output, "val", args.length, args.batch_size)
    metadata = {"fingerprint": dataset_fingerprint(train), "train_corpora": len(train), "val_corpora": len(val),
                "sequence_length": args.length, "tokenizer": "gpt2"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
