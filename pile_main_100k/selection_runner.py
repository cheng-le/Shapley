"""Retrain GPT-2 on a high-value subset and report held-out validation loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from pile_main_100k.common import METHODS, example_losses, load_rows, model, seed_all, tokenizer


class CorpusDataset(Dataset):
    def __init__(self, rows, tok, length=128):
        batch = tok([row["text"] for row in rows], padding="max_length", truncation=True, max_length=length, return_tensors="pt")
        self.ids, self.mask = batch["input_ids"], batch["attention_mask"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index], self.mask[index]


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    total, count = 0.0, 0
    for ids, mask in loader:
        losses = example_losses(net, {"input_ids": ids.to(device), "attention_mask": mask.to(device)})
        total += float(losses.sum())
        count += len(losses)
    return total / count


def train_and_evaluate(dataset, val_loader, device, seed, batch_size, lr):
    seed_all(seed)
    net = model(device, train_full=True)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, pin_memory=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    net.train()
    for ids, mask in loader:
        batch = {"input_ids": ids.to(device, non_blocking=True), "attention_mask": mask.to(device, non_blocking=True)}
        optimizer.zero_grad(set_to_none=True)
        loss = example_losses(net, batch).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
    value = evaluate(net, val_loader, device)
    del net, optimizer
    torch.cuda.empty_cache()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--budget", type=int, choices=[20, 40, 60, 80], required=True)
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_rows, val_rows = load_rows("train"), load_rows("val")
    tok, device = tokenizer(), torch.device(f"cuda:{args.gpu}")
    train_dataset = CorpusDataset(train_rows, tok)
    val_loader = DataLoader(CorpusDataset(val_rows, tok), batch_size=args.batch_size, shuffle=False, pin_memory=True)
    cache = np.load(args.cache, allow_pickle=False)
    count = round(len(train_rows) * args.budget / 100)
    results = {}
    for method in args.methods:
        values = cache[f"score::{method}"]
        selected = np.argpartition(values, -count)[-count:]
        subset = Subset(train_dataset, selected.tolist())
        runs = []
        for seed in args.seeds:
            value = train_and_evaluate(subset, val_loader, device, seed, args.batch_size, args.lr)
            runs.append(value)
            print(f"budget={args.budget} method={method} seed={seed} val_loss={value:.6f}", flush=True)
        results[method] = {"runs": runs, "mean": float(np.mean(runs)), "std": float(np.std(runs))}
    payload = {"budget": args.budget, "selected_corpora": count, "epochs": 1, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
