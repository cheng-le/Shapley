"""Process-isolated GPT-2 selection evaluation using token memmaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from pile_main_100k.common import METHODS, diversity_select, example_losses, model, seed_all


class MemmapDataset(Dataset):
    def __init__(self, folder: Path, split: str):
        self.ids = np.load(folder / f"{split}_ids.npy", mmap_mode="c")
        self.mask = np.load(folder / f"{split}_mask.npy", mmap_mode="c")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return torch.from_numpy(self.ids[index].astype(np.int64, copy=False)), torch.from_numpy(self.mask[index].astype(np.int64, copy=False))


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval(); loss_sum = examples = correct = tokens = 0
    for ids, mask in loader:
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        labels, valid = ids[:, 1:], mask[:, 1:].bool()
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none").view_as(labels)
        loss_sum += float(((losses * valid).sum(1) / valid.sum(1).clamp_min(1)).sum())
        examples += len(ids); correct += int(((logits.argmax(-1) == labels) & valid).sum()); tokens += int(valid.sum())
    return loss_sum / examples, correct / tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--budget", type=int, choices=[1, 2, 5, 10, 15, 20, 40, 60, 80], required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--protocol-split", type=Path)
    parser.add_argument("--eval-part", choices=["valuation", "evaluation"], default="evaluation")
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--graph-diversity", action="store_true")
    parser.add_argument("--diversity-lambda", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_all(args.seed); device = torch.device(f"cuda:{args.gpu}")
    train, val = MemmapDataset(args.token_cache, "train"), MemmapDataset(args.token_cache, "val")
    score_cache = np.load(args.cache, allow_pickle=False)
    metadata = json.loads((args.token_cache / "metadata.json").read_text())
    if str(score_cache["fingerprint"]) != metadata["fingerprint"]:
        raise RuntimeError("Attribution and token-cache fingerprints differ.")
    count = round(len(train) * args.budget / 100)
    scores = score_cache[f"score::{args.method}"]
    selection_details = {"strategy": "top_k"}
    if args.graph_diversity and args.method.startswith("Graph "):
        selected, diagnostics = diversity_select(
            scores,
            score_cache["graph_neighbors"],
            score_cache["graph_weights"],
            count,
            args.diversity_lambda,
        )
        selection_details = {"strategy": "symmetric_graph_greedy", **diagnostics}
    else:
        selected = np.argpartition(scores, -count)[-count:]
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(Subset(train, selected.tolist()), batch_size=args.batch_size, shuffle=True, generator=generator,
                        pin_memory=True, num_workers=2, persistent_workers=True)
    evaluation = val
    if args.protocol_split:
        split = json.loads(args.protocol_split.read_text())
        evaluation = Subset(val, split[f"{args.eval_part}_indices"])
    val_loader = DataLoader(evaluation, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    net = model(device, train_full=True)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1, foreach=False
    )
    total_steps = args.epochs * len(loader)
    scheduler = None
    if args.lr_schedule == "cosine":
        warmup_steps = max(1, round(total_steps * args.warmup_ratio))
        def lr_factor(step: int) -> float:
            if step < warmup_steps:
                return max((step + 1) / warmup_steps, args.min_lr_ratio)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    net.train()
    for _epoch in range(args.epochs):
        for ids, mask in loader:
            batch = {"input_ids": ids.to(device, non_blocking=True), "attention_mask": mask.to(device, non_blocking=True)}
            optimizer.zero_grad(set_to_none=True); loss = example_losses(net, batch).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
            if scheduler is not None:
                scheduler.step()
    val_loss, accuracy = evaluate(net, val_loader, device)
    payload = {"budget": args.budget, "selected_corpora": count, "epochs": args.epochs, "primary_metric": "validation_token_accuracy",
               "method": args.method, "seed": args.seed, "accuracy": accuracy, "val_loss": val_loss,
               "score_cache": str(args.cache), "token_cache": str(args.token_cache),
               "selection": selection_details, "protocol_split": str(args.protocol_split) if args.protocol_split else None,
               "eval_part": args.eval_part,
               "lr_schedule": args.lr_schedule, "warmup_ratio": args.warmup_ratio,
               "min_lr_ratio": args.min_lr_ratio, "optimizer_steps": total_steps,
               "selected_indices_sha256": hashlib.sha256(np.sort(selected).tobytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"budget={args.budget} method={args.method} seed={args.seed} val_accuracy={accuracy:.6f} val_loss={val_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
