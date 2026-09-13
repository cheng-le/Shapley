"""Evaluate a saved selection using the main GPT-2 training protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pile_main_100k.common import example_losses, model, seed_all
from pile_main_100k.selection_accuracy_cached_runner import MemmapDataset, evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cell = json.loads(args.cell.read_text())
    selected = np.load(cell['selected_indices'], allow_pickle=False)
    if hashlib.sha256(np.sort(selected).tobytes()).hexdigest() != cell['indices_sha256']:
        raise ValueError('Selected-index hash mismatch')
    seed_all(cell['seed'])
    device = torch.device(f'cuda:{args.gpu}')
    train = MemmapDataset(Path('pile_main_100k/token_cache'), 'train')
    val = MemmapDataset(Path('pile_main_100k/token_cache'), 'val')
    split = json.loads(Path('pile_main_100k/protocol_v2_split.json').read_text())
    loader = DataLoader(Subset(train, selected.tolist()), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(cell['seed']),
                        pin_memory=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(Subset(val, split['evaluation_indices']), batch_size=16,
                           shuffle=False, pin_memory=True, num_workers=2)
    net = model(device, train_full=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=6e-5, betas=(0.9, 0.95),
                                 weight_decay=0.1, foreach=False)
    net.train()
    for ids, mask in loader:
        batch = {'input_ids': ids.to(device), 'attention_mask': mask.to(device)}
        optimizer.zero_grad(set_to_none=True)
        loss = example_losses(net, batch).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
    loss, accuracy = evaluate(net, val_loader, device)
    result = {**cell, 'val_loss': loss, 'accuracy': accuracy, 'epochs': 1,
              'eval_part': 'evaluation', 'lr_schedule': 'constant', 'lr': 6e-5,
              'batch_size': 16, 'optimizer_steps': len(loader)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
