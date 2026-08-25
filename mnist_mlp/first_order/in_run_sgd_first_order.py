import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(str(Path(__file__).resolve().parents[2]))
from mnist_mlp.mnist_mlp_common import (  # noqa: E402
    DEVICE,
    LocalStepSnapshot,
    add_common_args,
    compute_single_grad,
    compute_val_grad_and_loss,
    config_from_args,
    dot_dict,
    finish_experiment,
    make_experiment_loaders,
    make_model,
    sample_validation_batch,
    save_scores,
    set_seed,
    summarize_scores,
)


def compute_scores(config):
    """一阶 SGD In-Run Data Shapley: phi_i = -(lr / B) * <g_val, g_i>。"""
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_experiment_loaders(config)
    model = make_model(config.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot = None
    score_start = time.perf_counter()

    for _ in range(config.epochs_score):
        model.train()
        for images, labels, indices in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _, val_grad = compute_val_grad_and_loss(model, loss_fn, val_imgs, val_labels, create_graph=False)
            batch_size = images.size(0)
            lr_scale = -optimizer.param_groups[0]["lr"] / batch_size
            local_scores = []

            for i in range(batch_size):
                grad_i = compute_single_grad(model, loss_fn, images[i], labels[i])
                score_i = lr_scale * dot_dict(val_grad, grad_i)
                local_scores.append(score_i)
                scores[int(indices[i])] += score_i

            if snapshot is None:
                snapshot = LocalStepSnapshot(
                    state_dict=state_before,
                    images=images.detach().cpu().clone(),
                    labels=labels.detach().cpu().clone(),
                    indices=indices.detach().cpu().clone(),
                    val_imgs=val_imgs.detach().cpu().clone(),
                    val_labels=val_labels.detach().cpu().clone(),
                    approx_scores=np.asarray(local_scores, dtype=np.float64),
                    lr=optimizer.param_groups[0]["lr"],
                    batch_size_ref=batch_size,
                    optimizer_name="sgd",
                )

            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

    timing = {"Score time": time.perf_counter() - score_start}
    return base_train, val_dataset, scores, snapshot, timing


def main():
    parser = argparse.ArgumentParser(description="一阶 SGD In-Run Data Shapley：保存分数和评估总图。")
    add_common_args(parser, "sgd")
    parser.set_defaults(output="mnist_mlp/first_order/mnist_sgd_first_order.pt", plot_output="mnist_mlp/first_order/mnist_sgd_first_order.png")
    config = config_from_args(parser.parse_args())
    base_train, val_dataset, scores, snapshot, timing = compute_scores(config)
    out = save_scores(config.output, scores)
    plot_out = finish_experiment("SGD first-order In-Run Data Shapley", config, "sgd", base_train, val_dataset, scores, snapshot, timing)
    summarize_scores("SGD 一阶分数", scores)
    print(f"分数已保存到: {out}")
    print(f"评估图已保存到: {plot_out}")


if __name__ == "__main__":
    main()
