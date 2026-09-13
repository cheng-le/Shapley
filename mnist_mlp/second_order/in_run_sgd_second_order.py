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
    add_dicts,
    compute_single_grad,
    compute_val_grad_and_loss,
    config_from_args,
    dot_dict,
    finish_experiment,
    hvp_from_val_grad,
    make_experiment_loaders,
    make_model,
    sample_validation_batch,
    save_scores,
    scale_dict,
    set_seed,
    summarize_scores,
)


def compute_scores(config):
    """
    二阶 SGD In-Run Data Shapley。
    d_i = -(lr / B) g_i，分数为 g_val^T d_i + 1/2 d_i^T H_val sum_j d_j。
    """
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_experiment_loaders(config)
    model = make_model(config.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot = None
    score_start = time.perf_counter()
    raw_score_time = 0.0
    second_order_time = 0.0

    for _ in range(config.epochs_score):
        model.train()
        for images, labels, indices in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _, val_grad = compute_val_grad_and_loss(model, loss_fn, val_imgs, val_labels, create_graph=True)
            batch_size = images.size(0)
            step_scale = -optimizer.param_groups[0]["lr"] / batch_size
            step_vectors = []
            first_terms = []
            raw_start = time.perf_counter()
            for i in range(batch_size):
                grad_i = compute_single_grad(model, loss_fn, images[i], labels[i])
                step_i = scale_dict(grad_i, step_scale)
                step_vectors.append(step_i)
                first_terms.append(dot_dict(val_grad, step_i))
            raw_score_time += time.perf_counter() - raw_start

            # 二阶项只包含 HVP 以及 1/2 d_i^T H_val sum_j d_j 的计算，便于与图距离代理单独比较。
            second_start = time.perf_counter()
            total_step = add_dicts(step_vectors)
            hvp_total = hvp_from_val_grad(val_grad, total_step, model)
            local_scores = []
            for i, step_i in enumerate(step_vectors):
                score_i = first_terms[i] + 0.5 * dot_dict(step_i, hvp_total)
                local_scores.append(score_i)
                scores[int(indices[i])] += score_i
            second_order_time += time.perf_counter() - second_start

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

    score_time = time.perf_counter() - score_start
    timing = {
        "Score time": score_time,
        "Raw first-order time": raw_score_time,
        "HVP second-order time": second_order_time,
        "Second-order component time": second_order_time,
        "Train update time": max(0.0, score_time - raw_score_time - second_order_time),
    }
    return base_train, val_dataset, scores, snapshot, timing


def main():
    parser = argparse.ArgumentParser(description="二阶 SGD In-Run Data Shapley：保存分数和评估总图。")
    add_common_args(parser, "sgd")
    parser.set_defaults(output="mnist_mlp/second_order/mnist_sgd_second_order.pt", plot_output="mnist_mlp/second_order/mnist_sgd_second_order.png")
    config = config_from_args(parser.parse_args())
    base_train, val_dataset, scores, snapshot, timing = compute_scores(config)
    out = save_scores(config.output, scores)
    plot_out = finish_experiment("SGD second-order In-Run Data Shapley", config, "sgd", base_train, val_dataset, scores, snapshot, timing)
    summarize_scores("SGD 二阶分数", scores)
    print(f"分数已保存到: {out}")
    print(f"评估图已保存到: {plot_out}")


if __name__ == "__main__":
    main()
