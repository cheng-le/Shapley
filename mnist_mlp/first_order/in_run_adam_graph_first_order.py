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
    adam_update_direction_from_grad,
    build_knn_graph_from_raw_pixels,
    clone_adam_state,
    compute_single_grad,
    compute_val_grad_and_loss,
    config_from_args,
    dot_dict,
    finish_graph_comparison_experiment,
    make_experiment_loaders,
    make_model,
    sample_validation_batch,
    save_scores,
    set_seed,
    summarize_scores,
    topology_regularize_scores,
)
from mnist_mlp.first_order.in_run_sgd_first_order import compute_scores as compute_sgd_scores  # noqa: E402
from mnist_mlp.first_order.in_run_adam_first_order import compute_scores as compute_adam_scores  # noqa: E402


def compute_scores(config):
    """一阶 Adam-aware + 图建模：raw 分数按 Adam 论文，图增强只作为自己的稳定化模块。"""
    set_seed(config.seed)
    base_train, val_dataset, train_loader, val_loader = make_experiment_loaders(config)
    graph_start = time.perf_counter()
    graph = build_knn_graph_from_raw_pixels(base_train, config.topo_k, config.topo_pca_dim, config.seed)
    graph_time = time.perf_counter() - graph_start
    model = make_model(config.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps)
    loss_fn = nn.CrossEntropyLoss()
    raw_scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    topo_scores = torch.zeros(len(train_loader.dataset), dtype=torch.float32)
    val_imgs, val_labels = sample_validation_batch(val_loader, config.val_samples)
    snapshot = None
    score_start = time.perf_counter()

    for _ in range(config.epochs_score):
        model.train()
        for images, labels, indices in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            adam_state = clone_adam_state(model, optimizer)
            _, val_grad = compute_val_grad_and_loss(model, loss_fn, val_imgs, val_labels, create_graph=False)
            lr_now = optimizer.param_groups[0]["lr"]
            raw_batch = []
            for i in range(images.size(0)):
                grad_i = compute_single_grad(model, loss_fn, images[i], labels[i])
                update_i = adam_update_direction_from_grad(grad_i, adam_state, config.beta1, config.beta2, config.eps)
                raw_batch.append(-lr_now * dot_dict(val_grad, update_i))

            batch_idx = indices.numpy()
            raw_np = np.asarray(raw_batch, dtype=np.float32)
            topo_np = topology_regularize_scores(batch_idx, raw_np, raw_scores.numpy(), graph, config.topo_alpha, config.topo_lambda)
            raw_scores[batch_idx] += torch.from_numpy(raw_np)
            topo_scores[batch_idx] += torch.from_numpy(topo_np.astype(np.float32))

            if snapshot is None:
                snapshot = LocalStepSnapshot(
                    state_dict=state_before,
                    adam_state={k: {"exp_avg": v["exp_avg"].detach().cpu().clone(), "exp_avg_sq": v["exp_avg_sq"].detach().cpu().clone(), "step": int(v["step"])} for k, v in adam_state.items()},
                    images=images.detach().cpu().clone(),
                    labels=labels.detach().cpu().clone(),
                    indices=indices.detach().cpu().clone(),
                    val_imgs=val_imgs.detach().cpu().clone(),
                    val_labels=val_labels.detach().cpu().clone(),
                    approx_scores=np.asarray(raw_batch, dtype=np.float64),
                    topo_approx_scores=np.asarray(topo_np, dtype=np.float64),
                    lr=lr_now,
                    batch_size_ref=images.size(0),
                    optimizer_name="adam",
                )

            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

    timing = {"Graph build time": graph_time, "Score time": time.perf_counter() - score_start, "Total graph+score time": graph_time + time.perf_counter() - score_start}
    return base_train, val_dataset, raw_scores, topo_scores, snapshot, timing


def main():
    parser = argparse.ArgumentParser(description="一阶 Adam-aware + 图增强 In-Run Data Shapley：保存分数和评估总图。")
    add_common_args(parser, "adam_graph")
    parser.set_defaults(output="mnist_mlp/first_order/mnist_adam_graph_first_order.pt", plot_output="mnist_mlp/first_order/mnist_adam_graph_first_order.png")
    config = config_from_args(parser.parse_args())
    base_train, val_dataset, raw_scores, topo_scores, snapshot, timing = compute_scores(config)
    _, _, sgd_scores, sgd_snapshot, sgd_timing = compute_sgd_scores(config)
    _, _, adam_scores, adam_snapshot, adam_timing = compute_adam_scores(config)
    out = save_scores(
        config.output,
        topo_scores,
        {"raw_scores": raw_scores, "sgd_scores": sgd_scores, "adam_scores": adam_scores},
    )
    plot_out = finish_graph_comparison_experiment(
        "Graph Adam-aware first-order comparison",
        config,
        base_train,
        val_dataset,
        [
            {"name": "SGD", "scores": sgd_scores, "snapshot": sgd_snapshot, "optimizer_name": "sgd", "timing": sgd_timing},
            {"name": "Adam-aware", "scores": adam_scores, "snapshot": adam_snapshot, "optimizer_name": "adam", "timing": adam_timing},
            {"name": "Graph", "scores": topo_scores, "snapshot": snapshot, "optimizer_name": "adam", "timing": timing, "use_topo_fidelity": True},
        ],
        config.plot_output,
    )
    summarize_scores("Adam 图增强一阶分数", topo_scores)
    print(f"分数已保存到: {out}")
    print(f"对比评估图已保存到: {plot_out}")


if __name__ == "__main__":
    main()
