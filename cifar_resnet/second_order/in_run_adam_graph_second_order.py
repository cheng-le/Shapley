import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))
from cifar_resnet_common import (
    add_cifar_args,
    cifar_config_from_args,
    compute_adam_graph_second_order,
    compute_adam_second_order,
    compute_sgd_second_order,
    finish_graph_comparison_experiment,
    save_scores,
    summarize_scores,
)


def main():
    parser = argparse.ArgumentParser(description="CIFAR10+ResNet18 二阶 Adam-aware + Graph 对比实验。")
    add_cifar_args(parser, "adam_graph")
    parser.set_defaults(output="cifar_resnet/second_order/cifar10_resnet18_adam_graph_second_order.pt", plot_output="cifar_resnet/second_order/cifar10_resnet18_adam_graph_second_order.png")
    config = cifar_config_from_args(parser.parse_args())
    base_train, val_dataset, raw_scores, topo_scores, snapshot, timing = compute_adam_graph_second_order(config)
    _, _, sgd_scores, sgd_snapshot, sgd_timing = compute_sgd_second_order(config)
    _, _, adam_scores, adam_snapshot, adam_timing = compute_adam_second_order(config)
    out = save_scores(config.output, topo_scores, {"raw_scores": raw_scores, "sgd_scores": sgd_scores, "adam_scores": adam_scores})
    plot_out = finish_graph_comparison_experiment(
        "CIFAR10 ResNet18 Graph Adam-aware second-order comparison",
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
    summarize_scores("CIFAR10 ResNet18 Adam 图增强二阶分数", topo_scores)
    print(f"分数已保存到: {out}")
    print(f"对比评估图已保存到: {plot_out}")

if __name__ == "__main__":
    main()
