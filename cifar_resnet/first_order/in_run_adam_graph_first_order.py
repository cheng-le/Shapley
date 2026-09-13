import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))
from cifar_resnet_common import add_cifar_args, cifar_config_from_args, compute_adam_graph_first_order, finish_experiment, save_scores, summarize_scores


def main():
    parser = argparse.ArgumentParser(description="CIFAR10+ResNet18 一阶 Adam-aware + Graph In-Run Data Shapley。")
    add_cifar_args(parser, "adam_graph")
    parser.set_defaults(output="cifar_resnet/first_order/cifar10_resnet18_adam_graph_first_order.pt", plot_output="cifar_resnet/first_order/cifar10_resnet18_adam_graph_first_order.png")
    config = cifar_config_from_args(parser.parse_args())
    base_train, val_dataset, raw_scores, topo_scores, snapshot, timing = compute_adam_graph_first_order(config)
    out = save_scores(config.output, topo_scores, {"raw_scores": raw_scores})
    plot_out = finish_experiment("CIFAR10 ResNet18 Adam-aware Graph first-order In-Run Data Shapley", config, "adam", base_train, val_dataset, topo_scores, snapshot, timing)
    summarize_scores("CIFAR10 ResNet18 Adam 图增强一阶分数", topo_scores)
    print(f"分数已保存到: {out}")
    print(f"评估图已保存到: {plot_out}")

if __name__ == "__main__":
    main()
