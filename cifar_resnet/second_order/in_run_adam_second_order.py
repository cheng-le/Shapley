import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))
from cifar_resnet_common import add_cifar_args, cifar_config_from_args, compute_adam_second_order, save_and_finish_single, summarize_scores


def main():
    parser = argparse.ArgumentParser(description="CIFAR10+ResNet18 二阶 Adam-aware In-Run Data Shapley。")
    add_cifar_args(parser, "adam")
    parser.set_defaults(output="cifar_resnet/second_order/cifar10_resnet18_adam_second_order.pt", plot_output="cifar_resnet/second_order/cifar10_resnet18_adam_second_order.png")
    config = cifar_config_from_args(parser.parse_args())
    base_train, val_dataset, scores, snapshot, timing = compute_adam_second_order(config)
    out, plot_out = save_and_finish_single("CIFAR10 ResNet18 Adam-aware second-order In-Run Data Shapley", config, "adam", scores, snapshot, timing, base_train, val_dataset)
    summarize_scores("CIFAR10 ResNet18 Adam 二阶分数", scores)
    print(f"分数已保存到: {out}")
    print(f"评估图已保存到: {plot_out}")

if __name__ == "__main__":
    main()
