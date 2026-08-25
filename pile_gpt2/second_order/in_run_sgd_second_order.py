import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))
from pile_gpt2_common import add_pile_args, compute_sgd_second_order, pile_config_from_args, save_and_finish_single, summarize_scores


def main():
    parser = argparse.ArgumentParser(description="Pile+GPT2 二阶 SGD In-Run Data Shapley。")
    add_pile_args(parser, "sgd")
    parser.set_defaults(output="pile_gpt2/second_order/pile_gpt2_sgd_second_order.pt", plot_output="pile_gpt2/second_order/pile_gpt2_sgd_second_order.png")
    config = pile_config_from_args(parser.parse_args())
    base_train, val_dataset, scores, snapshot, timing = compute_sgd_second_order(config)
    out, plot_out = save_and_finish_single("Pile GPT2 SGD second-order In-Run Data Shapley", config, "sgd", scores, snapshot, timing, base_train, val_dataset)
    summarize_scores("Pile GPT2 SGD 二阶分数", scores)
    print(f"分数已保存到: {out}")
    print(f"评估图已保存到: {plot_out}")


if __name__ == "__main__":
    main()
