import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))
from pile_gpt2_common import (
    add_pile_args,
    compute_adam_first_order,
    compute_adam_graph_first_order,
    compute_sgd_first_order,
    finish_graph_comparison_experiment,
    pile_config_from_args,
    save_scores,
    summarize_scores,
)


def main():
    parser = argparse.ArgumentParser(description="Pile+GPT2 一阶 Adam-aware Graph 对比实验。")
    add_pile_args(parser, "adam_graph")
    parser.set_defaults(output="pile_gpt2/first_order/pile_gpt2_adam_graph_first_order.pt", plot_output="pile_gpt2/first_order/pile_gpt2_adam_graph_first_order.png")
    config = pile_config_from_args(parser.parse_args())
    base_train, val_dataset, raw_scores, topo_scores, snapshot, timing = compute_adam_graph_first_order(config)
    _, _, sgd_scores, sgd_snapshot, sgd_timing = compute_sgd_first_order(config)
    _, _, adam_scores, adam_snapshot, adam_timing = compute_adam_first_order(config)
    out = save_scores(config.output, topo_scores, {"raw_scores": raw_scores, "sgd_scores": sgd_scores, "adam_scores": adam_scores})
    plot_out = finish_graph_comparison_experiment(
        "Pile GPT2 Graph Adam-aware first-order comparison",
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
    summarize_scores("Pile GPT2 Adam Graph 一阶分数", topo_scores)
    print(f"分数已保存到: {out}")
    print(f"对比评估图已保存到: {plot_out}")


if __name__ == "__main__":
    main()
