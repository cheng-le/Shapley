"""Aggregate low-ratio runs using validation loss (lower is better)."""
import json
from pathlib import Path
import numpy as np
from pile_main_100k.common import METHODS

BUDGETS = (1, 2, 5, 10)

def main():
    outdir = Path("pile_main_100k/low_ratio_validation_loss")
    results = {}
    for index, method in enumerate(METHODS):
        results[method] = {}
        for budget in BUDGETS:
            cells = [json.loads((outdir / "raw" / f"method{index}_seed{seed}_budget{budget}.json").read_text()) for seed in range(3)]
            losses = [cell["val_loss"] for cell in cells]
            accuracies = [cell["accuracy"] for cell in cells]
            results[method][str(budget)] = {
                "loss_runs": losses, "loss_mean": float(np.mean(losses)), "loss_std": float(np.std(losses)),
                "accuracy_runs": accuracies, "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies)),
            }
    payload = {
        "primary_metric": "validation_loss", "direction": "lower_is_better",
        "budgets": list(BUDGETS), "seeds": [0, 1, 2],
        "protocol_split": "../protocol_v2_split.json (evaluation indices only)",
        "score_cache_boundary": "frozen legacy scores; no attribution recomputation",
        "results": results,
    }
    (outdir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    header = "| Method | " + " | ".join(f"{budget}% Val. Loss ↓" for budget in BUDGETS) + " |"
    lines = [header, "|---|" + "---:|" * len(BUDGETS)]
    best = [min(range(len(METHODS)), key=lambda i: results[METHODS[i]][str(b)]["loss_mean"]) for b in BUDGETS]
    second = [sorted(range(len(METHODS)), key=lambda i: results[METHODS[i]][str(b)]["loss_mean"])[1] for b in BUDGETS]
    for index, method in enumerate(METHODS):
        cells = []
        for column, budget in enumerate(BUDGETS):
            value = results[method][str(budget)]
            text = f"{value['loss_mean']:.4f} ({value['loss_std']:.4f})"
            if index == best[column]: text = f"**{text}**"
            elif index == second[column]: text = f"<u>{text}</u>"
            cells.append(text)
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    (outdir / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
