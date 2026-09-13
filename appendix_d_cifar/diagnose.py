"""Describe class imbalance of selected sets without modifying scores."""
import json
from pathlib import Path
import numpy as np
import run

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def main():
    rows = []
    for ds in range(3):
        labels = np.load(HERE / "data" / ("seed%d.npz" % ds))["labels"]
        for ts in range(3):
            folder = OUT / ("data%d_train%d" % (ds, ts))
            for cell in json.loads((folder / "results.json").read_text()):
                with np.load(folder / ("fit_" + cell["unique_fit_id"] + ".npz")) as fit:
                    counts = np.bincount(labels[fit["selected"]], minlength=10)
                    predictions = np.bincount(fit["predictions"], minlength=10)
                    probs = counts[counts > 0] / 200
                    rows.append(dict(data_seed=ds, train_seed=ts, variant=cell["variant"],
                        selected_class_count=int((counts > 0).sum()),
                        largest_selected_class_fraction=float(counts.max() / 200),
                        selected_class_entropy=float(-(probs * np.log(probs)).sum()),
                        dominant_prediction_fraction=float(predictions.max() / 2000),
                        predicted_class_count=int((predictions > 0).sum())))
    run.csv_write(OUT / "class_imbalance_all.csv", rows)
    summary = []
    for name, _ in run.configs():
        group = [r for r in rows if r["variant"] == name]
        row = dict(variant=name)
        for metric in ("selected_class_count", "largest_selected_class_fraction", "selected_class_entropy", "dominant_prediction_fraction", "predicted_class_count"):
            row[metric + "_mean"] = float(np.mean([r[metric] for r in group]))
            row[metric + "_std"] = float(np.std([r[metric] for r in group], ddof=1))
        summary.append(row)
    run.csv_write(OUT / "class_imbalance_summary.csv", summary)
    print("Class-imbalance diagnostics written; no scores or selections changed.")


if __name__ == "__main__":
    main()
