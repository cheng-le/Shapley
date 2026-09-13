"""Export editable long-form source data and provenance for every main-table figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pile_main_100k.common import EXPERIMENT


def main() -> None:
    output = EXPERIMENT / "results"
    with (output / "main_table.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    metric_columns = [column for column in rows[0] if column not in {"Method", "Avg. Rank"}]
    with (output / "plot_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "metric", "value", "higher_is_better"])
        for row in rows:
            for metric in metric_columns:
                writer.writerow([row["Method"], metric, row[metric], metric in {"Noise AUROC", "Domain nRecall@1K"}])
    ranking = json.loads((output / "main_table.json").read_text())
    with (output / "per_metric_ranks.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method", *metric_columns, "Avg. Rank"])
        for row, ranks in zip(rows, ranking["per_metric_ranks"]):
            writer.writerow([row["Method"], *ranks, row["Avg. Rank"]])
    manifest = {
        "table_source": "main_table.csv",
        "plot_source": "plot_data.csv",
        "rank_source": "per_metric_ranks.csv",
        "raw_sources": {
            "noise": "../noise_auroc/results.json and ../cache/noise_seed*.npz",
            "domain": "../domain_recall/results.json",
            "selection": "../selection_{20,40,60,80}/results_gpu{0,1}.json",
            "dataset": "../../data/Pile_large/manifest.json",
        },
        "figure_formats": ["png", "pdf", "svg"],
    }
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
