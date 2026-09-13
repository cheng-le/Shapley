"""Independent Appendix-D input, prediction, trajectory and seed audit."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import run

HERE = Path(__file__).resolve().parent
OUT, DATA = HERE / "results", HERE / "data"


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest() == protocol["script_sha256"]
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, digest in manifest["input_hashes"].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((DATA / "resnet18-f37072fd.pth").read_bytes()).hexdigest() == manifest["checkpoint_sha256"]
    validation = set(manifest["validation_train_indices"])
    assert len(validation) == 1000 and len(set(manifest["final_test_indices"])) == 2000
    all_rows, references, unique_fits = [], [], 0
    for ds in range(3):
        assert not (validation & set(manifest["training_indices"][ds]))
        source = np.load(DATA / ("seed%d.npz" % ds))
        assert int(source["noise"].sum()) == 100
        np.testing.assert_array_equal(source["source_indices"], manifest["training_indices"][ds])
        test_y = np.load(DATA / "evaluation.npz")["test_y"]
        for ts in range(3):
            folder = OUT / ("data%d_train%d" % (ds, ts))
            rows = json.loads((folder / "results.json").read_text())
            assert len(rows) == len(run.configs()) == 44
            assert {r["variant"] for r in rows} == {name for name, _ in run.configs()}
            assert all(r["data_seed"] == ds and r["train_seed"] == ts for r in rows)
            with np.load(folder / "trajectory.npz") as trajectory, np.load(folder / "scores.npz") as scores:
                assert len(trajectory["ids"]) == 5000 and len(trajectory["ptr"]) == 81
                np.testing.assert_array_equal(np.bincount(trajectory["ids"], minlength=1000), np.full(1000, 5))
                for a, b in zip(trajectory["ptr"][:-1], trajectory["ptr"][1:]):
                    assert len(np.unique(trajectory["ids"][a:b])) == b - a <= 64
                raw = np.zeros(1000, dtype=np.float32)
                np.add.at(raw, trajectory["ids"], trajectory["raw"])
                np.testing.assert_array_equal(raw, scores["k10_a0_l0"])
                for gs in (1, 2):
                    np.testing.assert_array_equal(scores["full"], scores["exact_seed%d" % gs])
                for name in scores.files:
                    assert np.isfinite(scores[name]).all()
            ids = set()
            for row in rows:
                signature = row["unique_fit_id"]
                with np.load(folder / ("fit_" + signature + ".npz")) as fit:
                    selected = fit["selected"]
                    assert len(np.unique(selected)) == 200
                    assert hashlib.sha256(np.sort(selected).astype(np.int32).tobytes()).hexdigest()[:16] == signature
                    assert abs(float((fit["predictions"] == test_y).mean()) - row["test_accuracy"]) < 1e-12
                    assert abs(float(fit["losses"].mean()) - row["test_loss"]) < 1e-7
                    assert abs(float(source["noise"][selected].mean()) - row["selected_noise_fraction"]) < 1e-12
                    assert int(len(np.unique(source["labels"][selected]))) == row["selected_class_count"]
                    assert bool(row["utility_reused"]) == (signature in ids)
                    ids.add(signature)
            unique_fits += len(ids)
            by_name = {r["variant"]: r for r in rows}
            for gs in (1, 2):
                assert by_name["full"]["unique_fit_id"] == by_name["exact_seed%d" % gs]["unique_fit_id"]
            reference = json.loads((folder / "full_data_reference.json").read_text())
            with np.load(folder / "full_data_predictions.npz") as data:
                assert abs(float((data["predictions"] == test_y).mean()) - reference["test_accuracy"]) < 1e-12
            references.append(dict(data_seed=ds, train_seed=ts, **reference))
            all_rows.extend(rows)
        source.close()
    run.csv_write(OUT / "full_data_reference.csv", references)
    pairs = []
    for ds, ts in itertools.product(range(3), repeat=2):
        rows = [r for r in all_rows if r["data_seed"] == ds and r["train_seed"] == ts]
        base = next(r for r in rows if r["variant"] == "full")
        for row in rows:
            pairs.append(dict(data_seed=ds, train_seed=ts, variant=row["variant"],
                accuracy_delta_vs_full=row["test_accuracy"] - base["test_accuracy"],
                noise_auroc_delta_vs_full=row["noise_auroc"] - base["noise_auroc"]))
    run.csv_write(OUT / "paired_effects.csv", pairs)
    factorial = [r for r in all_rows if r["variant"] in ("ann8", "ann8_seed1", "ann8_seed2")]
    effects = []
    factors = ("data_seed", "train_seed", "graph_seed")
    for factor in factors:
        other = [f for f in factors if f != factor]
        for metric in ("test_accuracy", "noise_auroc", "ann_recall"):
            ranges = []
            for a, b in itertools.product(range(3), repeat=2):
                values = [r[metric] for r in factorial if r[other[0]] == a and r[other[1]] == b]
                assert len(values) == 3
                ranges.append(max(values) - min(values))
            marginal = [np.mean([r[metric] for r in factorial if r[factor] == level]) for level in range(3)]
            effects.append(dict(factor=factor, metric=metric, mean_conditional_range=float(np.mean(ranges)),
                                marginal_mean_sd=float(np.std(marginal, ddof=1))))
    run.csv_write(OUT / "seed_sensitivity_summary.csv", effects)
    run.dump(OUT / "audit.json", dict(status="passed", source_and_input_hashes="unchanged",
        disjoint_train_validation=True, separate_official_test_split=True,
        trajectories=9, result_cells=len(all_rows), unique_subset_fits=unique_fits,
        full_data_reference_fits=9, exact_graph_seed_invariance=True,
        raw_score_accumulation="bitwise_equal", stored_predictions_match_metrics=True))
    print((OUT / "audit.json").read_text())


if __name__ == "__main__":
    main()
